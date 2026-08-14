//! The chunk/piece/round extraction loop: turning one chunk of
//! document text into a validated `ChunkOutput`, including the
//! corrective-turn and ladder-splitting machinery.

use super::*;

/// The ladder's split rung halves a length-limited piece's cap, but
/// never below this floor: a pathological single-line document (a
/// base64 blob, minified markup) would otherwise degrade toward
/// per-character pieces. A piece at the floor that still overruns the
/// escalated budget fails the source instead.
pub(super) const MIN_SPLIT_CAP: usize = 512;

/// One extracted output alongside everything issue #199's Stage 2
/// (cross-chunk alias validation, `cross_output_issues`) needs to send
/// ONE targeted corrective turn if this output turns out to hold a
/// dangling/shadowing/conflicting alias once every output is known:
/// the conversation base that produced it (`chunk_index`/`user`, so
/// the turn is rebuilt exactly like Stage 1's rebuild-not-accumulate
/// retries) and the model's own final answer text (`answer`, replayed
/// through [`corrective_assistant_turn`] as the prior bad turn).
pub(super) struct ChunkOutput {
    pub(super) output: ModelOutput,
    /// The ORIGINAL chunk's coordinates, even for a split sub-piece
    /// (`extract_piece`'s recursion keeps `PieceContext::chunk_index`
    /// fixed) — used only for "chunk i/n" error text; several
    /// `ChunkOutput`s can share one `chunk_index` after a split.
    pub(super) chunk_index: usize,
    pub(super) user: String,
    pub(super) answer: String,
    /// ADR 0013: the path-addressed record of every item the
    /// mechanical pass removed from `output` — carried here (and
    /// through the checkpoint) so the document-level report can
    /// account for the removals of reused units too.
    pub(super) removed: Vec<String>,
}

/// How a Stage 1 corrective turn (issue #199) asks the model to try
/// again: [`corrective_message`]'s ordinary/SHORTER text for a genuine
/// parse failure, or [`corrective_validation_message`]'s path-addressed
/// text for a syntactically valid answer with validity issues. Held
/// rather than computed inline so the SAME text can be reused both to
/// build the next attempt's user turn and, on final failure, to
/// diagnose why.
pub(super) enum CorrectiveAsk {
    Syntax {
        parse_error: String,
        length_limited: bool,
    },
    Invalid {
        issues: Vec<String>,
    },
}

impl CorrectiveAsk {
    pub(super) fn user_message(&self, fact_budget: usize) -> String {
        match self {
            Self::Syntax {
                parse_error,
                length_limited,
            } => corrective_message(parse_error, *length_limited, fact_budget),
            Self::Invalid { issues } => corrective_validation_message(issues),
        }
    }
}

/// One chunk → one parsed model answer. A model that answers with
/// something other than the JSON object — or, under Stage 1 validation
/// (issue #199, `rules: Some`), a syntactically valid answer that
/// still carries path-addressed validity issues — gets up to
/// `policy.max_attempts - 1` corrective turns (1 total attempt at the
/// policy's floor). Each retry rebuilds the conversation from the
/// system/user base and appends only the most recent bad turn — never
/// the whole history — so `policy.corrective_context_cap` bounds every
/// retry alike, not just the first one. When the provider's own
/// `finish_reason` says the bad answer was cut off at the output cap,
/// the next corrective turn asks for a SHORTER answer instead of
/// repeating the same ask verbatim (see [`corrective_message`]) —
/// repeating it just reproduces the same cutoff, which is the stall
/// Issue #178 reported. At the all-defaults policy (`rules: None`,
/// lossy) this reproduces the previous fixed implementation's request
/// bodies exactly: 1st call is base only, 2nd (if needed) is base + the
/// 1st answer + the same corrective text as before.
#[allow(clippy::too_many_arguments)]
pub(super) fn extract_chunk(
    client: &ChatClient,
    system: &str,
    user: &str,
    source: &str,
    chunk_index: usize,
    policy: &CorrectionPolicy,
    fact_budget: usize,
    rules: Option<&ItemRules>,
    vocabulary: &HashSet<String>,
    sink: Option<&DiagnosticsSink>,
) -> Result<ChunkOutput, String> {
    let base = [
        serde_json::json!({"role": "system", "content": system}),
        serde_json::json!({"role": "user", "content": user}),
    ];
    let mut last_diagnosis = String::new();
    let mut prior_bad_answer: Option<String> = None;
    let mut pending: Option<CorrectiveAsk> = None;
    for attempt in 1..=policy.max_attempts {
        let mut messages = base.to_vec();
        if let Some(bad_answer) = &prior_bad_answer {
            messages.push(corrective_assistant_turn(
                bad_answer,
                policy.corrective_context_cap,
            ));
            messages.push(serde_json::json!({
                "role": "user",
                "content": pending
                    .as_ref()
                    .expect("set alongside prior_bad_answer")
                    .user_message(fact_budget),
            }));
        }
        let started = std::time::Instant::now();
        let response = match client.complete(&messages, &RequestOptions::default()) {
            Ok(response) => response,
            Err(error) => {
                if let Some(sink) = sink {
                    let message = error.to_string();
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "item",
                        chunk_index,
                        attempt,
                        max_attempts: policy.max_attempts,
                        state: match error.kind {
                            ChatFailure::Timeout => "timeout",
                            ChatFailure::Transport => "transport",
                        },
                        length_limited: false,
                        elapsed: started.elapsed(),
                        response: None,
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: None,
                        requested_max_tokens: None,
                    });
                }
                return Err(error.into());
            }
        };
        let elapsed = started.elapsed();
        match evaluate_answer(
            &response.content,
            rules,
            user_message_document(user),
            vocabulary,
        ) {
            Ok(evaluated) => {
                if let Some(sink) = sink {
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "item",
                        chunk_index,
                        attempt,
                        max_attempts: policy.max_attempts,
                        state: "stop_valid",
                        // Legacy accepts a length-terminated answer that
                        // still parses (today's behavior, unchanged) —
                        // this flag alone keeps that truncation visible
                        // in diagnostics without turning it into a
                        // failure the run never treated as one.
                        length_limited: indicates_length_limit(response.finish_reason.as_deref()),
                        elapsed,
                        response: Some(&response),
                        parse_error: None,
                        validation_issues: None,
                        removed_items: (!evaluated.removed.is_empty())
                            .then_some(evaluated.removed.as_slice()),
                        piece_bytes: None,
                        requested_max_tokens: None,
                    });
                }
                return Ok(ChunkOutput {
                    output: evaluated.output,
                    chunk_index,
                    user: user.to_string(),
                    answer: response.content,
                    removed: evaluated.removed,
                });
            }
            Err(AnswerFault::Syntax(error)) => {
                let length_limited = indicates_length_limit(response.finish_reason.as_deref());
                if let Some(sink) = sink {
                    // Diagnostics-only classification — is_empty_answer
                    // has no bearing on the corrective text below, which
                    // stays the ordinary Syntax path exactly as before.
                    let state = if is_empty_answer(&response.content) {
                        "empty"
                    } else {
                        "stop_malformed"
                    };
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "item",
                        chunk_index,
                        attempt,
                        max_attempts: policy.max_attempts,
                        state,
                        length_limited,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&error),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: None,
                        requested_max_tokens: None,
                    });
                }
                last_diagnosis = format!("the model would not produce the JSON object: {error}");
                pending = Some(CorrectiveAsk::Syntax {
                    parse_error: error,
                    length_limited,
                });
                prior_bad_answer = Some(response.content);
            }
            Err(AnswerFault::Invalid(issues)) => {
                let diagnosis = format!(
                    "the answer left {} invalid item(s) uncorrected: {}",
                    issues.len(),
                    issues.join("; ")
                );
                if let Some(sink) = sink {
                    sink.emit(DiagnosticsAttempt {
                        source,
                        stage: "item",
                        chunk_index,
                        attempt,
                        max_attempts: policy.max_attempts,
                        state: "stop_malformed",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&diagnosis),
                        validation_issues: Some(&issues),
                        removed_items: None,
                        piece_bytes: None,
                        requested_max_tokens: None,
                    });
                }
                last_diagnosis = diagnosis;
                pending = Some(CorrectiveAsk::Invalid { issues });
                prior_bad_answer = Some(response.content);
            }
        }
    }
    Err(last_diagnosis)
}

/// Issue #179's reuse guard, shared by both the legacy path
/// (`extract_chunk_or_ladder`'s `None` branch) and every level of
/// `extract_piece`'s split recursion: a unit is identified by its OWN
/// text's content hash, never `chunk_index` alone, so a split
/// sub-piece (ADR 0001 §7's Option D, which can change a document's
/// unit boundaries mid-run) is a correct, distinct cache key regardless
/// of how it came to exist.
pub(super) fn checkpointed_unit(checkpoints: &CheckpointStore, piece: &str) -> Option<ChunkOutput> {
    let unit_hash = sha256_hex(piece.as_bytes());
    checkpoints.lookup(&unit_hash).map(|unit| ChunkOutput {
        output: unit.output,
        chunk_index: unit.chunk_index,
        user: unit.user,
        answer: unit.answer,
        removed: unit.removed,
    })
}

/// Durably records one freshly-valid unit before its caller returns —
/// paired with [`checkpointed_unit`] above.
pub(super) fn record_checkpoint(
    checkpoints: &CheckpointStore,
    source: &str,
    piece: &str,
    output: &ChunkOutput,
) {
    let unit_hash = sha256_hex(piece.as_bytes());
    checkpoints.record(
        source,
        unit_hash,
        CheckpointUnit {
            chunk_index: output.chunk_index,
            output: output.output.clone(),
            user: output.user.clone(),
            answer: output.answer.clone(),
            removed: output.removed.clone(),
        },
    );
}

/// The one legacy/ladder fork. `None` is the pre-ladder loop
/// untouched — one chunk, one output, the SHORTER corrective on
/// `length` — reproducing today's requests and retries byte for byte.
/// `Some` runs the ADR 0001 §7 ladder, where one piece may fan out
/// into several outputs through the split rung.
#[allow(clippy::too_many_arguments)]
pub(super) fn extract_chunk_or_ladder(
    client: &ChatClient,
    system: &str,
    source: &str,
    chunk_index: usize,
    chunk_total: usize,
    piece: &str,
    policy: &CorrectionPolicy,
    fact_budget: usize,
    ladder: Option<&LadderConfig>,
    rules: Option<&ItemRules>,
    vocabulary: &HashSet<String>,
    sink: Option<&DiagnosticsSink>,
    checkpoints: &CheckpointStore,
) -> Result<Vec<ChunkOutput>, String> {
    match ladder {
        None => {
            if let Some(cached) = checkpointed_unit(checkpoints, piece) {
                return Ok(vec![cached]);
            }
            let user = user_message(source, chunk_index, chunk_total, piece);
            let output = extract_chunk(
                client,
                system,
                &user,
                source,
                chunk_index,
                policy,
                fact_budget,
                rules,
                vocabulary,
                sink,
            )?;
            record_checkpoint(checkpoints, source, piece, &output);
            Ok(vec![output])
        }
        Some(ladder) => {
            let context = PieceContext {
                client,
                system,
                source,
                chunk_index,
                chunk_total,
                ladder,
                policy,
                fact_budget,
                rules,
                vocabulary,
                sink,
                checkpoints,
            };
            extract_piece(&context, piece)
        }
    }
}

/// Everything one piece's ladder needs, bundled so the split
/// recursion doesn't thread eight arguments through every level.
/// `chunk_index`/`chunk_total` stay the ORIGINAL chunk's coordinates
/// all the way down: a split sub-piece is still "part K of N" of the
/// same document as far as the model is told.
pub(super) struct PieceContext<'a> {
    pub(super) client: &'a ChatClient,
    pub(super) system: &'a str,
    pub(super) source: &'a str,
    pub(super) chunk_index: usize,
    pub(super) chunk_total: usize,
    pub(super) ladder: &'a LadderConfig,
    pub(super) policy: &'a CorrectionPolicy,
    pub(super) fact_budget: usize,
    pub(super) rules: Option<&'a ItemRules>,
    pub(super) vocabulary: &'a HashSet<String>,
    pub(super) sink: Option<&'a DiagnosticsSink>,
    pub(super) checkpoints: &'a CheckpointStore,
}

/// ADR 0001 §7 for one piece: a round at the configured budget; on
/// `length`, one budget escalation — drop `max_tokens` and resend the
/// base ask NEUTRALLY, the truncated answer discarded, never replayed,
/// never salvaged as a prefix; on `length` again, split the piece and
/// run each sub-piece's ladder from the top; a piece too small to
/// split fails the source. Escalation happens at most once per piece
/// and each split halves the cap down to [`MIN_SPLIT_CAP`], so the
/// call count is bounded by piece size and `max_attempts`.
///
/// Checked against issue #179's checkpoint store before doing any of
/// that: a cache hit on THIS piece's own content hash returns
/// immediately with no model call. Since a split's sub-pieces re-enter
/// this same function with their own text, the one guard at the top
/// covers every recursion depth — a resumed run whose earlier attempt
/// split differently still reuses whatever units match today's actual
/// piece texts, and nothing else.
pub(super) fn extract_piece(
    context: &PieceContext,
    piece: &str,
) -> Result<Vec<ChunkOutput>, String> {
    if let Some(cached) = checkpointed_unit(context.checkpoints, piece) {
        return Ok(vec![cached]);
    }
    let user = user_message(
        context.source,
        context.chunk_index,
        context.chunk_total,
        piece,
    );
    let mut outcome = extract_round(
        context,
        &user,
        piece.len(),
        context.ladder.max_output_tokens,
    );
    if matches!(outcome, RoundOutcome::LengthLimited) && context.ladder.max_output_tokens.is_some()
    {
        outcome = extract_round(context, &user, piece.len(), None);
    }
    match outcome {
        RoundOutcome::Valid(chunk_output) => {
            record_checkpoint(context.checkpoints, context.source, piece, &chunk_output);
            Ok(vec![chunk_output])
        }
        RoundOutcome::Failed(message) => Err(message),
        RoundOutcome::Refusal(reason) => Err(format!(
            "the provider refused this content (finish_reason {reason}) — a policy \
             refusal is terminal; no corrective turn can change it"
        )),
        RoundOutcome::LengthLimited => {
            let cap = (piece.len() / 2).max(MIN_SPLIT_CAP);
            let sub_pieces = split_labeled_piece(piece, cap);
            if sub_pieces.len() <= 1 {
                return Err(format!(
                    "the answer still ended at the output cap for a {}-byte piece that \
                     cannot split further — failing the source rather than importing a \
                     truncated extraction",
                    piece.len()
                ));
            }
            let mut outputs = Vec::new();
            for sub_piece in &sub_pieces {
                outputs.extend(extract_piece(context, sub_piece)?);
            }
            Ok(outputs)
        }
    }
}

/// How one [`extract_round`] ended, seen from the ladder.
pub(super) enum RoundOutcome {
    Valid(ChunkOutput),
    /// The provider's metadata says this round's answer ended at the
    /// output cap — the ladder decides what changes; the round itself
    /// never re-asks under the limit it just hit.
    LengthLimited,
    /// A policy refusal, carrying the provider's spelling.
    Refusal(String),
    Failed(String),
}

/// One trip through the corrective loop at one FIXED output budget —
/// the ladder's unit. Malformed-`stop` and Stage-1-invalid answers
/// (issue #199) both get the ordinary corrective turns; the legacy
/// SHORTER ask never fires here because `length` exits the round
/// instead of becoming a prompt. An empty answer gets at most one
/// corrective in the whole round — however high `max_attempts` is —
/// then the named diagnosis.
pub(super) fn extract_round(
    context: &PieceContext,
    user: &str,
    piece_bytes: usize,
    max_tokens: Option<usize>,
) -> RoundOutcome {
    let options = RequestOptions {
        response_format: context.ladder.response_format.clone(),
        max_tokens,
    };
    let base = [
        serde_json::json!({"role": "system", "content": context.system}),
        serde_json::json!({"role": "user", "content": user}),
    ];
    let mut last_diagnosis = String::new();
    let mut prior_bad_answer: Option<String> = None;
    let mut pending: Option<CorrectiveAsk> = None;
    let mut empty_corrected = false;
    for attempt in 1..=context.policy.max_attempts {
        let mut messages = base.to_vec();
        if let Some(bad_answer) = &prior_bad_answer {
            messages.push(corrective_assistant_turn(
                bad_answer,
                context.policy.corrective_context_cap,
            ));
            messages.push(serde_json::json!({
                "role": "user",
                "content": pending
                    .as_ref()
                    .expect("set alongside prior_bad_answer")
                    .user_message(context.fact_budget),
            }));
        }
        let started = std::time::Instant::now();
        let response = match context.client.complete(&messages, &options) {
            Ok(response) => response,
            Err(error) => {
                if let Some(sink) = context.sink {
                    let message = error.to_string();
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: match error.kind {
                            ChatFailure::Timeout => "timeout",
                            ChatFailure::Transport => "transport",
                        },
                        length_limited: false,
                        elapsed: started.elapsed(),
                        response: None,
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                return RoundOutcome::Failed(error.into());
            }
        };
        let elapsed = started.elapsed();
        match classify_attempt(
            &response,
            context.rules,
            user_message_document(user),
            context.vocabulary,
        ) {
            AttemptOutcome::Valid(evaluated) => {
                if let Some(sink) = context.sink {
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: "stop_valid",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: None,
                        validation_issues: None,
                        removed_items: (!evaluated.removed.is_empty())
                            .then_some(evaluated.removed.as_slice()),
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                return RoundOutcome::Valid(ChunkOutput {
                    output: evaluated.output,
                    chunk_index: context.chunk_index,
                    user: user.to_string(),
                    answer: response.content,
                    removed: evaluated.removed,
                });
            }
            AttemptOutcome::LengthLimited => {
                if let Some(sink) = context.sink {
                    let reason = response.finish_reason.as_deref().unwrap_or("length");
                    let message = format!(
                        "the answer was cut off at the output limit (finish_reason {reason})"
                    );
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: "length_limited",
                        length_limited: true,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                return RoundOutcome::LengthLimited;
            }
            AttemptOutcome::Refusal(reason) => {
                if let Some(sink) = context.sink {
                    let message =
                        format!("the provider refused this content (finish_reason {reason})");
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: "refusal",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&message),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                return RoundOutcome::Refusal(reason);
            }
            AttemptOutcome::Empty => {
                let diagnosis = empty_answer_diagnosis();
                if let Some(sink) = context.sink {
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: "empty",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&diagnosis),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                if empty_corrected {
                    return RoundOutcome::Failed(diagnosis);
                }
                empty_corrected = true;
                last_diagnosis = diagnosis.clone();
                pending = Some(CorrectiveAsk::Syntax {
                    parse_error: diagnosis,
                    length_limited: false,
                });
                prior_bad_answer = Some(response.content);
            }
            AttemptOutcome::Malformed(error) => {
                if options.response_format.is_some() {
                    // A constrained answer that still fails validation
                    // is the provider not honoring its own contract —
                    // worth one visible line per occurrence, plus the
                    // diagnostics record below (issue #200).
                    eprintln!(
                        "taguru: extract: {}: provider non-conformance: the answer \
                         violated the requested response_format ({error})",
                        context.source
                    );
                }
                if let Some(sink) = context.sink {
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: "stop_malformed",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&error),
                        validation_issues: None,
                        removed_items: None,
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                last_diagnosis = format!("the model would not produce the JSON object: {error}");
                pending = Some(CorrectiveAsk::Syntax {
                    parse_error: error,
                    length_limited: false,
                });
                prior_bad_answer = Some(response.content);
            }
            AttemptOutcome::Invalid(issues) => {
                // NOT provider non-conformance: model_output_json_schema's
                // own doc comment names business rules (weight's
                // magnitude, alias resolution, byte caps) the wire
                // schema never encodes, so a schema-constrained answer
                // can carry these without the provider having broken
                // its response_format contract.
                let diagnosis = format!(
                    "the answer left {} invalid item(s) uncorrected: {}",
                    issues.len(),
                    issues.join("; ")
                );
                if let Some(sink) = context.sink {
                    sink.emit(DiagnosticsAttempt {
                        source: context.source,
                        stage: "item",
                        chunk_index: context.chunk_index,
                        attempt,
                        max_attempts: context.policy.max_attempts,
                        state: "stop_malformed",
                        length_limited: false,
                        elapsed,
                        response: Some(&response),
                        parse_error: Some(&diagnosis),
                        validation_issues: Some(&issues),
                        removed_items: None,
                        piece_bytes: Some(piece_bytes),
                        requested_max_tokens: max_tokens,
                    });
                }
                last_diagnosis = diagnosis;
                pending = Some(CorrectiveAsk::Invalid { issues });
                prior_bad_answer = Some(response.content);
            }
        }
    }
    RoundOutcome::Failed(last_diagnosis)
}

/// One attempt's §7 state, classified from provider metadata BEFORE
/// any parse-level interpretation.
pub(super) enum AttemptOutcome {
    Valid(EvaluatedAnswer),
    Malformed(String),
    /// Issue #199: syntactically valid JSON that still carries
    /// path-addressed Stage 1 validity issues.
    Invalid(Vec<String>),
    LengthLimited,
    Refusal(String),
    Empty,
}

/// A `length`-terminated answer is length-limited even when its
/// prefix happens to parse — a valid prefix of a cut-off extraction
/// is exactly the "deleted-subset called complete" ADR 0001 forbids.
/// Refusals are terminal before any parsing; an empty answer is named
/// before serde ever sees it. `rules: None` (lossy mode) never
/// produces `Invalid` — see [`evaluate_answer`].
pub(super) fn classify_attempt(
    response: &ChatCompletion,
    rules: Option<&ItemRules>,
    document: &str,
    vocabulary: &HashSet<String>,
) -> AttemptOutcome {
    let finish_reason = response.finish_reason.as_deref();
    if indicates_length_limit(finish_reason) {
        return AttemptOutcome::LengthLimited;
    }
    if let Some(reason) = finish_reason
        && indicates_refusal(reason)
    {
        return AttemptOutcome::Refusal(reason.to_string());
    }
    if is_empty_answer(&response.content) {
        return AttemptOutcome::Empty;
    }
    match evaluate_answer(&response.content, rules, document, vocabulary) {
        Ok(evaluated) => AttemptOutcome::Valid(evaluated),
        Err(AnswerFault::Syntax(error)) => AttemptOutcome::Malformed(error),
        Err(AnswerFault::Invalid(issues)) => AttemptOutcome::Invalid(issues),
    }
}

/// Whether `finish_reason` says the provider refused to answer on
/// policy grounds: `content_filter` is the OpenAI-compatible
/// spelling; `refusal` is Anthropic's `stop_reason`, met through
/// pass-through bridges exactly like `max_tokens` in
/// [`indicates_length_limit`]. Terminal — a corrective turn cannot
/// argue with a policy.
pub(super) fn indicates_refusal(finish_reason: &str) -> bool {
    matches!(finish_reason, "content_filter" | "refusal")
}

/// Whether a chat completion's `finish_reason` means the provider cut
/// the answer off at its own output-length cap — the pattern behind
/// Issue #178's stalls: one huge truncated answer, replayed back in
/// full, then re-asked for the very length the model just proved it
/// couldn't fit in. `"length"` is the OpenAI-compatible (and Ollama
/// `done_reason`) spelling; `"max_tokens"` is Anthropic's `stop_reason`
/// for the same cutoff, which the SDK twins meet through LangChain
/// metadata and this producer can meet through pass-through bridges.
/// Any other reason (`"stop"`, `None`, a provider-specific value) is
/// left to the ordinary corrective text.
pub(super) fn indicates_length_limit(finish_reason: Option<&str>) -> bool {
    matches!(finish_reason, Some("length" | "max_tokens"))
}

/// The corrective turn's user-facing ask, addressed to `parse_error`.
/// When `length_limited` is false this is byte-for-byte today's fixed
/// text. When true — the provider says the prior answer was cut off at
/// its output cap — the ask changes from "try again" to "try again
/// shorter," naming `fact_budget` when the run has one, since repeating
/// the same-length ask just reproduces the same cutoff.
pub(super) fn corrective_message(
    parse_error: &str,
    length_limited: bool,
    fact_budget: usize,
) -> String {
    if !length_limited {
        return format!(
            "That was not the single JSON object asked for ({parse_error}). \
             Answer again with only the JSON object."
        );
    }
    let budget_hint = if fact_budget > 0 {
        format!(" Keep it to at most {fact_budget} association(s) total.")
    } else {
        String::new()
    };
    format!(
        "That was not the single JSON object asked for ({parse_error}) — it looks like \
         the answer was cut off at the output limit. Answer again with a SHORTER JSON \
         object: fewer associations, shorter names and values.{budget_hint}"
    )
}

/// Cap on how many issues one corrective-validation message lists: a
/// pathological answer with hundreds of malformed items must not make
/// one turn's prompt balloon without bound — the model gets the worst
/// offenders (in the same associations→aliases→questions walk order
/// [`interpret_model_output`] collects them) and a count of the rest.
/// Shared with `api.rs`'s own issue-listing cap on a rejected HTTP
/// write (issue #622 finding 8: a compiler-linked constant, not two
/// independently defined `20`s that could silently disagree).
pub(super) use crate::api::MAX_LISTED_ISSUES;

/// The corrective turn's ask when an answer parsed as JSON but failed
/// Stage 1/Stage 2 validation (issue #199, ADR 0001 §8 bucket 2): name
/// every issue by its path, then ask for the complete corrected
/// object — preserve every item, correct rather than delete, add
/// nothing that wasn't already there, JSON only. Distinct from
/// [`corrective_message`], which stays reserved for a genuine parse
/// failure (`AnswerFault::Syntax`, ADR 0001 §7's `STOP_MALFORMED`
/// syntax half); this is the "valid JSON, invalid extraction" half,
/// and its wording is the cross-language corrective-text baseline
/// #180/#181 mirror byte for byte.
pub(super) fn corrective_validation_message(issues: &[String]) -> String {
    let mut listed = String::new();
    for issue in issues.iter().take(MAX_LISTED_ISSUES) {
        listed.push_str("\n- ");
        listed.push_str(issue);
    }
    let remainder = issues.len().saturating_sub(MAX_LISTED_ISSUES);
    if remainder > 0 {
        listed.push_str(&format!("\n… and {remainder} more issue(s)"));
    }
    format!(
        "That was valid JSON but not a valid extraction ({} issue(s)):{listed}\n\
         Answer again with the complete corrected JSON object: keep every item, correct the \
         fields listed above instead of deleting their items, add nothing that was not already \
         there, and answer with only the JSON object.",
        issues.len()
    )
}

/// How one attempt's answer failed Stage 1 (issue #199): a genuine
/// parse failure (today's [`corrective_message`] path, unchanged) or a
/// syntactically valid answer that still carries path-addressed
/// validity issues (the new [`corrective_validation_message`] path).
#[derive(Debug)]
pub(super) enum AnswerFault {
    Syntax(String),
    Invalid(Vec<String>),
}

/// A strict-mode answer that passed the gate: the output with
/// mechanically-removed items already gone and their path-addressed
/// records (ADR 0013). Lossy mode always returns `removed` empty —
/// its contract is that nothing is validated at all.
pub(super) struct EvaluatedAnswer {
    pub(super) output: ModelOutput,
    pub(super) removed: Vec<String>,
}

/// The Stage 1 gate every corrective-loop entry point calls instead of
/// [`parse_model_output`] directly: parse, then — when `rules` is
/// `Some`, i.e. this run is not `--lossy` — run the mechanical pass
/// (ADR 0013): items that cannot import as answered are removed with
/// accounting, and only the issues removal cannot judge (a present but
/// wrong-typed or out-of-range value) still fail the answer into a
/// corrective turn. `document` is the document text this answer replied
/// to — the occurrence check's haystack. `rules: None` (lossy mode)
/// parses only and discards whatever `interpret_model_output` would
/// have flagged, reproducing the pre-#199 behavior byte for byte: the
/// same request goes out, the same answer is accepted, `merge()` alone
/// decides what survives.
pub(super) fn evaluate_answer(
    content: &str,
    rules: Option<&ItemRules>,
    document: &str,
    vocabulary: &HashSet<String>,
) -> Result<EvaluatedAnswer, AnswerFault> {
    let value = candidate_json(content).map_err(AnswerFault::Syntax)?;
    match rules {
        None => {
            let lenient_rules = ItemRules {
                paragraph_count: usize::MAX,
                questions_requested: true,
            };
            let (output, _issues) = interpret_model_output(&value, &lenient_rules);
            Ok(EvaluatedAnswer {
                output,
                removed: Vec::new(),
            })
        }
        Some(rules) => {
            let evaluation = mechanical_interpret(&value, rules, document, vocabulary);
            if evaluation.issues.is_empty() {
                Ok(EvaluatedAnswer {
                    output: evaluation.output,
                    removed: evaluation.removed,
                })
            } else {
                // The whole answer goes back for correction; this
                // attempt's removals die with it — the accepted
                // attempt's own mechanical pass is the one that counts.
                Err(AnswerFault::Invalid(evaluation.issues))
            }
        }
    }
}

/// The corrective turn's replay of the model's own prior bad answer,
/// shaped by `cap` (`Run::correction`'s `corrective_context_cap`):
/// `None` replays it in full, `Some(0)` omits it behind a placeholder,
/// `Some(n)` truncates it to `n` bytes at a char boundary with a
/// trailing marker. The turn itself is always present at some content
/// — dropping it instead of placeholding it would leave two
/// consecutive `user` messages, which most chat APIs reject.
pub(super) fn corrective_assistant_turn(content: &str, cap: Option<usize>) -> serde_json::Value {
    let text = match cap {
        None => content.to_string(),
        Some(0) => "[omitted: not the requested JSON object]".to_string(),
        Some(n) if n >= content.len() => content.to_string(),
        Some(n) => format!(
            "{}… [truncated to {n} bytes]",
            &content[..floor_char_boundary(content, n)]
        ),
    };
    serde_json::json!({"role": "assistant", "content": text})
}
