//! The OpenAI-compatible `/chat/completions` client (`ChatClient`) and
//! the request/response shapes and error classification around it.

use super::*;

/// Total attempts (1 initial + retries) at one chat completion before
/// a chunk fails. `--parallel` multiplies 429 pressure, so this leans
/// toward more attempts than a purely sequential client would need.
pub(super) const RETRY_ATTEMPTS: usize = 4;

/// OpenAI-compatible `/chat/completions` client — deliberately the
/// same protocol choice as embeddings: one wire shape here, vendor
/// APIs bridged outside (docs/bedrock.html shows how). Crate-visible
/// because `taguru communities` reuses it (same env vars, same retry
/// discipline) for its summary prompts.
pub(crate) struct ChatClient {
    pub(super) url: String,
    pub(super) model: String,
    pub(super) api_key: Option<String>,
    pub(super) agent: ureq::Agent,
}

/// ADR 0023 §3.2: one extraction completion's identity — the run it
/// belongs to and its 1-based position among the completions that
/// run issued. Equality of two `AttemptRef`s means the same HTTP
/// conversation, wherever the two records live (trace file,
/// diagnostics sidecar, checkpoint).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, Deserialize)]
pub(crate) struct AttemptRef {
    pub(crate) run_id: String,
    pub(crate) attempt_seq: u64,
}

/// 16 hex characters from the OS random source — a run id (ADR 0023).
pub(super) fn mint_run_id() -> String {
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("the OS random source must work");
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Whether [`ChatClient::complete`] gave up before the provider ever
/// answered (`Timeout`) or after some other transport-level trouble —
/// connection refused, a malformed/oversized body, an HTTP error status
/// that exhausted its retries (`Transport`). ADR 0001 §7 draws exactly
/// this line between its `TIMEOUT` and `TRANSPORT` terminal states; the
/// diagnostics sink (issue #200) is the only reader — every existing
/// caller still just formats [`ChatError`] with `{error}`, unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatFailure {
    Timeout,
    Transport,
}

/// [`ChatClient::complete`]'s error: the same message every caller has
/// always surfaced via `Display`, plus the [`ChatFailure`]
/// classification issue #200's diagnostics sink needs. `From<ChatError>
/// for String` keeps every `?`-based caller compiling exactly as it did
/// when `complete` returned `Result<ChatCompletion, String>`.
#[derive(Debug)]
pub(crate) struct ChatError {
    pub(crate) kind: ChatFailure,
    pub(super) message: String,
    /// ADR 0029 (#791): failed tries before this error was returned
    /// (the first try counts as 0) — [`ChatClient::complete`] stamps
    /// it at every return; [`ChatError::new`] starts it at 0.
    pub(crate) transport_retries: usize,
    /// ADR 0031 §3.2 (#823): the original attempt this error reused,
    /// when it reconstructs a recorded `timeout`/`transport` outcome
    /// instead of a live failure — mirrors [`ChatCompletion::replayed_from`];
    /// `None` at every other construction site, all of them
    /// [`ChatError::new`].
    pub(crate) replayed_from: Option<AttemptRef>,
}

impl ChatError {
    pub(super) fn new(kind: ChatFailure, message: String) -> Self {
        Self {
            kind,
            message,
            transport_retries: 0,
            replayed_from: None,
        }
    }
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<ChatError> for String {
    fn from(error: ChatError) -> Self {
        error.message
    }
}

/// Classifies the error `ureq::Agent::send`/`call` itself raised (never
/// reaching an HTTP response) as TIMEOUT or TRANSPORT. ureq surfaces a
/// deadline both as its own `Timeout` variant and, for some transports,
/// as an `Io` error carrying `ErrorKind::TimedOut` — both read as the
/// same ADR 0001 §7 state.
pub(super) fn classify_send_error(error: &ureq::Error) -> ChatFailure {
    match error {
        ureq::Error::Timeout(_) => ChatFailure::Timeout,
        ureq::Error::Io(io_error) if io_error.kind() == std::io::ErrorKind::TimedOut => {
            ChatFailure::Timeout
        }
        _ => ChatFailure::Transport,
    }
}

/// Same classification as [`classify_send_error`], for an `io::Error`
/// hit while reading an already-established response body.
pub(super) fn classify_io_error(error: &std::io::Error) -> ChatFailure {
    if error.kind() == std::io::ErrorKind::TimedOut {
        ChatFailure::Timeout
    } else {
        ChatFailure::Transport
    }
}

/// Token counts a provider reported for one completion, translated from
/// the OpenAI-compatible wire names (`prompt_tokens`/`completion_tokens`/
/// `total_tokens`) to the vocabulary `taguru-langchain`'s
/// `ProviderMetadata` already uses (`input_tokens`/`output_tokens`/
/// `total_tokens`) — the one place that translation happens. `None`
/// fields mean the response's `usage` object omitted them.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TokenUsage {
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
}

/// One chat completion's assistant text plus the provider's own account
/// of why it stopped — `finish_reason` straight from the response body
/// (`"length"` means the output hit the token cap mid-answer; `None`
/// covers providers that omit the field). `usage` is `None` when the
/// response carries no `usage` object at all (see [`TokenUsage`]).
pub(crate) struct ChatCompletion {
    pub(crate) content: String,
    pub(crate) finish_reason: Option<String>,
    pub(crate) usage: Option<TokenUsage>,
    /// ADR 0029 (#791): how many transport-layer tries failed before
    /// this completion arrived — 0 for a clean first try; the
    /// `transport`/429/5xx retries ADR 0001 §10 folds into one
    /// attempt, now counted on it.
    pub(crate) transport_retries: usize,
    /// ADR 0031 §3.2 (#823): the original attempt this completion
    /// reused, when it came from `--replay` instead of a live call —
    /// `None` for every completion `ChatClient::complete` itself
    /// builds (the only other construction site). `Completions::complete`
    /// is the sole caller that can ever see `Some` here, since only it
    /// consults a `ReplayIndex` at all.
    pub(crate) replayed_from: Option<AttemptRef>,
}

/// The optional OpenAI-compatible parameters one completion carries
/// beyond the fixed `{model, temperature, messages}` base. Per-call
/// rather than per-client: the extraction ladder changes `max_tokens`
/// between attempts of one piece, and `--parallel` shares one client
/// across workers. The default adds nothing — [`build_chat_body`]'s
/// output is then byte-for-byte the pre-ladder body, which `taguru
/// communities` (the other caller) relies on.
#[derive(Default, Clone)]
pub(crate) struct RequestOptions {
    pub(crate) response_format: Option<serde_json::Value>,
    pub(crate) max_tokens: Option<usize>,
    /// ADR 0020 (#762): return the first `Timeout` instead of
    /// retrying it at the same size — the §7 ladder's own next step
    /// (split the piece) IS the retry, and four same-size attempts at
    /// a piece the hardware cannot finish in time only multiply the
    /// cost. Transport failures, 429, and 5xx keep their retries
    /// either way. `false` (the default, and every non-ladder caller)
    /// is byte-for-byte the pre-0020 retry discipline.
    pub(crate) fail_fast_on_timeout: bool,
}

/// The request body [`ChatClient::complete`] sends. serde_json's maps
/// order keys alphabetically (this crate does not enable
/// `preserve_order`), so the base three keys serialize exactly as they
/// always have and the optional keys slot in only when set.
pub(super) fn build_chat_body(
    model: &str,
    messages: &[serde_json::Value],
    options: &RequestOptions,
) -> String {
    let mut body = serde_json::json!({
        "model": model,
        "temperature": 0,
        "messages": messages,
    });
    if let Some(format) = &options.response_format {
        body["response_format"] = format.clone();
    }
    if let Some(max_tokens) = options.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    body.to_string()
}

impl ChatClient {
    pub(crate) fn from_env() -> Result<Self, String> {
        let url = std::env::var("TAGURU_EXTRACT_URL").map_err(|_| {
            "TAGURU_EXTRACT_URL is not set — extract needs an OpenAI-compatible \
             /chat/completions endpoint (docs/extract.html)"
                .to_string()
        })?;
        let model = std::env::var("TAGURU_EXTRACT_MODEL")
            .map_err(|_| "TAGURU_EXTRACT_MODEL is not set".to_string())?;
        let timeout = crate::env::env_number("TAGURU_EXTRACT_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS);
        // 4xx/5xx answers carry a body `complete` quotes in its error
        // messages, so have them come back as responses, not errors.
        let mut config = ureq::Agent::config_builder().http_status_as_error(false);
        if timeout > 0 {
            config = config.timeout_global(Some(Duration::from_secs(timeout as u64)));
        }
        Ok(Self {
            url,
            model,
            api_key: std::env::var("TAGURU_EXTRACT_API_KEY").ok(),
            agent: config.build().into(),
        })
    }

    /// One chat completion, returning the assistant text alongside the
    /// provider's `finish_reason`. Transient trouble — transport
    /// errors, 429, 5xx — is retried up to
    /// [`RETRY_ATTEMPTS`] times total, waiting [`jittered_backoff`]
    /// between attempts; a 429 that carries `Retry-After` uses that
    /// delay instead, verbatim. Everything else is the caller's
    /// problem.
    pub(crate) fn complete(
        &self,
        messages: &[serde_json::Value],
        options: &RequestOptions,
    ) -> Result<ChatCompletion, ChatError> {
        let body = build_chat_body(&self.model, messages, options);
        let mut last: Option<ChatError> = None;
        for attempt in 0..RETRY_ATTEMPTS {
            let mut request = self
                .agent
                .post(&self.url)
                .header("Content-Type", "application/json");
            if let Some(key) = &self.api_key {
                request = request.header("Authorization", format!("Bearer {key}"));
            }
            // The server's own instruction wins over a computed guess —
            // only ever consulted on 429, and only ever shortens or
            // lengthens THIS wait, never dilutes with jitter. `None`
            // means "use the computed jittered backoff instead."
            let retry_after = match request.send(&body) {
                // Read/parse/shape failures here go through the SAME
                // retry bookkeeping every other branch uses (`last =
                // Some(..)`, loop around) instead of `?`-ing straight out
                // of `complete` — a body that stops streaming or a
                // truncated/garbled reply on an otherwise-200 response is
                // exactly the transient trouble this loop exists to
                // absorb (see this fn's own doc), and `parse_chat_completion`
                // already tags every one of its failures `Timeout` or
                // `Transport` to say so.
                Ok(response) if response.status().as_u16() < 400 => {
                    match parse_chat_completion(response.into_body()) {
                        Ok(mut completion) => {
                            completion.transport_retries = attempt;
                            return Ok(completion);
                        }
                        Err(mut error) => {
                            if options.fail_fast_on_timeout && error.kind == ChatFailure::Timeout {
                                error.transport_retries = attempt;
                                return Err(error);
                            }
                            last = Some(error);
                            None
                        }
                    }
                }
                Ok(response) => {
                    let code = response.status().as_u16();
                    let retry_after = (code == 429)
                        .then(|| {
                            response
                                .headers()
                                .get("retry-after")
                                .and_then(|value| value.to_str().ok())
                                .and_then(parse_retry_after)
                        })
                        .flatten();
                    let error_body =
                        read_capped_chat_body(response.into_body()).unwrap_or_default();
                    let error = ChatError::new(
                        ChatFailure::Transport,
                        format!(
                            "chat endpoint answered {code}: {}",
                            snippet(&String::from_utf8_lossy(&error_body))
                        ),
                    );
                    if code != 429 && code < 500 {
                        let mut error = error;
                        error.transport_retries = attempt;
                        return Err(error);
                    }
                    last = Some(error);
                    retry_after
                }
                Err(error) => {
                    let mut error = ChatError::new(
                        classify_send_error(&error),
                        format!("chat request failed: {error}"),
                    );
                    if options.fail_fast_on_timeout && error.kind == ChatFailure::Timeout {
                        error.transport_retries = attempt;
                        return Err(error);
                    }
                    last = Some(error);
                    None
                }
            };
            if attempt + 1 < RETRY_ATTEMPTS {
                std::thread::sleep(
                    retry_after.unwrap_or_else(|| jittered_backoff(attempt as u32 + 1)),
                );
            }
        }
        let last = last.expect("RETRY_ATTEMPTS >= 1, so the loop set this at least once");
        let mut error = ChatError::new(
            last.kind,
            format!("after {RETRY_ATTEMPTS} attempts: {}", last.message),
        );
        error.transport_retries = RETRY_ATTEMPTS - 1;
        Err(error)
    }
}

/// Parses a successful (status < 400) chat completion body: read, JSON
/// parse, then pull `content`/`finish_reason`/`usage` out of the
/// OpenAI-shaped envelope. Split out of [`ChatClient::complete`] so its
/// failures — a body that stops streaming mid-read, a truncated or
/// garbled JSON reply, one missing `choices[0].message.content` — return
/// a plain [`ChatError`] instead of `?`-ing out of `complete` itself,
/// which would skip its retry ladder entirely for exactly the kind of
/// one-off transient trouble that ladder exists to absorb.
pub(super) fn parse_chat_completion(body: ureq::Body) -> Result<ChatCompletion, ChatError> {
    let bytes = read_capped_chat_body(body)?;
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        ChatError::new(
            ChatFailure::Transport,
            format!("chat response unreadable: {error}"),
        )
    })?;
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| {
            ChatError::new(
                ChatFailure::Transport,
                "chat response carries no assistant text".to_string(),
            )
        })?;
    let finish_reason = parsed["choices"][0]["finish_reason"]
        .as_str()
        .map(str::to_string);
    let usage = parsed
        .get("usage")
        .filter(|value| value.is_object())
        .map(|usage| TokenUsage {
            input_tokens: usage["prompt_tokens"].as_u64(),
            output_tokens: usage["completion_tokens"].as_u64(),
            total_tokens: usage["total_tokens"].as_u64(),
        });
    Ok(ChatCompletion {
        content,
        finish_reason,
        usage,
        transport_retries: 0,
        replayed_from: None,
    })
}
