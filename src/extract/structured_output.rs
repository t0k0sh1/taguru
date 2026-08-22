//! ADR 0001 §6's structured-output ladder: resolving and probing a
//! `response_format`, retry backoff, and the capped chat-body reader
//! the probes and real requests share.

use super::*;

/// Full-jitter exponential backoff between attempts: the n-th retry
/// sleeps `random(0, min(RETRY_MAX_BACKOFF, RETRY_BASE_BACKOFF *
/// 2^(n-1)))` (see [`jittered_backoff`]). A 429 carrying `Retry-After`
/// uses that instead, clamped to the same ceiling.
pub(super) const RETRY_BASE_BACKOFF: Duration = Duration::from_secs(1);

pub(super) const RETRY_MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Output budget for the startup capability probes — small enough to
/// bound what a rambling endpoint can spend, large enough that a
/// compliant answer to the tiny probe ask is never cut off (a
/// `length`-terminated probe reads as "rung not verified").
pub(super) const PROBE_MAX_TOKENS: usize = 256;

/// What the run's structured-output rung resolved to, reported once on
/// stderr so a log shows which rung actually carried the run. Pinned
/// modes trust the operator (a backend that rejects the parameter
/// surfaces its 400 on the first document); `auto` verifies against
/// the live endpoint before relying on anything, because a backend may
/// accept a parameter without honoring it (ADR 0001 §6).
pub(super) fn resolve_response_format(
    client: &ChatClient,
    mode: StructuredOutputMode,
) -> Option<serde_json::Value> {
    match mode {
        StructuredOutputMode::Off => None,
        StructuredOutputMode::JsonSchema => {
            eprintln!("taguru: extract: structured output: json_schema (pinned)");
            Some(json_schema_response_format())
        }
        StructuredOutputMode::JsonObject => {
            eprintln!("taguru: extract: structured output: json_object (pinned)");
            Some(json_object_response_format())
        }
        StructuredOutputMode::Auto => match probe_structured_output(client) {
            ProbeVerdict::JsonSchema => {
                eprintln!("taguru: extract: structured output: json_schema (probe verified)");
                Some(json_schema_response_format())
            }
            ProbeVerdict::JsonObject => {
                eprintln!(
                    "taguru: extract: structured output: json_object \
                     (the json_schema probe failed)"
                );
                Some(json_object_response_format())
            }
            ProbeVerdict::Prompted => {
                eprintln!(
                    "taguru: extract: structured output: prompted JSON only \
                     (both probes failed)"
                );
                None
            }
        },
    }
}

pub(super) enum ProbeVerdict {
    JsonSchema,
    JsonObject,
    Prompted,
}

/// One startup probe per rung, sending EXACTLY the `response_format`
/// extraction will send — a probe that passes proves the real request
/// shape is both accepted and honored, not a lookalike. The
/// json_schema ask invites prose and never says "JSON": only an
/// endpoint that actually constrains decoding answers it with the
/// canonical `{associations, aliases}` object. The json_object ask
/// names json (OpenAI's json_object mode refuses requests that
/// don't), so it only verifies that the answer is JSON at all — which
/// is all that rung promises. Transport errors, 400s, truncation, and
/// wrong-shaped answers all read the same way: rung not verified,
/// fall one down.
pub(super) fn probe_structured_output(client: &ChatClient) -> ProbeVerdict {
    let ask = |content: &str| {
        [
            serde_json::json!({"role": "system", "content": "You answer questions."}),
            serde_json::json!({"role": "user", "content": content}),
        ]
    };
    let schema_options = RequestOptions {
        fail_fast_on_timeout: false,
        response_format: Some(json_schema_response_format()),
        max_tokens: Some(PROBE_MAX_TOKENS),
    };
    let schema_probe = ask("In one short sentence, name the color of a clear daytime sky.");
    if let Ok(response) = client.complete(&schema_probe, &schema_options)
        && !indicates_length_limit(response.finish_reason.as_deref())
        && conforms_to_model_output_shape(&response.content)
    {
        return ProbeVerdict::JsonSchema;
    }
    let object_options = RequestOptions {
        fail_fast_on_timeout: false,
        response_format: Some(json_object_response_format()),
        max_tokens: Some(PROBE_MAX_TOKENS),
    };
    let object_probe = ask("Reply with a json object naming the color of a clear daytime sky.");
    if let Ok(response) = client.complete(&object_probe, &object_options)
        && !indicates_length_limit(response.finish_reason.as_deref())
        && serde_json::from_str::<serde_json::Value>(strip_fences(response.content.trim()))
            .map(|value| value.is_object())
            .unwrap_or(false)
    {
        return ProbeVerdict::JsonObject;
    }
    ProbeVerdict::Prompted
}

/// Whether a probe answer proves schema-constrained decoding: the
/// canonical schema requires `associations` and `aliases`, so a
/// constrained endpoint cannot answer the prose-inviting probe with
/// anything else. JSON of some other shape (what a json_object-only
/// endpoint would send) and prose both fail.
pub(super) fn conforms_to_model_output_shape(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(strip_fences(content.trim()))
        .map(|value| value["associations"].is_array() && value["aliases"].is_array())
        .unwrap_or(false)
}

/// The OpenAI-compatible `response_format` carrying the canonical
/// schema — ADR 0001's mechanism B in the exact shape its experiment
/// measured against Ollama's `/v1` wire. `strict` is requested
/// honestly: a strictly-validating backend that cannot express the
/// canonical schema's optional `weight`/`paragraph` answers 400
/// instead of silently weakening the constraint, and `auto` then
/// falls one rung (docs/extract.html notes this for OpenAI's strict
/// mode).
///
/// `pub(crate)` so `benchmark` can hash this exact canonical
/// `response_format` into `manifest.json`'s `extraction_settings.
/// schema_sha256` (ADR 0003 §9.1) instead of re-deriving the schema
/// shape a second time.
pub(crate) fn json_schema_response_format() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "ModelOutput",
            "strict": true,
            "schema": model_output_json_schema(),
        }
    })
}

pub(super) fn json_object_response_format() -> serde_json::Value {
    serde_json::json!({"type": "json_object"})
}

/// Full-jitter exponential backoff for the n-th retry (n ≥ 1):
/// `random(0, min(RETRY_MAX_BACKOFF, RETRY_BASE_BACKOFF * 2^(n-1)))` —
/// full jitter spreads retries out instead of having every stalled
/// worker wake up at exactly the same instant.
pub(super) fn jittered_backoff(retry_number: u32) -> Duration {
    let factor = 1u32
        .checked_shl(retry_number.saturating_sub(1))
        .unwrap_or(u32::MAX);
    let exponential = RETRY_BASE_BACKOFF
        .saturating_mul(factor)
        .min(RETRY_MAX_BACKOFF);
    random_duration_up_to(exponential)
}

/// A uniformly random duration in `[0, cap]`, drawn the same way
/// `oauth.rs` draws its CSRF/PKCE bytes — no new dependency for jitter.
pub(super) fn random_duration_up_to(cap: Duration) -> Duration {
    let cap_nanos = cap.as_nanos().min(u64::MAX as u128) as u64;
    if cap_nanos == 0 {
        return Duration::ZERO;
    }
    let mut bytes = [0u8; 8];
    getrandom::fill(&mut bytes).expect("the OS random source must work");
    Duration::from_nanos(u64::from_le_bytes(bytes) % cap_nanos)
}

/// A `Retry-After` value as delta-seconds, clamped to
/// `RETRY_MAX_BACKOFF` so a huge or adversarial value cannot stall a
/// run indefinitely. HTTP-date values are not recognized — like the
/// rest of this codebase, extract avoids pulling in a datetime-parsing
/// dependency for the one header that would otherwise need one.
pub(super) fn parse_retry_after(value: &str) -> Option<Duration> {
    let seconds: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(seconds).min(RETRY_MAX_BACKOFF))
}

/// Reads a chat endpoint's response body capped at
/// [`MAX_CHAT_RESPONSE_BYTES`], so a misbehaving or misaddressed
/// endpoint cannot hand `complete` an unbounded buffer on either the
/// success or the error-diagnostic path.
pub(super) fn read_capped_chat_body(body: ureq::Body) -> Result<Vec<u8>, ChatError> {
    use std::io::Read;
    let mut buffer = Vec::new();
    body.into_reader()
        .take(MAX_CHAT_RESPONSE_BYTES + 1)
        .read_to_end(&mut buffer)
        .map_err(|error| {
            ChatError::new(
                classify_io_error(&error),
                format!("chat response unreadable: {error}"),
            )
        })?;
    if buffer.len() as u64 > MAX_CHAT_RESPONSE_BYTES {
        return Err(ChatError::new(
            ChatFailure::Transport,
            format!(
                "chat response is larger than {MAX_CHAT_RESPONSE_BYTES} bytes; refusing to \
                 buffer it"
            ),
        ));
    }
    Ok(buffer)
}

/// Provider error bodies can run long; a line is enough to act on.
pub(super) fn snippet(text: &str) -> String {
    let trimmed = text.trim();
    let cut = floor_char_boundary(trimmed, 200);
    if cut < trimmed.len() {
        format!("{}…", &trimmed[..cut])
    } else {
        trimmed.to_string()
    }
}

/// Chat completion response cap. ureq's own `read_to_string`/`read_json`
/// already cap at 10 MiB, but that ceiling is undocumented at the call
/// site and unconfigurable — read through an explicit one instead, same
/// treatment as `embedding.rs`'s `HttpEmbeddings::decode`. 16 MiB clears
/// a legitimate answer to one [`CHUNK_BYTES`] chunk (associations plus,
/// with `--questions`, per-paragraph search questions) many times over,
/// while still bounding a misbehaving or misaddressed endpoint's buffer.
pub(super) const MAX_CHAT_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
