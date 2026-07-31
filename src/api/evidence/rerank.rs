//! The optional evidence reranker (#307, ADR 0006 §12): a
//! provider-neutral boundary that may only *reorder* the already fused,
//! deduplicated, near-duplicate-suppressed candidate pool [`select`]
//! hands it, immediately before diversity-aware admission. Absent
//! config, or on any failure — unreachable provider, timeout, circuit
//! open, a malformed or non-permutation response — selection degrades
//! to the deterministic §7 order with a machine-readable
//! `plan.reranker.reason` token; it never turns into a non-2xx error
//! (ADR 0006 §11).
//!
//! [`EvidenceReranker`] mirrors [`crate::embedding::EmbeddingProvider`]'s
//! shape, not the batch-oriented `ChatClient`'s (ADR 0006 §12: "this is
//! an interactive read path serving one caller's request
//! synchronously"). [`HttpReranker`] is the one real implementation, a
//! Cohere/Jina-compatible `POST /rerank` client
//! (`{model, query, documents, top_n}` in, `{results: [{index,
//! relevance_score}]}` out) — the same "any OpenAI/Cohere-shaped
//! adapter plugs in" posture [`crate::embedding::HttpEmbeddings`] takes
//! for embeddings, so TEI, Infinity, vLLM, Cohere, and Jina all speak
//! to it unmodified. Configured via `TAGURU_RERANK_URL` /
//! `TAGURU_RERANK_MODEL` / `TAGURU_RERANK_API_KEY` /
//! `TAGURU_RERANK_TIMEOUT_SECS`; absent config disables the tier
//! entirely — no credential or network access is required by default.
//!
//! **Privacy** (ADR 0006 §12): candidate text is sent to a configured
//! reranker provider and nowhere else. [`RerankCandidate`] deliberately
//! does not derive `Debug`, and every failure this module can produce
//! carries only a fixed `&'static str` reason token
//! ([`RerankFailure::reason`]) — never a provider's own response body,
//! URL, or the `TAGURU_RERANK_API_KEY` value. The only reranker-
//! identifying information that reaches the response or metrics is the
//! model identity string (`RerankerPlan.model`).

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::breaker::ProviderBreaker;

use super::{AssociationOut, CandidatePayload, FusedCandidate};

/// The request-side `rerank` object (ADR 0006 §5.1): every field
/// optional. `model` pins the caller to a specific provider identity —
/// a request naming a model that does not match the configured
/// provider's own [`EvidenceReranker::model`] degrades with
/// `reason: "model_mismatch"` rather than silently reranking against a
/// different model than the caller asked for.
#[derive(Debug, Default, Deserialize)]
pub struct RerankRequest {
    pub model: Option<String>,
}

/// ADR 0006 §12's `plan.reranker`: whether a provider is configured at
/// all, whether it actually ran for this call, its model identity on
/// success, and a machine-readable reason token on any degrade.
/// `reason` is always one of the fixed tokens this module and
/// [`RerankFailure::reason`] produce — never a provider's own prose.
#[derive(Debug, Serialize, Deserialize)]
pub struct RerankerPlan {
    pub configured: bool,
    pub ran: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// A `rerank` was not requested at all — the overwhelmingly common
/// case, and the one ADR 0006 §12 requires cost nothing: no provider
/// call, no metrics, not even this module's own `drive` runs.
impl RerankerPlan {
    pub(crate) fn not_requested(configured: bool) -> Self {
        Self {
            configured,
            ran: false,
            model: None,
            reason: None,
        }
    }
}

/// `plan.reranker.reason` when `rerank` was requested but no provider
/// is configured on this server.
pub(crate) const REASON_NOT_CONFIGURED: &str = "not_configured";
/// `rerank.model` named a model the configured provider does not
/// serve.
pub(crate) const REASON_MODEL_MISMATCH: &str = "model_mismatch";
/// Fewer than two candidates survived to be reordered — nothing a
/// reranker could usefully do.
pub(crate) const REASON_EMPTY_POOL: &str = "empty_pool";
/// The provider's response was not a complete permutation of
/// `0..candidates.len()` — wrong length, an out-of-range index, or a
/// repeated index (ADR 0006 §12).
pub(crate) const REASON_INVALID_PERMUTATION: &str = "invalid_permutation";
/// The provider's circuit breaker is open.
pub(crate) const REASON_CIRCUIT_OPEN: &str = "circuit_open";
/// The request deadline was already spent, or was exhausted mid-call.
pub(crate) const REASON_TIMEOUT: &str = "timeout";
/// Every other provider failure: unreachable, a non-2xx status, or a
/// response that could not be decoded at all.
pub(crate) const REASON_PROVIDER_ERROR: &str = "provider_error";

/// One candidate as the reranker sees it (ADR 0006 §12): its own text
/// — the passage text, community summary paragraph, or a plain
/// rendering of subject/label/object for an association — plus enough
/// provenance for a provider that wants it, never more. Deliberately
/// does not derive `Debug`: nothing that carries candidate text may be
/// logged, matching [`super::EvidenceCandidate`]'s own discipline.
#[derive(Serialize)]
pub(crate) struct RerankCandidate {
    pub(crate) kind: String,
    pub(crate) lane_rank: usize,
    pub(crate) text: String,
}

/// One failed reranker attempt: a fixed, `'static` reason token safe to
/// put on the wire or a metric label verbatim, and whether trying again
/// could plausibly answer differently.
pub(crate) struct RerankFailure {
    pub(crate) reason: &'static str,
    retryable: bool,
}

impl RerankFailure {
    fn new(reason: &'static str, retryable: bool) -> Self {
        Self { reason, retryable }
    }
}

/// Anything that can reorder a candidate pool. The HTTP provider is the
/// real one; tests inject a deterministic fake.
pub(crate) trait EvidenceReranker: Send + Sync {
    fn model(&self) -> &str;
    /// Returns a permutation of `0..candidates.len()`: the candidate
    /// indices in the reranker's preferred order. [`drive`] treats
    /// anything else — wrong length, an out-of-range index, a repeated
    /// index — as [`REASON_INVALID_PERMUTATION`], regardless of which
    /// implementation produced it.
    fn rerank(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
        deadline: Deadline,
    ) -> Result<Vec<usize>, RerankFailure>;
    /// The provider's circuit breaker, when it has one — the registry
    /// reads it for /metrics. Fakes keep the default `None`.
    fn breaker(&self) -> Option<&ProviderBreaker> {
        None
    }
}

/// Cohere/Jina-compatible `/rerank` client: `{model, query, documents,
/// top_n}` in, `{results: [{index, relevance_score}]}` out — the same
/// "any compatible adapter plugs in" posture
/// [`crate::embedding::HttpEmbeddings`] takes for OpenAI-shaped
/// `/embeddings` servers.
pub(crate) struct HttpReranker {
    url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
    /// One attempt's wall-clock ceiling (`TAGURU_RERANK_TIMEOUT_SECS`,
    /// default 5s — this is an interactive read path, not a batch job;
    /// a request whose remaining budget is smaller shrinks the attempt
    /// to that budget.
    timeout: Duration,
    breaker: ProviderBreaker,
}

/// One retry past the first attempt (ADR 0006 §12: "bounded retry on
/// transient failures only, matching `RETRY_ATTEMPTS`/backoff shape" —
/// the shape, not the exact numbers, which #307 fixes for this shorter,
/// interactive-timeout provider: a flat 100ms backoff rather than
/// embeddings' exponential ladder, since the whole call is bounded to a
/// few seconds by default).
const RETRY_ATTEMPTS: usize = 1;
const RETRY_BACKOFF: Duration = Duration::from_millis(100);

/// Caps the buffered response the same way
/// [`crate::embedding::HttpEmbeddings`] caps its own decode read — a
/// rerank response is just an index/score array, several orders of
/// magnitude smaller than an embedding matrix, so this cap is
/// correspondingly smaller.
const MAX_RESPONSE_BYTES: u64 = 4 * 1024 * 1024;

impl HttpReranker {
    pub(crate) fn from_env() -> Option<Self> {
        let url = std::env::var("TAGURU_RERANK_URL").ok()?;
        let model = std::env::var("TAGURU_RERANK_MODEL").ok()?;
        let timeout_secs = crate::env::env_number("TAGURU_RERANK_TIMEOUT_SECS", 5).max(1);
        let timeout = Duration::from_secs(timeout_secs as u64);
        Some(Self {
            url,
            model,
            api_key: std::env::var("TAGURU_RERANK_API_KEY").ok(),
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .build()
                .into(),
            timeout,
            breaker: ProviderBreaker::new("reranker provider"),
        })
    }

    /// One provider round trip, classified for the retry loop in
    /// [`EvidenceReranker::rerank`]. The messages stay generic
    /// (`REASON_PROVIDER_ERROR`, never the provider URL or response
    /// body) for the same reason `HttpEmbeddings`' own attempt keeps
    /// its refusal strings clear of infrastructure detail — this one
    /// travels no further than a metric label and `plan.reranker.reason`,
    /// but the discipline is the same.
    fn attempt(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
        deadline: Deadline,
    ) -> Result<Vec<usize>, RerankFailure> {
        let documents: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": documents.len(),
        })
        .to_string();
        let timeout = self.timeout.min(deadline.remaining());
        let mut request = self
            .agent
            .post(&self.url)
            .config()
            .timeout_global(Some(timeout))
            .build()
            .header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let response = request.send(body.as_str()).map_err(|error| match &error {
            // Overload and server-side failure answer differently in a
            // moment; a 4xx refusal does not.
            ureq::Error::StatusCode(code) => {
                RerankFailure::new(REASON_PROVIDER_ERROR, *code == 429 || *code >= 500)
            }
            // The per-attempt timeout above is `min(self.timeout,
            // deadline.remaining())` — when it fires AND the caller's
            // own deadline has since expired, the deadline was almost
            // certainly the binding ceiling, not the provider being
            // slow in general: report `REASON_TIMEOUT` (no retry —
            // another attempt would just find the deadline expired at
            // the top of the loop) rather than the generic
            // `provider_error` a real transport failure gets.
            ureq::Error::Timeout(_) if deadline.expired() => {
                RerankFailure::new(REASON_TIMEOUT, false)
            }
            // Dropped connections and every other transport-level
            // failure (including a timeout that fired before the
            // caller's own deadline did — genuinely a slow provider)
            // are the blip the one retry exists for.
            _ => RerankFailure::new(REASON_PROVIDER_ERROR, true),
        })?;
        decode(response)
    }
}

impl EvidenceReranker for HttpReranker {
    fn model(&self) -> &str {
        &self.model
    }

    fn rerank(
        &self,
        query: &str,
        candidates: &[RerankCandidate],
        deadline: Deadline,
    ) -> Result<Vec<usize>, RerankFailure> {
        let span = tracing::info_span!(
            "rerank",
            otel.kind = "client",
            rerank.model = %self.model,
            rerank.candidates = candidates.len(),
        );
        let _guard = span.enter();

        let mut attempt = 0;
        loop {
            if deadline.expired() {
                return Err(RerankFailure::new(REASON_TIMEOUT, false));
            }
            let admission = self
                .breaker
                .admit()
                .map_err(|_refusal| RerankFailure::new(REASON_CIRCUIT_OPEN, false))?;
            match self.attempt(query, candidates, deadline) {
                Ok(order) => {
                    self.breaker.record(admission, true);
                    return Ok(order);
                }
                Err(failure) => {
                    self.breaker.record(admission, false);
                    if failure.retryable && attempt < RETRY_ATTEMPTS && !deadline.expired() {
                        attempt += 1;
                        tracing::warn!(
                            attempt,
                            of = RETRY_ATTEMPTS,
                            reason = failure.reason,
                            "transient reranker failure; retrying"
                        );
                        std::thread::sleep(RETRY_BACKOFF.min(deadline.remaining()));
                        continue;
                    }
                    return Err(failure);
                }
            }
        }
    }

    fn breaker(&self) -> Option<&ProviderBreaker> {
        Some(&self.breaker)
    }
}

/// Decodes one successful response into the raw index list — NOT yet
/// validated as a permutation; that check is [`drive`]'s job (ADR 0006
/// §12), the same regardless of which [`EvidenceReranker`] produced the
/// `Ok`. A structurally malformed response (no `results` array, a
/// non-integer `index`) fails here as [`REASON_PROVIDER_ERROR`]; a
/// well-formed but incomplete/duplicated/out-of-range index list
/// reaches [`drive`] as `Ok` and is caught there as
/// [`REASON_INVALID_PERMUTATION`].
fn decode(response: ureq::http::Response<ureq::Body>) -> Result<Vec<usize>, RerankFailure> {
    let mut body = Vec::new();
    {
        use std::io::Read;
        response
            .into_body()
            .into_reader()
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut body)
            .map_err(|_| RerankFailure::new(REASON_PROVIDER_ERROR, false))?;
    }
    if body.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(RerankFailure::new(REASON_PROVIDER_ERROR, false));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|_| RerankFailure::new(REASON_PROVIDER_ERROR, false))?;
    let results = parsed
        .get("results")
        .and_then(|value| value.as_array())
        .ok_or_else(|| RerankFailure::new(REASON_PROVIDER_ERROR, false))?;
    results
        .iter()
        .map(|entry| {
            entry
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| RerankFailure::new(REASON_PROVIDER_ERROR, false))
        })
        .collect()
}

/// `order` names a complete reordering of `0..len` — the same length,
/// every index in range, no index repeated. Anything else means the
/// provider's response cannot be applied at all (ADR 0006 §12).
fn is_valid_permutation(order: &[usize], len: usize) -> bool {
    if order.len() != len {
        return false;
    }
    let mut seen = vec![false; len];
    for &index in order {
        if index >= len || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}

/// ADR 0006 §12: "a plain rendering of subject/label/object" — the
/// association side of [`RerankCandidate::text`]. No existing helper in
/// this tree renders an association as human/model-readable prose;
/// [`super::EvidenceCandidate::canonical_key`] is a NUL-joined identity
/// key, not text a reranker should read.
fn render_association_text(association: &AssociationOut) -> String {
    format!(
        "{} {} {}",
        association.subject, association.label, association.object
    )
}

fn build_candidates(survivors: &[FusedCandidate]) -> Vec<RerankCandidate> {
    survivors
        .iter()
        .map(|fused| {
            let text = match &fused.candidate.payload {
                CandidatePayload::Association(association) => render_association_text(association),
                CandidatePayload::Passage(hit) => hit.text.clone(),
                CandidatePayload::Community(hit) => hit.text.clone(),
            };
            RerankCandidate {
                kind: fused.candidate.kind.clone(),
                lane_rank: fused.candidate.lane_rank,
                text,
            }
        })
        .collect()
}

/// One [`drive`] call's outcome, for the caller to record on
/// `taguru_rerank_outcomes_total`/`taguru_rerank_duration_seconds`
/// (#307 metrics) — kept out of this module's own responsibility the
/// same way [`select`](super::select) stays a pure function and lets
/// its caller own HTTP/metrics concerns (ADR 0006 §9's module doc).
pub(crate) struct RerankOutcome {
    pub(crate) token: &'static str,
    pub(crate) duration: Duration,
}

/// Runs the whole ADR 0006 §12 decision for one already-requested
/// rerank: provider configured? model pinned to the right one? enough
/// candidates to reorder? — then calls the provider once and validates
/// its response as a permutation. Callers only reach this when
/// `rerank` was actually requested; [`RerankerPlan::not_requested`]
/// covers every other call for free, with no provider touched and
/// nothing recorded.
pub(crate) fn drive(
    provider: Option<&dyn EvidenceReranker>,
    request: &RerankRequest,
    query: &str,
    survivors: &[FusedCandidate],
    deadline: Deadline,
) -> (Option<Vec<usize>>, RerankerPlan, RerankOutcome) {
    let started = Instant::now();
    let finish = |order: Option<Vec<usize>>, plan: RerankerPlan, token: &'static str| {
        let outcome = RerankOutcome {
            token,
            duration: started.elapsed(),
        };
        (order, plan, outcome)
    };

    let Some(provider) = provider else {
        return finish(
            None,
            RerankerPlan {
                configured: false,
                ran: false,
                model: None,
                reason: Some(REASON_NOT_CONFIGURED.to_string()),
            },
            REASON_NOT_CONFIGURED,
        );
    };

    if let Some(wanted) = &request.model
        && wanted != provider.model()
    {
        return finish(
            None,
            RerankerPlan {
                configured: true,
                ran: false,
                model: None,
                reason: Some(REASON_MODEL_MISMATCH.to_string()),
            },
            REASON_MODEL_MISMATCH,
        );
    }

    if survivors.len() < 2 {
        return finish(
            None,
            RerankerPlan {
                configured: true,
                ran: false,
                model: None,
                reason: Some(REASON_EMPTY_POOL.to_string()),
            },
            REASON_EMPTY_POOL,
        );
    }

    let candidates = build_candidates(survivors);
    match provider.rerank(query, &candidates, deadline) {
        Ok(order) if is_valid_permutation(&order, survivors.len()) => finish(
            Some(order),
            RerankerPlan {
                configured: true,
                ran: true,
                model: Some(provider.model().to_string()),
                reason: None,
            },
            "ok",
        ),
        Ok(_invalid) => finish(
            None,
            RerankerPlan {
                configured: true,
                ran: false,
                model: None,
                reason: Some(REASON_INVALID_PERMUTATION.to_string()),
            },
            REASON_INVALID_PERMUTATION,
        ),
        Err(failure) => finish(
            None,
            RerankerPlan {
                configured: true,
                ran: false,
                model: None,
                reason: Some(failure.reason.to_string()),
            },
            failure.reason,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::evidence::select::select_with_reorder;
    use crate::api::evidence::{EvidenceCandidate, fuse};
    use serde_json::{Value, json};
    use std::collections::HashMap;

    /// A fake that returns a fixed permutation regardless of input —
    /// enough to prove `drive`/`select_with_reorder` apply whatever
    /// order a provider chooses.
    struct PermutingReranker {
        order: Vec<usize>,
    }

    impl EvidenceReranker for PermutingReranker {
        fn model(&self) -> &str {
            "permuting"
        }

        fn rerank(
            &self,
            _query: &str,
            _candidates: &[RerankCandidate],
            _deadline: Deadline,
        ) -> Result<Vec<usize>, RerankFailure> {
            Ok(self.order.clone())
        }
    }

    /// Fails every attempt with a fixed reason/retryability, and counts
    /// how many times it was called.
    struct FailingReranker {
        reason: &'static str,
        retryable: bool,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl EvidenceReranker for FailingReranker {
        fn model(&self) -> &str {
            "failing"
        }

        fn rerank(
            &self,
            _query: &str,
            _candidates: &[RerankCandidate],
            _deadline: Deadline,
        ) -> Result<Vec<usize>, RerankFailure> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(RerankFailure::new(self.reason, self.retryable))
        }
    }

    fn assoc_candidate(subject: &str, rank: usize) -> EvidenceCandidate {
        let association = AssociationOut {
            subject: subject.to_string(),
            label: "likes".to_string(),
            object: "sushi".to_string(),
            weight: 1.0,
            count: 1,
            attributions: Vec::new(),
        };
        EvidenceCandidate::from_association("ctx", association, rank)
    }

    fn pool(n: usize) -> Vec<EvidenceCandidate> {
        (1..=n)
            .map(|i| assoc_candidate(&format!("s{i}"), i))
            .collect()
    }

    fn empty_lookup() -> HashMap<(String, u32), crate::api::sources::Citation> {
        HashMap::new()
    }

    #[test]
    fn a_reversing_reranker_reverses_admission_order() {
        let (fused, dropped) = fuse(pool(4));
        let order: Vec<usize> = (0..4).rev().collect();
        let reranker = PermutingReranker {
            order: order.clone(),
        };
        let request = RerankRequest { model: None };

        let mut plan = None;
        let mut outcome_token = "";
        let selected = select_with_reorder(
            fused,
            dropped,
            &crate::api::evidence::budget::BudgetLimits::resolve(None),
            &empty_lookup(),
            Some(&mut |survivors: &[FusedCandidate]| {
                let (order, this_plan, outcome) = drive(
                    Some(&reranker),
                    &request,
                    "q",
                    survivors,
                    Deadline::unbounded(),
                );
                plan = Some(this_plan);
                outcome_token = outcome.token;
                order
            }),
        );

        let plan = plan.expect("reorder callback ran");
        assert!(plan.ran, "{plan:?}");
        assert_eq!(plan.model.as_deref(), Some("permuting"));
        assert_eq!(outcome_token, "ok");
        // The last-fused-rank subject (s4) is now first in admission
        // order — the reranker's whole observable effect.
        let subjects: Vec<&str> = selected
            .items
            .iter()
            .map(|item| item.association.as_ref().unwrap().subject.as_str())
            .collect();
        assert_eq!(subjects, vec!["s4", "s3", "s2", "s1"]);
    }

    #[test]
    fn invalid_permutations_fall_back_to_the_deterministic_order() {
        let cases: Vec<(&str, Vec<usize>)> = vec![
            ("too short", vec![0, 1, 2]),
            ("too long", vec![0, 1, 2, 3, 0]),
            ("duplicate index", vec![0, 0, 1, 2]),
            ("out of range", vec![0, 1, 2, 9]),
        ];
        for (label, bad_order) in cases {
            let (fused_plain, dropped_plain) = fuse(pool(4));
            let baseline = crate::api::evidence::select::select(
                fused_plain,
                dropped_plain,
                &crate::api::evidence::budget::BudgetLimits::resolve(None),
                &empty_lookup(),
            );

            let (fused, dropped) = fuse(pool(4));
            let reranker = PermutingReranker { order: bad_order };
            let request = RerankRequest { model: None };
            let mut plan = None;
            let degraded = select_with_reorder(
                fused,
                dropped,
                &crate::api::evidence::budget::BudgetLimits::resolve(None),
                &empty_lookup(),
                Some(&mut |survivors: &[FusedCandidate]| {
                    let (order, this_plan, _outcome) = drive(
                        Some(&reranker),
                        &request,
                        "q",
                        survivors,
                        Deadline::unbounded(),
                    );
                    plan = Some(this_plan);
                    order
                }),
            );

            let plan = plan.expect("reorder callback ran");
            assert!(!plan.ran, "{label}: {plan:?}");
            assert_eq!(
                plan.reason.as_deref(),
                Some(REASON_INVALID_PERMUTATION),
                "{label}"
            );
            assert_eq!(
                serde_json::to_value(&degraded.items).unwrap(),
                serde_json::to_value(&baseline.items).unwrap(),
                "{label}: byte-identical to the unconfigured order"
            );
        }
    }

    /// A provider aimed at a local stub listener: a fresh breaker per
    /// test, so no test can trip another's circuit — the same
    /// precedent `embedding.rs`'s own `stub_provider` sets.
    fn stub_reranker(addr: std::net::SocketAddr, timeout: Duration) -> HttpReranker {
        HttpReranker {
            url: format!("http://{addr}"),
            model: "stub-model".to_string(),
            api_key: None,
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .build()
                .into(),
            timeout,
            breaker: ProviderBreaker::new("reranker provider"),
        }
    }

    fn read_full_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        use std::io::Read;
        let mut request = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            if let Some(header_end) = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|at| at + 4)
            {
                let headers = String::from_utf8_lossy(&request[..header_end]).to_lowercase();
                let body_len = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= header_end + body_len {
                    return request;
                }
            }
            let read = stream.read(&mut buffer).unwrap_or(0);
            if read == 0 {
                return request;
            }
            request.extend_from_slice(&buffer[..read]);
        }
    }

    /// One full retry ladder (`RETRY_ATTEMPTS = 1`, two attempts total)
    /// against a provider that always answers 503 — retry lives inside
    /// `HttpReranker::rerank` itself, not `drive`, so this exercises the
    /// real HTTP client rather than a fake that bypasses it.
    #[test]
    fn a_retryable_failure_is_retried_once_then_gives_up() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = stream.write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                      Connection: close\r\n\r\n",
                );
            }
        });

        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(3)).0);
        let error = provider
            .rerank("q", &candidates, Deadline::unbounded())
            .unwrap_err();
        assert_eq!(error.reason, REASON_PROVIDER_ERROR);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "one retry past the first attempt"
        );
    }

    /// A non-retryable status (404) is not retried at all.
    #[test]
    fn a_permanent_failure_is_not_retried() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
            }
        });

        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(3)).0);
        let error = provider
            .rerank("q", &candidates, Deadline::unbounded())
            .unwrap_err();
        assert_eq!(error.reason, REASON_PROVIDER_ERROR);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "no retry on a hard refusal"
        );
    }

    /// A ureq transport timeout that fires only because the caller's
    /// own deadline ran out first is reported as `REASON_TIMEOUT`, not
    /// the generic `REASON_PROVIDER_ERROR` a real transport failure
    /// gets — the deadline, not the provider, is why this attempt
    /// failed, and unlike a generic transport error it is not retried
    /// (another attempt would just find the deadline already expired).
    #[test]
    fn a_deadline_driven_transport_timeout_is_reported_as_timeout() {
        use std::sync::atomic::AtomicUsize;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Never responds — outlives every deadline below, so
                // the per-attempt ureq timeout is what actually fires.
                std::thread::sleep(Duration::from_secs(5));
            }
        });

        // A provider timeout longer than the deadline, so the
        // deadline — not `self.timeout` — is what actually binds the
        // per-attempt ceiling (`min(self.timeout, deadline.remaining())`).
        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(3)).0);
        let error = provider
            .rerank("q", &candidates, Deadline::after(Duration::from_millis(50)))
            .unwrap_err();
        assert_eq!(error.reason, REASON_TIMEOUT);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a deadline-driven timeout is not retried"
        );
    }

    /// `decode`'s own inputs come from a `ureq::http::Response`, which
    /// nothing outside an actual round trip can construct — so every
    /// failure branch here runs through the real stub listener rather
    /// than calling `decode` directly, the same way every other
    /// HTTP-level test in this module works.
    fn respond_with(status_line: &'static str, body: &'static [u8]) -> std::net::SocketAddr {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                let head = format!(
                    "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });
        addr
    }

    #[test]
    fn a_non_json_body_is_a_provider_error() {
        let addr = respond_with("200 OK", b"not json at all");
        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(2)).0);
        let error = provider
            .rerank("q", &candidates, Deadline::unbounded())
            .unwrap_err();
        assert_eq!(error.reason, REASON_PROVIDER_ERROR);
    }

    #[test]
    fn a_body_with_no_results_array_is_a_provider_error() {
        let addr = respond_with("200 OK", br#"{"other":"shape"}"#);
        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(2)).0);
        let error = provider
            .rerank("q", &candidates, Deadline::unbounded())
            .unwrap_err();
        assert_eq!(error.reason, REASON_PROVIDER_ERROR);
    }

    #[test]
    fn a_non_integer_or_negative_index_is_a_provider_error() {
        for body in [
            br#"{"results":[{"index":"not-a-number"}]}"#.as_slice(),
            br#"{"results":[{"index":-1}]}"#.as_slice(),
        ] {
            let addr = respond_with("200 OK", body);
            let provider = stub_reranker(addr, Duration::from_secs(5));
            let candidates = build_candidates(&fuse(pool(2)).0);
            let error = provider
                .rerank("q", &candidates, Deadline::unbounded())
                .unwrap_err();
            assert_eq!(
                error.reason,
                REASON_PROVIDER_ERROR,
                "{}",
                String::from_utf8_lossy(body)
            );
        }
    }

    #[test]
    fn a_body_over_the_response_cap_is_a_provider_error() {
        // The size check runs before JSON parsing, so the filler need
        // not be valid JSON — only larger than MAX_RESPONSE_BYTES.
        let oversized: Vec<u8> = vec![b'x'; MAX_RESPONSE_BYTES as usize + 1024];
        // `respond_with` needs a `'static` body; leak the one-off
        // buffer this single test allocates rather than threading a
        // lifetime through the shared helper for one caller.
        let oversized: &'static [u8] = oversized.leak();
        let addr = respond_with("200 OK", oversized);
        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(2)).0);
        let error = provider
            .rerank("q", &candidates, Deadline::unbounded())
            .unwrap_err();
        assert_eq!(error.reason, REASON_PROVIDER_ERROR);
    }

    /// The one real end-to-end success round trip in this module:
    /// confirms the OUTGOING request shape (`model`/`query`/
    /// `documents`/`top_n`) and that a well-formed 200 decodes into
    /// the exact permutation the stub named — every other test either
    /// fakes the provider (`EvidenceReranker` directly) or only
    /// exercises a failure branch through the real HTTP client.
    #[test]
    fn a_successful_response_decodes_the_named_permutation_and_the_request_matches_the_contract() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let received: std::sync::Arc<std::sync::Mutex<Option<Value>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        let captured = std::sync::Arc::clone(&received);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let request = read_full_request(&mut stream);
                let body_start = request.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
                let parsed: Value = serde_json::from_slice(&request[body_start..]).unwrap();
                *captured.lock().unwrap() = Some(parsed);
                let body = br#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            }
        });

        let provider = stub_reranker(addr, Duration::from_secs(5));
        let candidates = build_candidates(&fuse(pool(2)).0);
        let order = match provider.rerank("the query text", &candidates, Deadline::unbounded()) {
            Ok(order) => order,
            // `RerankFailure` deliberately does not derive `Debug`
            // (it never carries candidate text, but nothing here
            // should need it to either) — `reason` alone is enough to
            // fail loudly.
            Err(failure) => panic!("a well-formed 200 must decode: {}", failure.reason),
        };
        assert_eq!(order, vec![1, 0]);

        let request = received.lock().unwrap().take().expect("stub was called");
        assert_eq!(request["model"], json!("stub-model"));
        assert_eq!(request["query"], json!("the query text"));
        assert_eq!(request["top_n"], json!(2));
        let documents = request["documents"].as_array().expect("documents array");
        assert_eq!(documents.len(), 2);
    }

    /// A breaker that opens after enough consecutive failures short-
    /// circuits the next call without touching the provider —
    /// `HttpReranker::rerank` consults its own breaker exactly the way
    /// `HttpEmbeddings::embed` does.
    #[test]
    fn the_breaker_opens_and_short_circuits_the_next_call() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _ = stream.write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                      Connection: close\r\n\r\n",
                );
            }
        });

        let mut provider = stub_reranker(addr, Duration::from_secs(5));
        provider.breaker =
            ProviderBreaker::with_policy("reranker provider", 2, Duration::from_secs(30));
        let candidates = build_candidates(&fuse(pool(3)).0);
        // First call: two attempts (one retry), both failures — reaches
        // the threshold of 2 and opens the breaker.
        let _ = provider.rerank("q", &candidates, Deadline::unbounded());
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);

        let error = provider
            .rerank("q", &candidates, Deadline::unbounded())
            .unwrap_err();
        assert_eq!(error.reason, REASON_CIRCUIT_OPEN);
        assert_eq!(
            hits.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "no third round trip once the breaker is open"
        );
    }

    /// `HttpReranker::rerank` checks its own deadline before ever
    /// calling out (`REASON_TIMEOUT`, no attempt made) — this fake
    /// mirrors that same contract so `drive` is exercised against it
    /// without a real network round trip.
    struct AlwaysExpired {
        calls: std::sync::atomic::AtomicUsize,
    }
    impl EvidenceReranker for AlwaysExpired {
        fn model(&self) -> &str {
            "always-expired"
        }
        fn rerank(
            &self,
            _query: &str,
            _candidates: &[RerankCandidate],
            deadline: Deadline,
        ) -> Result<Vec<usize>, RerankFailure> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if deadline.expired() {
                Err(RerankFailure::new(REASON_TIMEOUT, false))
            } else {
                Ok(vec![])
            }
        }
    }

    #[test]
    fn an_expired_deadline_is_reported_as_timeout() {
        let reranker = AlwaysExpired {
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let (fused, _dropped) = fuse(pool(3));
        let request = RerankRequest { model: None };
        let (order, plan, outcome) = drive(
            Some(&reranker),
            &request,
            "q",
            &fused,
            Deadline::after(Duration::ZERO),
        );
        assert!(order.is_none());
        assert_eq!(plan.reason.as_deref(), Some(REASON_TIMEOUT));
        assert_eq!(outcome.token, REASON_TIMEOUT);
        assert_eq!(
            reranker.calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "drive still calls the trait; the provider itself owns the deadline check"
        );
    }

    #[test]
    fn fewer_than_two_survivors_skips_the_provider() {
        let reranker = FailingReranker {
            reason: "should never be reached",
            retryable: false,
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let (fused, _dropped) = fuse(pool(1));
        let request = RerankRequest { model: None };
        let (order, plan, outcome) = drive(
            Some(&reranker),
            &request,
            "q",
            &fused,
            Deadline::unbounded(),
        );
        assert!(order.is_none());
        assert_eq!(plan.reason.as_deref(), Some(REASON_EMPTY_POOL));
        assert_eq!(outcome.token, REASON_EMPTY_POOL);
        assert_eq!(reranker.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn a_model_mismatch_skips_the_provider() {
        let reranker = FailingReranker {
            reason: "should never be reached",
            retryable: false,
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let (fused, _dropped) = fuse(pool(3));
        let request = RerankRequest {
            model: Some("a-different-model".to_string()),
        };
        let (order, plan, outcome) = drive(
            Some(&reranker),
            &request,
            "q",
            &fused,
            Deadline::unbounded(),
        );
        assert!(order.is_none());
        assert_eq!(plan.reason.as_deref(), Some(REASON_MODEL_MISMATCH));
        assert_eq!(outcome.token, REASON_MODEL_MISMATCH);
        assert_eq!(reranker.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn no_provider_configured_reports_not_configured() {
        let (fused, _dropped) = fuse(pool(3));
        let request = RerankRequest { model: None };
        let (order, plan, outcome) = drive(None, &request, "q", &fused, Deadline::unbounded());
        assert!(order.is_none());
        assert!(!plan.configured);
        assert_eq!(plan.reason.as_deref(), Some(REASON_NOT_CONFIGURED));
        assert_eq!(outcome.token, REASON_NOT_CONFIGURED);
    }

    #[test]
    fn permutation_validation_accepts_only_a_complete_reordering() {
        assert!(is_valid_permutation(&[0, 1, 2], 3));
        assert!(is_valid_permutation(&[2, 0, 1], 3));
        assert!(!is_valid_permutation(&[0, 1], 3));
        assert!(!is_valid_permutation(&[0, 1, 2, 3], 3));
        assert!(!is_valid_permutation(&[0, 0, 1], 3));
        assert!(!is_valid_permutation(&[0, 1, 5], 3));
    }
}
