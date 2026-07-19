//! Optional semantic entry tier: embeddings over concept and label
//! GLOSSES — each name plus its heaviest graph facts, because a lone
//! word carries too little signal for sentence-trained embedding
//! models, and the graph already owns the context. Retrieval itself
//! stays structural; only the entry needs meaning. Configured via `TAGURU_EMBED_URL` /
//! `TAGURU_EMBED_MODEL` / `TAGURU_EMBED_API_KEY` against any
//! OpenAI-compatible `/embeddings` endpoint; absent config disables the
//! tier and resolve stays purely lexical.
//!
//! Vectors are a derived cache of (model × name), kept per context in a
//! `{name}.vectors.bin` sidecar — refreshed explicitly (POST
//! /contexts/{name}/embeddings/refresh), loaded on demand by the
//! semantic fallback, and discarded wholesale when the model changes.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use taguru::deadline::{Deadline, DeadlineExceeded};

/// Why a batch is being embedded: glosses entering the store (`Index`)
/// or a live cue looking things up (`Query`). Modern embedding APIs
/// (Cohere, Voyage) encode documents and queries asymmetrically; the
/// OpenAI request shape cannot carry the distinction, so the HTTP
/// provider forwards it as an `X-Taguru-Embed-Purpose` header for
/// bridging proxies to map (e.g. onto Cohere's `input_type`). Plain
/// OpenAI-compatible servers ignore the header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedPurpose {
    Index,
    Query,
}

impl EmbedPurpose {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedPurpose::Index => "index",
            EmbedPurpose::Query => "query",
        }
    }
}

/// The process-wide "a stop signal has arrived" latch, cloned into
/// [`HttpEmbeddings`] so an in-flight provider wait is abandoned the
/// moment the graceful drain starts — the one blocking call in the
/// serve path that neither the request timeout nor the runtime can
/// preempt otherwise. Set once, never cleared: the process is exiting.
#[derive(Clone, Default)]
pub struct ShutdownFlag(Arc<AtomicBool>);

impl ShutdownFlag {
    pub fn begin(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn active(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Consecutive provider-attempt failures that open the breaker. Counted
/// per ATTEMPT, not per `embed()` call, so one full retry ladder against
/// a dead provider is enough to open it — the next caller fails fast
/// instead of paying the timeout again.
const BREAKER_THRESHOLD: u32 = 3;
/// How long an open breaker refuses before letting one probe through.
const BREAKER_COOLDOWN: Duration = Duration::from_secs(30);

/// A minimal circuit breaker over the embedding provider: open after
/// [`BREAKER_THRESHOLD`] consecutive attempt failures, then one probe
/// per [`BREAKER_COOLDOWN`] until a success closes it. Exists so a dead
/// provider costs each request a refusal instead of a full timeout —
/// the lanes behind it already know how to degrade; this makes the
/// degradation cheap. Shared by handle: [`HttpEmbeddings`] holds one
/// and the registry reads the same one for /metrics and its own
/// fast-fail gate.
#[derive(Clone)]
pub struct EmbedBreaker(Arc<BreakerInner>);

struct BreakerInner {
    threshold: u32,
    cooldown: Duration,
    state: Mutex<BreakerState>,
    opened_total: AtomicU64,
    short_circuits_total: AtomicU64,
}

enum BreakerState {
    Closed {
        consecutive_failures: u32,
    },
    Open {
        since: Instant,
    },
    /// One probe at a time: `probing` is the in-flight claim, released
    /// by the probe's outcome (or [`EmbedBreaker::abandon`]).
    HalfOpen {
        probing: bool,
    },
}

/// What [`EmbedBreaker::admit`] granted: an ordinary call, or the one
/// half-open probe whose outcome decides the breaker's next state.
#[derive(Clone, Copy, Debug)]
pub enum Admission {
    Normal,
    Probe,
}

/// One scrape's view of the breaker, rendered under
/// `taguru_embedding_breaker_*` on /metrics.
pub struct BreakerSnapshot {
    /// 0 = closed, 1 = open, 2 = half-open.
    pub state: u8,
    pub consecutive_failures: u64,
    pub opened_total: u64,
    pub short_circuits_total: u64,
}

impl Default for EmbedBreaker {
    fn default() -> Self {
        Self::with_policy(BREAKER_THRESHOLD, BREAKER_COOLDOWN)
    }
}

impl EmbedBreaker {
    /// The production policy is the constants above; tests shrink the
    /// cooldown to keep the half-open walk fast.
    pub fn with_policy(threshold: u32, cooldown: Duration) -> Self {
        Self(Arc::new(BreakerInner {
            threshold: threshold.max(1),
            cooldown,
            state: Mutex::new(BreakerState::Closed {
                consecutive_failures: 0,
            }),
            opened_total: AtomicU64::new(0),
            short_circuits_total: AtomicU64::new(0),
        }))
    }

    /// Gate one provider attempt. `Err` is the refusal message the
    /// caller returns without touching the provider; `Ok` says whether
    /// this attempt is the half-open probe.
    fn admit(&self) -> Result<Admission, String> {
        let mut state = self.0.state.lock();
        match &mut *state {
            BreakerState::Closed { .. } => Ok(Admission::Normal),
            BreakerState::Open { since } => {
                let elapsed = since.elapsed();
                if elapsed >= self.0.cooldown {
                    *state = BreakerState::HalfOpen { probing: true };
                    Ok(Admission::Probe)
                } else {
                    self.0.short_circuits_total.fetch_add(1, Ordering::Relaxed);
                    Err(self.open_refusal(self.0.cooldown - elapsed))
                }
            }
            BreakerState::HalfOpen { probing } => {
                if *probing {
                    self.0.short_circuits_total.fetch_add(1, Ordering::Relaxed);
                    Err(PROBE_BUSY_REFUSAL.to_string())
                } else {
                    *probing = true;
                    Ok(Admission::Probe)
                }
            }
        }
    }

    /// The same refusal arms as [`Self::admit`], WITHOUT claiming the
    /// probe slot — the registry's pre-flight gate, so a short-circuit
    /// never lands in the provider-latency histogram as a ~0s sample.
    /// `None` means "call the provider" (whose own `admit` then claims
    /// the probe if one is due).
    pub fn refusal(&self) -> Option<String> {
        let state = self.0.state.lock();
        let refusal = match &*state {
            BreakerState::Closed { .. } => None,
            BreakerState::Open { since } => {
                let elapsed = since.elapsed();
                (elapsed < self.0.cooldown).then(|| self.open_refusal(self.0.cooldown - elapsed))
            }
            BreakerState::HalfOpen { probing } => probing.then(|| PROBE_BUSY_REFUSAL.to_string()),
        };
        if refusal.is_some() {
            self.0.short_circuits_total.fetch_add(1, Ordering::Relaxed);
        }
        refusal
    }

    /// One admitted attempt's outcome. A success closes the breaker
    /// whatever granted the attempt; a failed probe re-opens for a
    /// fresh cooldown; failed normal attempts count toward the
    /// threshold only while closed (an already-open breaker needs no
    /// more evidence, and a stale straggler must not kill a pending
    /// probe).
    fn record(&self, admission: Admission, ok: bool) {
        let mut state = self.0.state.lock();
        if ok {
            *state = BreakerState::Closed {
                consecutive_failures: 0,
            };
            return;
        }
        match (admission, &mut *state) {
            (Admission::Probe, _) => {
                *state = BreakerState::Open {
                    since: Instant::now(),
                };
                self.0.opened_total.fetch_add(1, Ordering::Relaxed);
            }
            (
                Admission::Normal,
                BreakerState::Closed {
                    consecutive_failures,
                },
            ) => {
                *consecutive_failures += 1;
                if *consecutive_failures >= self.0.threshold {
                    *state = BreakerState::Open {
                        since: Instant::now(),
                    };
                    self.0.opened_total.fetch_add(1, Ordering::Relaxed);
                }
            }
            (Admission::Normal, _) => {}
        }
    }

    /// Releases an admitted attempt that produced NO outcome (the
    /// shutdown abandon): a claimed probe slot must not wedge the
    /// breaker half-open forever, and an abandoned wait is no evidence
    /// about the provider either way.
    fn abandon(&self, admission: Admission) {
        if let Admission::Probe = admission
            && let BreakerState::HalfOpen { probing } = &mut *self.0.state.lock()
        {
            *probing = false;
        }
    }

    pub fn snapshot(&self) -> BreakerSnapshot {
        let state = self.0.state.lock();
        let (state_code, consecutive_failures) = match &*state {
            BreakerState::Closed {
                consecutive_failures,
            } => (0, u64::from(*consecutive_failures)),
            BreakerState::Open { .. } => (1, u64::from(self.0.threshold)),
            BreakerState::HalfOpen { .. } => (2, u64::from(self.0.threshold)),
        };
        BreakerSnapshot {
            state: state_code,
            consecutive_failures,
            opened_total: self.0.opened_total.load(Ordering::Relaxed),
            short_circuits_total: self.0.short_circuits_total.load(Ordering::Relaxed),
        }
    }

    /// The open-state refusal. Client-facing (it rides 502 bodies and
    /// explain reasons), so it names the recovery path, never the URL.
    fn open_refusal(&self, retry_in: Duration) -> String {
        format!(
            "the embedding provider circuit is open after {} consecutive failures; \
             the next probe is allowed in {}s",
            self.0.threshold,
            retry_in.as_secs().max(1)
        )
    }
}

const PROBE_BUSY_REFUSAL: &str = "the embedding provider circuit is half-open with a probe already in flight; \
     not adding load until it settles";

/// Anything that can turn short strings into vectors. The HTTP provider
/// is the real one; tests inject a deterministic mock.
pub trait EmbeddingProvider: Send + Sync {
    fn model(&self) -> &str;
    /// Returns one vector per input text, all the same dimension. Each
    /// attempt is bounded by the smaller of the provider timeout and
    /// `deadline`'s remaining budget, and the whole call gives up
    /// early once the deadline expires or the process starts shutting
    /// down.
    fn embed(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String>;
    /// The provider's circuit breaker, when it has one — the registry
    /// reads it for /metrics and its own fast-fail gate. Mocks keep
    /// the default `None` and stay breaker-free.
    fn breaker(&self) -> Option<&EmbedBreaker> {
        None
    }
}

/// OpenAI-compatible `/embeddings` client: `{model, input: [...]}` in,
/// `{data: [{embedding: [...]}]}` out.
pub struct HttpEmbeddings {
    url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
    /// One attempt's wall-clock ceiling (`TAGURU_EMBED_TIMEOUT_SECS`);
    /// a request whose remaining budget is smaller shrinks the attempt
    /// to that budget.
    timeout: Duration,
    shutdown: ShutdownFlag,
    breaker: EmbedBreaker,
}

/// Transient provider failures retry this many times past the first
/// attempt, sleeping [`RETRY_INITIAL_BACKOFF`] then five times that.
/// Small on purpose: the refresh paths hold per-context refresh locks
/// across the round trip, and resolve's caller is an interactive
/// request — a provider that is down stays down; the retries are for
/// the blip, not the outage.
const RETRY_ATTEMPTS: usize = 2;
const RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(100);

/// How often the wait on a worker-thread attempt re-checks the
/// shutdown flag.
const SHUTDOWN_POLL: Duration = Duration::from_millis(50);

/// What `embed()` answers once the stop signal is up — before an
/// attempt starts, or mid-wait when the attempt is abandoned.
const SHUTDOWN_REFUSAL: &str = "the embedding call was abandoned: the server is shutting down";

/// One failed attempt: what to tell the caller, and whether trying
/// again could plausibly answer differently — a dropped connection, a
/// timeout, a 429 or a 5xx can; a 4xx refusal or a malformed response
/// body cannot.
struct Refusal {
    message: String,
    retryable: bool,
}

impl HttpEmbeddings {
    pub fn from_env(shutdown: ShutdownFlag) -> Option<Self> {
        let url = std::env::var("TAGURU_EMBED_URL").ok()?;
        let model = std::env::var("TAGURU_EMBED_MODEL").ok()?;
        // The per-attempt ceiling (TAGURU_EMBED_TIMEOUT_SECS, default
        // 60): one provider round trip never runs longer — and a
        // request deadline tighter than this bounds the attempt
        // further still (see `supervised_attempt`). Floored to 1 like
        // the other second knobs.
        let timeout_secs = crate::env::env_number("TAGURU_EMBED_TIMEOUT_SECS", 60).max(1);
        let timeout = Duration::from_secs(timeout_secs as u64);
        Some(Self {
            url,
            model,
            api_key: std::env::var("TAGURU_EMBED_API_KEY").ok(),
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .build()
                .into(),
            timeout,
            shutdown,
            breaker: EmbedBreaker::default(),
        })
    }
}

impl EmbeddingProvider for HttpEmbeddings {
    fn model(&self) -> &str {
        &self.model
    }

    fn embed(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String> {
        // A client span per provider round trip — the one downstream
        // call whose latency Taguru's own timings cannot explain. Runs
        // on the caller's thread (block_in_place), so a request span
        // in scope becomes its parent automatically. Retries stay
        // inside the one span: one logical embed, one client call.
        let span = tracing::info_span!(
            "embed",
            otel.kind = "client",
            embed.model = %self.model,
            embed.inputs = texts.len(),
            embed.purpose = purpose.as_str(),
        );
        let _guard = span.enter();
        let mut backoff = RETRY_INITIAL_BACKOFF;
        let mut attempt = 0;
        loop {
            if self.shutdown.active() {
                return Err(SHUTDOWN_REFUSAL.to_string());
            }
            if deadline.expired() {
                return Err(DeadlineExceeded.to_string());
            }
            let admission = self.breaker.admit()?;
            match self.supervised_attempt(texts, purpose, deadline) {
                AttemptOutcome::ShutDown => {
                    // No outcome to judge the provider by — release the
                    // admission (a claimed probe slot must not wedge)
                    // and report the drain, not the provider.
                    self.breaker.abandon(admission);
                    return Err(SHUTDOWN_REFUSAL.to_string());
                }
                AttemptOutcome::Finished(Ok(vectors)) => {
                    self.breaker.record(admission, true);
                    return Ok(vectors);
                }
                AttemptOutcome::Finished(Err(refusal)) => {
                    self.breaker.record(admission, false);
                    if refusal.retryable
                        && attempt < RETRY_ATTEMPTS
                        && !deadline.expired()
                        && !self.shutdown.active()
                    {
                        attempt += 1;
                        tracing::warn!(
                            attempt,
                            of = RETRY_ATTEMPTS,
                            error = %refusal.message,
                            "transient embedding failure; retrying"
                        );
                        std::thread::sleep(backoff.min(deadline.remaining()));
                        backoff *= 5;
                    } else {
                        return Err(refusal.message);
                    }
                }
            }
        }
    }

    fn breaker(&self) -> Option<&EmbedBreaker> {
        Some(&self.breaker)
    }
}

/// One worker-thread attempt's result — or the news that nobody stayed
/// to hear it.
enum AttemptOutcome {
    Finished(Result<Vec<Vec<f32>>, Refusal>),
    /// The stop signal arrived mid-wait: the worker thread is left to
    /// run out its own (bounded) timeout and exit; its result goes
    /// nowhere.
    ShutDown,
}

impl HttpEmbeddings {
    /// Runs one provider round trip on its own thread and waits for it,
    /// re-checking the shutdown flag every [`SHUTDOWN_POLL`]. A
    /// blocking ureq call cannot be interrupted — but the WAIT can be
    /// abandoned, which is what lets a graceful drain proceed the
    /// moment the stop signal lands instead of waiting out the
    /// provider timeout. The attempt itself is bounded by the smaller
    /// of the provider timeout and the deadline's remaining budget, so
    /// an abandoned worker exits on its own soon after.
    fn supervised_attempt(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        deadline: Deadline,
    ) -> AttemptOutcome {
        let call = Arc::new(AttemptCall {
            agent: self.agent.clone(),
            url: self.url.clone(),
            api_key: self.api_key.clone(),
            body: serde_json::json!({ "model": self.model, "input": texts }).to_string(),
            purpose,
            timeout: self.timeout.min(deadline.remaining()),
            expected: texts.len(),
        });
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = {
            let call = Arc::clone(&call);
            move || {
                let _ = sender.send(call.run());
            }
        };
        if std::thread::Builder::new()
            .name("taguru-embed".to_string())
            .spawn(worker)
            .is_err()
        {
            // No thread to be had (fd/memory pressure): run inline.
            // Loses only the shutdown abandon, never the deadline
            // bound the call itself carries.
            return AttemptOutcome::Finished(call.run());
        }
        loop {
            match receiver.recv_timeout(SHUTDOWN_POLL) {
                Ok(outcome) => return AttemptOutcome::Finished(outcome),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if self.shutdown.active() {
                        return AttemptOutcome::ShutDown;
                    }
                }
                // The worker panicking would drop its sender; ureq does
                // not panic on I/O failure, so this names a bug rather
                // than a provider state.
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return AttemptOutcome::Finished(Err(Refusal {
                        message: "the embedding worker thread died mid-call".to_string(),
                        retryable: false,
                    }));
                }
            }
        }
    }
}

/// Everything one provider round trip needs, owned outright, so the
/// attempt can run on a worker thread the caller is free to stop
/// waiting on.
struct AttemptCall {
    agent: ureq::Agent,
    url: String,
    api_key: Option<String>,
    body: String,
    purpose: EmbedPurpose,
    /// This attempt's whole-call ceiling: the provider timeout, or the
    /// request's remaining budget when that is tighter.
    timeout: Duration,
    expected: usize,
}

impl AttemptCall {
    /// One provider round trip, classified for the retry loop above.
    fn run(&self) -> Result<Vec<Vec<f32>>, Refusal> {
        // Everything past the transport is a hard refusal: a malformed
        // body will be malformed again.
        let hard = |message: String| Refusal {
            message,
            retryable: false,
        };
        let mut request = self
            .agent
            .post(&self.url)
            .config()
            .timeout_global(Some(self.timeout))
            .build()
            .header("Content-Type", "application/json")
            .header("X-Taguru-Embed-Purpose", self.purpose.as_str());
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        // The messages stay clear of the provider URL: these strings
        // travel to CLIENTS in 502 bodies (refresh, resolve's semantic
        // tier), and the endpoint an operator configured is
        // infrastructure detail, not client business. The status code /
        // transport kind is what a caller can act on.
        let response = request.send(self.body.as_str()).map_err(|error| {
            let (message, retryable) = match &error {
                // Overload and server-side failure answer differently
                // in a moment; a 4xx refusal does not.
                ureq::Error::StatusCode(code) => (
                    format!("embedding provider answered HTTP {code}"),
                    *code == 429 || *code >= 500,
                ),
                // Dropped connections and timeouts are the blip the
                // retries exist for.
                transport => (
                    format!("embedding provider unreachable ({transport})"),
                    true,
                ),
            };
            Refusal { message, retryable }
        })?;
        decode(response, self.expected).map_err(hard)
    }
}

/// Decodes one successful response into per-input vectors (`expected`
/// of them).
fn decode(
    response: ureq::http::Response<ureq::Body>,
    expected: usize,
) -> Result<Vec<Vec<f32>>, String> {
    // ureq's `read_to_string`/`read_json` cap their reads at 10 MiB,
    // below the largest honest reply — and the agent's 60s timeout
    // bounds time, not bytes, so a misbehaving or misaddressed
    // provider must not get an unbounded buffer either. Read through
    // an explicit cap instead: 64 MiB clears the largest honest
    // reply several times over (128 inputs × 4096 dims × ~24 JSON
    // bytes per component ≈ 13 MiB) while keeping the buffer finite.
    const MAX_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;
    let mut body = Vec::new();
    {
        use std::io::Read;
        response
            .into_body()
            .into_reader()
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut body)
            .map_err(|error| format!("embedding response unreadable: {error}"))?;
    }
    if body.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(format!(
            "embedding response is larger than {MAX_RESPONSE_BYTES} bytes; \
                 refusing to buffer it"
        ));
    }
    let parsed: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|error| format!("embedding response unreadable: {error}"))?;
    let data = parsed
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or("embedding response carries no data array")?;
    if data.len() != expected {
        return Err(format!(
            "embedding response returned {} vectors for {expected} inputs",
            data.len(),
        ));
    }
    // A provider or proxy may return `data` out of input order; each
    // entry's `index` names the input it belongs to (the OpenAI
    // embedding contract). Place each vector at its index rather than
    // trust array order, so a reordered response cannot silently pair
    // every embedding with the wrong text. An entry that omits `index`
    // falls back to its array position — the old, order-trusting
    // behavior, so a provider that never sends `index` still works.
    let mut slots: Vec<Option<Vec<f32>>> = vec![None; expected];
    let mut width: Option<usize> = None;
    for (position, item) in data.iter().enumerate() {
        let index = item
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or(position);
        if index >= slots.len() {
            return Err(format!(
                "embedding response index {index} out of range for {expected} inputs"
            ));
        }
        let embedding = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or("embedding entry carries no vector")?;
        // Every other malformed shape here (missing vector, wrong
        // width, out-of-range or repeated index) fails the whole
        // response rather than patching around it; a non-numeric
        // component gets the same treatment instead of silently
        // becoming 0.0 and corrupting the stored vector's geometry.
        let mut vector: Vec<f32> = embedding
            .iter()
            .map(|component| {
                let value = component.as_f64().ok_or_else(|| {
                    format!("embedding entry {index} contains a non-numeric component")
                })? as f32;
                // A finite f64 past f32::MAX casts to ±inf; carried
                // into `normalize` it poisons the stored vector (inf
                // becomes NaN, or a norm that overflows and zeroes
                // every component). Name it at the boundary like any
                // other malformed component instead of persisting it.
                if !value.is_finite() {
                    return Err(format!(
                        "embedding entry {index} contains an out-of-range component"
                    ));
                }
                Ok(value)
            })
            .collect::<Result<_, _>>()?;
        // One response, one model, one width. A mixed-width response
        // is a provider bug; carried into a store it would feed
        // `similarity` mismatched dimensions later, which score as
        // nothing — better to name the bug at the boundary.
        match width {
            None => width = Some(vector.len()),
            Some(expected) if expected != vector.len() => {
                return Err(format!(
                    "embedding response mixes vector widths ({expected} and {})",
                    vector.len()
                ));
            }
            Some(_) => {}
        }
        normalize(&mut vector)?;
        if slots[index].replace(vector).is_some() {
            return Err(format!("embedding response repeated index {index}"));
        }
    }
    // No slot can remain empty here — the count matched and no index
    // repeated — but guard rather than unwrap so a response that both
    // omits `index` and collides two entries onto one position fails
    // cleanly instead of panicking.
    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.ok_or_else(|| format!("embedding response missing index {index}"))
        })
        .collect()
}

/// Scales a vector to unit length, so similarity is a plain dot
/// product. A zero vector is left as-is — there is nothing to scale
/// toward, and an all-zero embedding is a legitimate (if useless)
/// value, not an error.
///
/// A non-finite norm is rejected outright rather than silently left
/// unnormalized. Accumulating in f64 (a large but finite f32
/// component squares past f32::MAX, overflowing an f32 sum of squares
/// to +inf) keeps this unreachable for any realistic width given the
/// caller's own component-by-component finiteness check — but should
/// that guard ever lapse, a NaN- or Inf-carrying vector must not
/// reach the store silently: every similarity score computed against
/// it downstream would be corrupted, invisibly, until then.
pub fn normalize(vector: &mut [f32]) -> Result<(), String> {
    let norm = vector
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt() as f32;
    if !norm.is_finite() {
        return Err(format!("embedding vector norm is not finite: {norm}"));
    }
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
    Ok(())
}

/// Cosine similarity of two unit vectors (a dot product).
///
/// Widths must agree: `zip` would silently truncate to the shorter
/// side, turning a dimension mismatch (a provider changing output
/// width behind a stable model name) into a plausible-looking but
/// meaningless score. No signal is the honest answer.
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// name → (gloss hash, unit vector). The hash is of the gloss the
/// vector was computed from, so a refresh re-embeds exactly the names
/// whose graph context changed.
pub type VectorTable = HashMap<String, (u64, Vec<f32>)>;

/// One context's name vectors: a derived cache keyed by the model that
/// produced it. Wholesale discarded on model change — vectors from two
/// models must never be compared.
#[derive(Debug, Default, Clone)]
pub struct VectorStore {
    pub model: String,
    pub concepts: VectorTable,
    pub labels: VectorTable,
}

const VECTOR_MAGIC: &[u8; 8] = b"TAGURUV2";

impl VectorStore {
    /// Reads a sidecar, returning an empty store on any problem — a
    /// corrupt vector cache costs a re-embed, never an outage.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_bytes(&bytes).unwrap_or_else(|| {
                tracing::warn!("ignoring corrupt vector store at {}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        crate::storage::write_atomic(path, &self.to_bytes())
    }

    /// Rough resident bytes when this store is held in memory, so the
    /// cache budget can account for it.
    pub fn footprint(&self) -> usize {
        const ENTRY_OVERHEAD: usize = 64;
        let table = |table: &VectorTable| -> usize {
            table
                .iter()
                .map(|(name, (_, vector))| name.len() + vector.len() * 4 + ENTRY_OVERHEAD)
                .sum()
        };
        table(&self.concepts) + table(&self.labels)
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(VECTOR_MAGIC);
        write_string(&mut out, &self.model);
        write_table(&mut out, &self.concepts);
        write_table(&mut out, &self.labels);
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        if bytes.get(..8)? != VECTOR_MAGIC {
            return None;
        }
        pos += 8;
        let model = read_string(bytes, &mut pos)?;
        let concepts = read_table(bytes, &mut pos)?;
        let labels = read_table(bytes, &mut pos)?;
        (pos == bytes.len()).then_some(Self {
            model,
            concepts,
            labels,
        })
    }
}

/// One embedded row's identity: the source id attributions carry, the
/// paragraph's position within that source's current split, and the
/// FNV-1a hash of exactly its bytes — the staleness detector, same as
/// gloss vectors. `question_hash` discriminates doc2query rows: a
/// paragraph can carry several vectors (its own text plus each stored
/// question, all pointing AT the paragraph), and without the
/// discriminator they would share one key and collapse into a single
/// carried-forward row on refresh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageKey {
    pub source: String,
    pub index: u32,
    pub hash: u64,
    /// `None` = the paragraph text's own row; `Some(fnv1a(question))`
    /// = a question row for that paragraph.
    pub question_hash: Option<u64>,
}

const PASSAGE_VECTOR_MAGIC: &[u8; 8] = b"TAGURUP2";
/// The pre-doc2query row format. Discarded politely on load — the same
/// wholesale-discard contract as a model change: the next refresh
/// re-embeds, nothing is lost but provider spend.
const LEGACY_PASSAGE_VECTOR_MAGIC: &[u8; 8] = b"TAGURUP1";

/// Below this many rows, [`PassageVectorStore::top_matches`] sweeps
/// every row exactly — the flat layout already IS a fine index at
/// that scale, and exact ranking is worth more than the fraction of a
/// millisecond an approximation would save. At or above it, an
/// approximate [`PassageAnnIndex`] narrows the sweep instead of
/// scanning every row on every query.
///
/// Calibrated by `passage_ann_threshold_calibration_sweep` (run it
/// after touching the index): at 10k rows the exact sweep costs
/// 6–14 ms of read-fenced CPU per query (dim 768–1536) against
/// <1 ms for the index at ~100% measured recall@10, and the one-time
/// lazy build (0.6–1.3 s) fits comfortably inside a request budget —
/// which the build at the old 50k threshold (7–15 s) did not reliably
/// do. Sits at half [`crate::registry::DEFAULT_PASSAGE_VECTOR_LIMIT`]
/// so the index engages in default configuration with headroom (a
/// `const` assertion beside that limit pins the relationship).
pub const PASSAGE_ANN_THRESHOLD: usize = 10_000;

/// Paragraph vectors for one context, `{stem}.pvectors.bin`. Unlike
/// the gloss [`VectorStore`] this is a FLAT table — one contiguous
/// `Vec<f32>` of unit rows — because every query scans all of it (a
/// nearest-neighbor sweep has no point lookups), and at 10⁴–10⁵ rows
/// sequential prefetch is what the scan lives on. Kept separate from
/// the gloss store on purpose: glosses are small and ride every
/// resolve, paragraphs are big and ride only passage search — one
/// file, one Arc, one budget entry each, so neither pays for the
/// other. Same derived-cache contract: corrupt means re-embed, never
/// an outage, and a model change discards it wholesale.
#[derive(Debug, Default)]
pub struct PassageVectorStore {
    pub model: String,
    dim: usize,
    keys: Vec<PassageKey>,
    data: Vec<f32>,
    /// Built lazily by the first search that needs it (see
    /// [`Self::top_matches`]) and cached for this store's lifetime —
    /// nested here rather than beside it in the registry's `Entry` so
    /// the index's lifetime is tied to the exact rows it was built
    /// from: a refresh replaces the whole `Arc<PassageVectorStore>`
    /// rather than mutating this one, so there is no separate
    /// generation to invalidate this against.
    ann: Mutex<Option<Arc<PassageAnnIndex>>>,
}

impl PassageVectorStore {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
        }
    }

    /// Appends one row. The first row fixes the dimension; a
    /// mismatched later row is a provider bug and is dropped loudly
    /// rather than corrupting the flat layout.
    pub fn push(&mut self, key: PassageKey, vector: Vec<f32>) {
        if self.keys.is_empty() {
            self.dim = vector.len();
        }
        if vector.len() != self.dim || self.dim == 0 {
            tracing::warn!(
                "dropping a {}-dim passage vector from a {}-dim store",
                vector.len(),
                self.dim
            );
            return;
        }
        self.keys.push(key);
        self.data.extend_from_slice(&vector);
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Every (key, unit row) pair, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&PassageKey, &[f32])> {
        self.keys
            .iter()
            .zip(self.data.chunks_exact(self.dim.max(1)))
    }

    /// Rough resident bytes when held in memory, for the cache budget.
    pub fn footprint(&self) -> usize {
        const KEY_OVERHEAD: usize = 48;
        self.data.len() * 4
            + self
                .keys
                .iter()
                .map(|key| key.source.len() + KEY_OVERHEAD)
                .sum::<usize>()
    }

    /// Top `limit` rows by cosine against a unit query. Ties break by
    /// (source, index) ascending for deterministic output; a query of
    /// the wrong dimension matches nothing.
    ///
    /// A caller asking for every row (`limit >= len()`, as
    /// `explain_passage_search` always does) always gets an exact
    /// sweep — the only way to honestly return every row is to look at
    /// every row. Below [`PASSAGE_ANN_THRESHOLD`] every caller does,
    /// full stop. At or above it, a narrower request instead consults
    /// a [`PassageAnnIndex`] built lazily here (within `deadline`) and
    /// cached on this store for as long as it lives. A `deadline` too
    /// tight to build the index yet falls back to the same exact sweep
    /// — approximation is an optimization here, never a requirement.
    pub fn top_matches(
        &self,
        query: &[f32],
        limit: usize,
        deadline: Deadline,
    ) -> Vec<(&PassageKey, f32)> {
        if self.dim == 0 || query.len() != self.dim {
            return Vec::new();
        }
        if limit < self.len()
            && self.len() >= PASSAGE_ANN_THRESHOLD
            && let Some(index) = self.ensure_ann_index(deadline)
        {
            return index.search(self, query, limit);
        }
        self.exact_top_matches(query, limit)
    }

    /// The linear cosine sweep [`Self::top_matches`] takes below
    /// [`PASSAGE_ANN_THRESHOLD`], and always for a `limit` covering the
    /// whole store — the flat layout alone IS a fine index at that
    /// scale or for that shape of query.
    fn exact_top_matches(&self, query: &[f32], limit: usize) -> Vec<(&PassageKey, f32)> {
        let mut scored: Vec<(&PassageKey, f32)> = self
            .iter()
            .map(|(key, row)| (key, similarity(query, row)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.source.cmp(&b.0.source))
                .then_with(|| a.0.index.cmp(&b.0.index))
        });
        scored.truncate(limit);
        scored
    }

    /// The cached index, building it first if this is the first search
    /// to need one. `None` only when `deadline` is already spent —
    /// the caller falls back to the exact sweep for just this call and
    /// tries again next time.
    fn ensure_ann_index(&self, deadline: Deadline) -> Option<Arc<PassageAnnIndex>> {
        let mut cached = self.ann.lock();
        if let Some(index) = &*cached {
            return Some(Arc::clone(index));
        }
        if deadline.expired() {
            return None;
        }
        let index = Arc::new(PassageAnnIndex::build(self, deadline));
        *cached = Some(Arc::clone(&index));
        Some(index)
    }

    /// Reads the sidecar, returning an empty store on any problem — a
    /// corrupt vector cache costs a re-embed, never an outage. A
    /// pre-doc2query file (TAGURUP1) is discarded the same way, minus
    /// the alarm: it is an upgrade, not corruption.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_bytes(&bytes).unwrap_or_else(|| {
                if bytes.get(..8) == Some(LEGACY_PASSAGE_VECTOR_MAGIC.as_slice()) {
                    tracing::info!(
                        "passage vectors at {} predate doc2query; re-embedding",
                        path.display()
                    );
                } else {
                    tracing::warn!(
                        "ignoring corrupt passage vector store at {}",
                        path.display()
                    );
                }
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        crate::storage::write_atomic(path, &self.to_bytes())
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PASSAGE_VECTOR_MAGIC);
        write_string(&mut out, &self.model);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for key in &self.keys {
            write_string(&mut out, &key.source);
            out.extend_from_slice(&key.index.to_le_bytes());
            out.extend_from_slice(&key.hash.to_le_bytes());
            match key.question_hash {
                Some(question_hash) => {
                    out.push(1);
                    out.extend_from_slice(&question_hash.to_le_bytes());
                }
                None => out.push(0),
            }
        }
        for value in &self.data {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        if bytes.get(..8)? != PASSAGE_VECTOR_MAGIC {
            return None;
        }
        pos += 8;
        let model = read_string(bytes, &mut pos)?;
        let dim = read_u32(bytes, &mut pos)? as usize;
        let count = read_u32(bytes, &mut pos)? as usize;
        // Rows without a dimension cannot exist through push(); a file
        // claiming them is corrupt, not merely empty.
        if count > 0 && dim == 0 {
            return None;
        }
        let mut keys = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            let source = read_string(bytes, &mut pos)?;
            let index = read_u32(bytes, &mut pos)?;
            let hash_bytes = bytes.get(pos..pos + 8)?;
            pos += 8;
            let hash = u64::from_le_bytes(hash_bytes.try_into().ok()?);
            let question_hash = match bytes.get(pos)? {
                0 => {
                    pos += 1;
                    None
                }
                1 => {
                    pos += 1;
                    let question_bytes = bytes.get(pos..pos + 8)?;
                    pos += 8;
                    Some(u64::from_le_bytes(question_bytes.try_into().ok()?))
                }
                _ => return None,
            };
            keys.push(PassageKey {
                source,
                index,
                hash,
                question_hash,
            });
        }
        let mut data = Vec::with_capacity((count * dim).min(1 << 26));
        for _ in 0..count * dim {
            let slice = bytes.get(pos..pos + 4)?;
            pos += 4;
            let value = f32::from_le_bytes(slice.try_into().ok()?);
            // NaN/inf can only reach this file through corruption or a
            // hand-edited sidecar — the write path (`normalize`, the
            // HTTP boundary above) never produces one. Carried into
            // `similarity`'s dot product it poisons every score it
            // touches; `total_cmp` then sorts that poisoned row as the
            // maximum instead of a candidate that would fail cleanly.
            if !value.is_finite() {
                return None;
            }
            data.push(value);
        }
        (pos == bytes.len()).then_some(Self {
            model,
            dim,
            keys,
            data,
            ann: Mutex::new(None),
        })
    }
}

/// How many extra candidate rows [`PassageAnnIndex::search`] gathers
/// beyond `limit` before scoring exactly, to absorb the
/// approximation's ragged edges — a true top-`limit` row landing in
/// the 2nd- or 3rd-nearest cluster rather than the closest one.
const PASSAGE_ANN_OVERSAMPLE: usize = 8;
/// A floor under the oversample product so a small `limit` (a narrow
/// rerank pool) still probes enough of the index for reasonable
/// recall.
const PASSAGE_ANN_MIN_CANDIDATES: usize = 256;

/// A coarse IVF (inverted-file) index over one [`PassageVectorStore`]'s
/// rows: past [`PASSAGE_ANN_THRESHOLD`] rows, sweeping every one of
/// them on every query stops paying for itself, so
/// [`PassageVectorStore::top_matches`] narrows the sweep to whichever
/// clusters are nearest the query instead. Hand-rolled rather than an
/// external ANN crate, matching every other binary index in this
/// codebase (BM25, the graph image itself).
///
/// Centroids come from one deterministic farthest-point pass (greedy
/// k-center: pick a seed row, then repeatedly add whichever remaining
/// row is least similar to every centroid chosen so far) rather than
/// iterated k-means — no RNG, no non-convergence to guard against, and
/// a single `O(centroids × rows)` sweep gets both the centroids AND
/// (for free, as a side effect of tracking each row's best centroid so
/// far) every row's list assignment in the same pass.
///
/// Approximate by construction — see `search`'s oversample — so this
/// must never back an exact-ranking contract. `top_matches` keeps that
/// guarantee structurally, via its own `limit >= len()` brute-force
/// branch, not by asking this type to be exact.
#[derive(Debug)]
struct PassageAnnIndex {
    /// Centroid rows, flat like `PassageVectorStore::data`:
    /// `centroids.len() / dim` of them.
    centroids: Vec<f32>,
    /// Row indices under each centroid; `lists.len() == centroids.len() / dim`.
    lists: Vec<Vec<u32>>,
}

impl PassageAnnIndex {
    /// One coarse cluster per ~sqrt(rows) — the classic IVF rule of
    /// thumb, keeping both the centroid sweep and the average list
    /// size close to `sqrt(rows)` so neither half of a query's cost
    /// dominates the other.
    fn centroid_count(rows: usize) -> usize {
        (rows as f64).sqrt().round().max(1.0) as usize
    }

    /// Builds centroids and inverted lists in one pass, stopping early
    /// (with however many centroids it has so far) if `deadline` runs
    /// out — callers that need every row exactly never reach here (see
    /// [`PassageVectorStore::top_matches`]), so a coarser-than-intended
    /// index only ever costs recall, never correctness.
    fn build(store: &PassageVectorStore, deadline: Deadline) -> Self {
        let n = store.len();
        let dim = store.dim.max(1);
        let target = Self::centroid_count(n);
        let row = |i: usize| -> &[f32] { &store.data[i * dim..i * dim + dim] };

        let mut centroids: Vec<f32> = Vec::with_capacity(target * dim);
        // Running "similarity to the nearest centroid chosen so far"
        // per row, and which centroid that was — updated incrementally
        // as each new centroid is added, so the final pass over these
        // two arrays IS the list assignment, no separate pass needed.
        let mut best_sim = vec![f32::NEG_INFINITY; n];
        let mut best_centroid = vec![0u32; n];
        let mut seed = 0usize;
        let mut built = 0usize;
        while built < target {
            if built > 0 && deadline.expired() {
                break;
            }
            let centroid_row = row(seed);
            centroids.extend_from_slice(centroid_row);
            let id = built as u32;
            for (i, slot) in best_sim.iter_mut().enumerate() {
                let sim = similarity(row(i), centroid_row);
                if sim > *slot {
                    *slot = sim;
                    best_centroid[i] = id;
                }
            }
            built += 1;
            if built < target {
                // The farthest-from-everything-chosen-so-far row is
                // the least-committed candidate for the next cluster.
                seed = (0..n)
                    .min_by(|&a, &b| best_sim[a].total_cmp(&best_sim[b]))
                    .unwrap_or(0);
            }
        }

        let mut lists = vec![Vec::new(); built];
        for (i, &centroid) in best_centroid.iter().enumerate() {
            lists[centroid as usize].push(i as u32);
        }
        Self { centroids, lists }
    }

    fn centroid(&self, id: usize, dim: usize) -> &[f32] {
        &self.centroids[id * dim..id * dim + dim]
    }

    /// Scores the rows under whichever centroids are nearest `query`,
    /// growing the probe cluster-by-cluster until there are enough
    /// candidates for `limit` to be a meaningful cutoff (or every
    /// cluster is in), then ranks exactly within just that candidate
    /// set — same tie-break as [`PassageVectorStore::exact_top_matches`].
    fn search<'a>(
        &self,
        store: &'a PassageVectorStore,
        query: &[f32],
        limit: usize,
    ) -> Vec<(&'a PassageKey, f32)> {
        let dim = store.dim.max(1);
        let mut order: Vec<(u32, f32)> = (0..self.lists.len())
            .map(|id| (id as u32, similarity(query, self.centroid(id, dim))))
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1));

        let wanted = limit
            .saturating_mul(PASSAGE_ANN_OVERSAMPLE)
            .max(PASSAGE_ANN_MIN_CANDIDATES);
        let mut candidates: Vec<u32> = Vec::new();
        for (id, _) in order {
            candidates.extend_from_slice(&self.lists[id as usize]);
            if candidates.len() >= wanted {
                break;
            }
        }

        let mut scored: Vec<(&PassageKey, f32)> = candidates
            .into_iter()
            .map(|row| {
                let row = row as usize;
                let vector = &store.data[row * dim..row * dim + dim];
                (&store.keys[row], similarity(query, vector))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.source.cmp(&b.0.source))
                .then_with(|| a.0.index.cmp(&b.0.index))
        });
        scored.truncate(limit);
        scored
    }
}

fn write_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn write_table(out: &mut Vec<u8>, table: &VectorTable) {
    out.extend_from_slice(&(table.len() as u32).to_le_bytes());
    // Sorted for byte-stable files under identical content.
    let mut names: Vec<&String> = table.keys().collect();
    names.sort();
    for name in names {
        let (hash, vector) = &table[name];
        write_string(out, name);
        out.extend_from_slice(&hash.to_le_bytes());
        out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        for value in vector {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn read_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    let len = read_u32(bytes, pos)? as usize;
    let slice = bytes.get(*pos..*pos + len)?;
    *pos += len;
    String::from_utf8(slice.to_vec()).ok()
}

fn read_table(bytes: &[u8], pos: &mut usize) -> Option<VectorTable> {
    let count = read_u32(bytes, pos)? as usize;
    let mut table = HashMap::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        let name = read_string(bytes, pos)?;
        let hash_bytes = bytes.get(*pos..*pos + 8)?;
        *pos += 8;
        let hash = u64::from_le_bytes(hash_bytes.try_into().ok()?);
        let dim = read_u32(bytes, pos)? as usize;
        let mut vector = Vec::with_capacity(dim.min(1 << 16));
        for _ in 0..dim {
            let slice = bytes.get(*pos..*pos + 4)?;
            *pos += 4;
            let value = f32::from_le_bytes(slice.try_into().ok()?);
            // Same guard as `PassageVectorStore::from_bytes`: a NaN or
            // infinite component poisons every `similarity` dot product
            // it touches, and `total_cmp` then sorts that poisoned row
            // as the maximum instead of failing the load cleanly.
            if !value.is_finite() {
                return None;
            }
            vector.push(value);
        }
        table.insert(name, (hash, vector));
    }
    Some(table)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let slice = bytes.get(*pos..*pos + 4)?;
    *pos += 4;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{FNV1A_OFFSET, fnv1a_fold};

    /// A provider aimed at a local stub listener: a fresh breaker and a
    /// never-raised shutdown flag per test, so no test can trip
    /// another's circuit.
    fn stub_provider(addr: std::net::SocketAddr, timeout: Duration) -> HttpEmbeddings {
        HttpEmbeddings {
            url: format!("http://{addr}"),
            model: "stub-model".to_string(),
            api_key: None,
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .build()
                .into(),
            timeout,
            shutdown: ShutdownFlag::default(),
            breaker: EmbedBreaker::default(),
        }
    }

    #[test]
    fn vector_store_roundtrips_and_rejects_garbage() {
        let mut store = VectorStore {
            model: "test-model".into(),
            ..Default::default()
        };
        store.concepts.insert("りんご".into(), (7, vec![1.0, 0.0]));
        store.labels.insert("好き".into(), (9, vec![0.0, 1.0]));

        let bytes = store.to_bytes();
        let loaded = VectorStore::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.concepts["りんご"], (7, vec![1.0, 0.0]));
        assert_eq!(loaded.labels["好き"], (9, vec![0.0, 1.0]));

        assert!(VectorStore::from_bytes(b"garbage").is_none());
        assert!(VectorStore::from_bytes(&bytes[..bytes.len() - 1]).is_none());
    }

    /// A NaN or infinite component can only reach this file through
    /// corruption — the write path never produces one — but if it did,
    /// it would poison every `similarity` dot product it touches and
    /// `total_cmp` would then sort that poisoned row as the maximum.
    #[test]
    fn vector_store_rejects_non_finite_components() {
        let mut store = VectorStore {
            model: "test-model".into(),
            ..Default::default()
        };
        // A distinctive value unlikely to collide with any length
        // prefix or hash elsewhere in the serialized bytes.
        let marker = 12345.678_f32;
        store.concepts.insert("a".into(), (1, vec![marker, 2.0]));

        let bytes = store.to_bytes();
        let at = bytes
            .windows(4)
            .position(|window| window == marker.to_le_bytes())
            .expect("marker component must appear in the serialized bytes");

        for corrupt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut corrupted = bytes.clone();
            corrupted[at..at + 4].copy_from_slice(&corrupt.to_le_bytes());
            assert!(VectorStore::from_bytes(&corrupted).is_none());
        }
    }

    #[test]
    fn similarity_is_a_dot_product_over_unit_vectors() {
        let mut a = vec![3.0, 4.0];
        normalize(&mut a).unwrap();
        assert!((similarity(&a, &a) - 1.0).abs() < 1e-6);
        let b = vec![-a[1], a[0]];
        assert!(similarity(&a, &b).abs() < 1e-6);
        // A width mismatch is no signal — never a truncated dot product
        // that would read as a plausible score.
        assert_eq!(similarity(&a, &[1.0]), 0.0);
        assert_eq!(similarity(&[], &a), 0.0);
    }

    #[test]
    fn normalize_leaves_a_zero_vector_untouched() {
        let mut zero = vec![0.0, 0.0, 0.0];
        normalize(&mut zero).unwrap();
        assert_eq!(zero, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_rejects_a_non_finite_norm_instead_of_leaving_it_unnormalized() {
        for corrupt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut vector = vec![corrupt, 1.0];
            assert!(
                normalize(&mut vector).is_err(),
                "{corrupt} must be rejected, not silently passed through"
            );
        }
    }

    /// Reads one whole HTTP request — headers plus the Content-Length
    /// body. Replying after the headers alone races the reply (and the
    /// close behind it) against the client's body write; under parallel
    /// test load that surfaces as a connection reset inside `embed`.
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
            let read = stream.read(&mut buffer).unwrap();
            if read == 0 {
                return request;
            }
            request.extend_from_slice(&buffer[..read]);
        }
    }

    /// Two 503s then a 200: the transient-failure retries absorb the
    /// blip and the caller sees one clean Ok — while a 4xx refusal
    /// (here 401) surfaces immediately, because retrying a rejected
    /// credential three times just triples the noise.
    #[test]
    fn transient_provider_failures_retry_and_hard_refusals_do_not() {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let _ = read_full_request(&mut stream);
                let attempt = counted.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    let _ = stream.write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                          Connection: close\r\n\r\n",
                    );
                } else {
                    let body = br#"{"data":[{"embedding":[3.0,4.0]}]}"#;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body);
                }
            }
        });
        let provider = stub_provider(addr, Duration::from_secs(5));
        let vectors = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap();
        assert_eq!(vectors, vec![vec![0.6, 0.8]]);
        assert_eq!(hits.load(Ordering::SeqCst), 3, "503, 503, then the 200");

        // The hard-refusal half: one 401, no second attempt.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                counted.fetch_add(1, Ordering::SeqCst);
                let _ = stream.write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\
                      Connection: close\r\n\r\n",
                );
            }
        });
        let provider = stub_provider(addr, Duration::from_secs(5));
        let error = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        assert!(error.contains("401"), "{error}");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "a 4xx never retries");
    }

    /// A listener that accepts, reads the request, and never answers —
    /// the black-holed provider every cancellation test needs. Held
    /// open (not closed) so the client sees silence, not a reset.
    fn black_hole() -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                std::thread::sleep(Duration::from_secs(120));
            }
        });
        addr
    }

    /// A single slow round trip used to be uncancellable — the agent's
    /// global timeout was the only ceiling, whatever the request had
    /// left. The attempt itself is now bounded by the deadline's
    /// remaining budget: a black-holed provider costs a request about
    /// its own budget, never the provider ceiling.
    #[test]
    fn an_attempt_is_bounded_by_the_request_deadline() {
        let addr = black_hole();
        let provider = stub_provider(addr, Duration::from_secs(60));
        let started = std::time::Instant::now();
        let error = provider
            .embed(
                &["x"],
                EmbedPurpose::Query,
                Deadline::after(Duration::from_millis(500)),
            )
            .unwrap_err();
        assert!(error.contains("unreachable"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the attempt must be cut near the deadline, not the 60s provider \
             ceiling: {:?}",
            started.elapsed()
        );
    }

    /// The stop signal abandons an in-flight provider wait: the drain
    /// proceeds in well under the provider timeout, and the orphaned
    /// worker thread runs out its own bounded call in the background.
    #[test]
    fn the_stop_signal_abandons_an_in_flight_call() {
        let addr = black_hole();
        let shutdown = ShutdownFlag::default();
        let mut provider = stub_provider(addr, Duration::from_secs(30));
        provider.shutdown = shutdown.clone();

        let raiser = std::thread::spawn({
            let shutdown = shutdown.clone();
            move || {
                std::thread::sleep(Duration::from_millis(300));
                shutdown.begin();
            }
        });
        let started = std::time::Instant::now();
        let error = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        raiser.join().unwrap();
        assert!(error.contains("shutting down"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the wait must end moments after the flag, not at the 30s \
             provider ceiling: {:?}",
            started.elapsed()
        );
    }

    /// One full retry ladder against a dead provider opens the breaker
    /// (attempts count, not calls), and the next call is refused
    /// without a round trip — the whole point: an outage costs each
    /// caller a refusal, not a timeout.
    #[test]
    fn the_breaker_opens_after_one_ladder_and_short_circuits_the_next_call() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                counted.fetch_add(1, Ordering::SeqCst);
                let _ = stream.write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                      Connection: close\r\n\r\n",
                );
            }
        });

        let provider = stub_provider(addr, Duration::from_secs(5));
        let error = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        assert!(error.contains("503"), "{error}");
        assert_eq!(hits.load(Ordering::SeqCst), 3, "the full ladder ran");

        let error = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        assert!(error.contains("circuit is open"), "{error}");
        assert_eq!(hits.load(Ordering::SeqCst), 3, "no fourth round trip");

        let snapshot = provider.breaker.snapshot();
        assert_eq!(snapshot.state, 1);
        assert_eq!(snapshot.opened_total, 1);
        assert!(snapshot.short_circuits_total >= 1);
    }

    /// Once the cooldown lets a probe through and the provider has
    /// recovered, the breaker closes and traffic resumes — the outage
    /// costs nothing past its own duration plus one cooldown.
    #[test]
    fn a_half_open_probe_closes_the_breaker_once_the_provider_recovers() {
        use std::io::Write;
        use std::sync::atomic::AtomicUsize;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&hits);
        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let _ = read_full_request(&mut stream);
                let attempt = counted.fetch_add(1, Ordering::SeqCst);
                if attempt < 3 {
                    let _ = stream.write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\
                          Connection: close\r\n\r\n",
                    );
                } else {
                    let body = br#"{"data":[{"embedding":[3.0,4.0]}]}"#;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(body);
                }
            }
        });

        let mut provider = stub_provider(addr, Duration::from_secs(5));
        provider.breaker = EmbedBreaker::with_policy(3, Duration::ZERO);
        let error = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        assert!(error.contains("503"), "{error}");
        assert_eq!(provider.breaker.snapshot().state, 1, "opened by the ladder");

        let vectors = provider
            .embed(&["x"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap();
        assert_eq!(vectors, vec![vec![0.6, 0.8]]);
        assert_eq!(provider.breaker.snapshot().state, 0, "the probe closed it");
        assert_eq!(
            hits.load(Ordering::SeqCst),
            4,
            "one probe, no extra traffic"
        );
    }

    /// The state machine itself: a success resets the consecutive
    /// count, one probe flies at a time, a failed probe re-opens, an
    /// abandoned probe releases its slot without judging the provider.
    #[test]
    fn the_breaker_state_machine_counts_probes_and_resets() {
        let breaker = EmbedBreaker::with_policy(2, Duration::from_secs(1000));
        let first = breaker.admit().unwrap();
        breaker.record(first, false);
        let second = breaker.admit().unwrap();
        breaker.record(second, true);
        let third = breaker.admit().unwrap();
        breaker.record(third, false);
        assert_eq!(
            breaker.snapshot().state,
            0,
            "one failure after a reset stays closed"
        );
        let fourth = breaker.admit().unwrap();
        breaker.record(fourth, false);
        assert_eq!(breaker.snapshot().state, 1, "the second in a row opens");
        assert!(breaker.admit().is_err(), "cooling: nothing admitted");
        assert!(
            breaker.refusal().is_some(),
            "the pre-flight gate refuses too"
        );

        let breaker = EmbedBreaker::with_policy(1, Duration::ZERO);
        let only = breaker.admit().unwrap();
        breaker.record(only, false);
        let probe = breaker.admit().unwrap();
        assert!(matches!(probe, Admission::Probe));
        assert!(breaker.admit().is_err(), "one probe at a time");
        breaker.record(probe, false);
        assert_eq!(
            breaker.snapshot().opened_total,
            2,
            "a failed probe re-opens"
        );
        let probe = breaker.admit().unwrap();
        breaker.abandon(probe);
        let probe = breaker
            .admit()
            .expect("an abandoned probe's slot frees immediately");
        breaker.record(probe, true);
        assert_eq!(breaker.snapshot().state, 0);
        assert!(breaker.refusal().is_none());
    }

    /// The HTTP provider must tell a bridging proxy WHY it is embedding
    /// (Cohere-style asymmetric models need `input_type`), and must
    /// normalize whatever comes back. One stub round trip checks both.
    #[test]
    fn http_embeddings_sends_the_purpose_header_and_normalizes() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_full_request(&mut stream);
            let body = br#"{"data":[{"embedding":[3.0,4.0]}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
            String::from_utf8_lossy(&request).to_string()
        });

        let provider = stub_provider(addr, Duration::from_secs(5));
        let vectors = provider
            .embed(&["こんにちは"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap();
        // 3-4-5 triangle, normalized on receipt.
        assert_eq!(vectors, vec![vec![0.6, 0.8]]);

        let request = served.join().unwrap().to_lowercase();
        assert!(
            request.contains("x-taguru-embed-purpose: query"),
            "{request}"
        );
    }

    /// A provider (or a proxy in front of one) that returns an
    /// oversized body must be refused at the byte cap, never buffered
    /// whole — the agent's timeout bounds seconds, not bytes.
    #[test]
    fn an_oversized_embedding_response_is_refused_not_buffered() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_full_request(&mut stream);
            // 65 MiB claimed and streamed: one past the reader's cap.
            // The client stops reading there and hangs up, so tolerate
            // the broken pipe instead of unwrapping it.
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                65 * 1024 * 1024
            );
            if stream.write_all(head.as_bytes()).is_err() {
                return;
            }
            let chunk = vec![b'x'; 1024 * 1024];
            for _ in 0..65 {
                if stream.write_all(&chunk).is_err() {
                    return;
                }
            }
        });

        let provider = stub_provider(addr, Duration::from_secs(10));
        let error = provider
            .embed(&["こんにちは"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        assert!(error.contains("refusing to buffer"), "{error}");
    }

    /// A provider that returns `data` out of input order is realigned by
    /// each entry's `index`, never trusted in array order — otherwise
    /// every embedding would pair with the wrong text.
    #[test]
    fn http_embeddings_realigns_a_reordered_response_by_index() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_full_request(&mut stream);
            // Wire order is the REVERSE of the input order: index 1 first.
            let body = br#"{"data":[{"index":1,"embedding":[0.0,2.0]},{"index":0,"embedding":[3.0,4.0]}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });

        let provider = stub_provider(addr, Duration::from_secs(5));
        let vectors = provider
            .embed(
                &["first", "second"],
                EmbedPurpose::Query,
                Deadline::unbounded(),
            )
            .unwrap();
        served.join().unwrap();
        // Input 0 keeps index 0's vector (3-4-5 → 0.6, 0.8); input 1 keeps
        // index 1's (0, 2 → 0, 1). Trusting array order would swap them.
        assert_eq!(vectors, vec![vec![0.6, 0.8], vec![0.0, 1.0]]);
    }

    /// One response, one model, one width: an entry whose vector width
    /// disagrees with the rest is a provider bug named at the boundary,
    /// not carried into stores for `similarity` to score as nothing.
    #[test]
    fn http_embeddings_reject_a_mixed_width_response() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_full_request(&mut stream);
            let body =
                br#"{"data":[{"index":0,"embedding":[3.0,4.0]},{"index":1,"embedding":[1.0]}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });

        let provider = stub_provider(addr, Duration::from_secs(5));
        let error = provider
            .embed(
                &["first", "second"],
                EmbedPurpose::Query,
                Deadline::unbounded(),
            )
            .unwrap_err();
        served.join().unwrap();
        assert!(error.contains("widths"), "{error}");
    }

    /// A non-numeric component must fail the whole response, the same as
    /// a missing vector or a mixed width — never fall back to 0.0 and
    /// carry a silently corrupted vector into the store.
    #[test]
    fn http_embeddings_reject_a_non_numeric_component() {
        use std::io::Write;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_full_request(&mut stream);
            let body = br#"{"data":[{"index":0,"embedding":[3.0,"oops"]}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });

        let provider = stub_provider(addr, Duration::from_secs(5));
        let error = provider
            .embed(&["first"], EmbedPurpose::Query, Deadline::unbounded())
            .unwrap_err();
        served.join().unwrap();
        assert!(error.contains("non-numeric"), "{error}");
    }

    #[test]
    fn passage_vector_store_round_trips_and_rejects_garbage() {
        let mut store = PassageVectorStore::new("test-model");
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 0,
                hash: 7,
                question_hash: None,
            },
            vec![1.0, 0.0],
        );
        // A doc2query question row: SAME paragraph key, its own
        // discriminator — two rows for one paragraph must both survive
        // the round trip.
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 0,
                hash: 7,
                question_hash: Some(41),
            },
            vec![0.6, 0.8],
        );
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 1,
                hash: 9,
                question_hash: None,
            },
            vec![0.0, 1.0],
        );
        // A wrong-dimension row is dropped, not silently misaligned.
        store.push(
            PassageKey {
                source: "bad".into(),
                index: 0,
                hash: 1,
                question_hash: None,
            },
            vec![1.0, 0.0, 0.0],
        );
        assert_eq!(store.len(), 3);

        let bytes = store.to_bytes();
        let loaded = PassageVectorStore::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.len(), 3);
        let rows: Vec<(&PassageKey, &[f32])> = loaded.iter().collect();
        assert_eq!(rows[0].0.index, 0);
        assert_eq!(rows[0].0.question_hash, None);
        assert_eq!(rows[0].1, &[1.0, 0.0]);
        assert_eq!(rows[1].0.question_hash, Some(41));
        assert_eq!(rows[1].1, &[0.6, 0.8]);
        assert_eq!(rows[2].0.hash, 9);
        assert_eq!(rows[2].1, &[0.0, 1.0]);

        assert!(PassageVectorStore::from_bytes(b"garbage").is_none());
        assert!(PassageVectorStore::from_bytes(&bytes[..bytes.len() - 1]).is_none());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(PassageVectorStore::from_bytes(&padded).is_none());
    }

    /// Same guard as `vector_store_rejects_non_finite_components`, for
    /// the flat passage-vector layout.
    #[test]
    fn passage_vector_store_rejects_non_finite_components() {
        let mut store = PassageVectorStore::new("test-model");
        let marker = 12345.678_f32;
        store.push(
            PassageKey {
                source: "docs/aomine.md".into(),
                index: 0,
                hash: 7,
                question_hash: None,
            },
            vec![marker, 2.0],
        );

        let bytes = store.to_bytes();
        let at = bytes
            .windows(4)
            .position(|window| window == marker.to_le_bytes())
            .expect("marker component must appear in the serialized bytes");

        for corrupt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut corrupted = bytes.clone();
            corrupted[at..at + 4].copy_from_slice(&corrupt.to_le_bytes());
            assert!(PassageVectorStore::from_bytes(&corrupted).is_none());
        }
    }

    // -----------------------------------------------------------------
    // Passage ANN index
    // -----------------------------------------------------------------

    /// A small, fully deterministic "random-looking" unit vector, built
    /// from [`fnv1a_fold`] rather than an RNG crate — every ANN test
    /// below needs reproducible data, not entropy.
    fn deterministic_unit_vector(seed: u64, dim: usize) -> Vec<f32> {
        let base = fnv1a_fold(FNV1A_OFFSET, seed.to_le_bytes());
        let mut v: Vec<f32> = (0..dim as u64)
            .map(|d| {
                let bits = fnv1a_fold(base, d.to_le_bytes());
                (bits as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn synthetic_passage_store(rows: usize, dim: usize) -> PassageVectorStore {
        let mut store = PassageVectorStore::new("ann-test-model");
        for i in 0..rows {
            store.push(
                PassageKey {
                    source: format!("doc-{i}"),
                    index: 0,
                    hash: i as u64,
                    question_hash: None,
                },
                deterministic_unit_vector(i as u64, dim),
            );
        }
        store
    }

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// Rows drawn around a few dozen topic centroids — the shape real
    /// text embeddings take, and the structure IVF recall lives on;
    /// [`synthetic_passage_store`]'s uniform rows are its structureless
    /// counterpart, the adversarial floor for the same measurements.
    fn clustered_passage_store(rows: usize, dim: usize) -> PassageVectorStore {
        const CENTERS: u64 = 48;
        let mut store = PassageVectorStore::new("ann-test-model");
        for i in 0..rows {
            let center = deterministic_unit_vector(0xC_000_000 + (i as u64 % CENTERS), dim);
            let noise = deterministic_unit_vector(i as u64, dim);
            let v = unit(
                center
                    .iter()
                    .zip(&noise)
                    .map(|(c, n)| c + 0.55 * n)
                    .collect(),
            );
            store.push(
                PassageKey {
                    source: format!("doc-{i}"),
                    index: 0,
                    hash: i as u64,
                    question_hash: None,
                },
                v,
            );
        }
        store
    }

    /// Sorted descending by score, same tie-break as
    /// [`PassageVectorStore::exact_top_matches`] — computed independently
    /// here (not by calling that private method) so these tests stay an
    /// honest ground truth rather than checking the code against itself.
    fn brute_force_top<'a>(
        store: &'a PassageVectorStore,
        query: &[f32],
        limit: usize,
    ) -> Vec<(&'a PassageKey, f32)> {
        let mut scored: Vec<(&PassageKey, f32)> = store
            .iter()
            .map(|(key, row)| (key, similarity(query, row)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.source.cmp(&b.0.source))
                .then_with(|| a.0.index.cmp(&b.0.index))
        });
        scored.truncate(limit);
        scored
    }

    #[test]
    fn passage_ann_index_centroid_count_grows_like_sqrt_of_rows() {
        assert_eq!(PassageAnnIndex::centroid_count(1), 1);
        assert_eq!(PassageAnnIndex::centroid_count(100), 10);
        assert_eq!(PassageAnnIndex::centroid_count(PASSAGE_ANN_THRESHOLD), 100);
    }

    #[test]
    fn passage_ann_index_build_partitions_every_row_into_exactly_one_list() {
        let rows = 500;
        let store = synthetic_passage_store(rows, 6);
        let index = PassageAnnIndex::build(&store, Deadline::unbounded());
        assert_eq!(index.lists.len(), PassageAnnIndex::centroid_count(rows));

        let mut seen = vec![false; rows];
        let mut total = 0usize;
        for list in &index.lists {
            for &row in list {
                assert!(!seen[row as usize], "row {row} assigned to two lists");
                seen[row as usize] = true;
                total += 1;
            }
        }
        assert_eq!(total, rows);
        assert!(
            seen.into_iter().all(|hit| hit),
            "every row must land somewhere"
        );
    }

    #[test]
    fn passage_ann_index_search_is_sorted_descending_and_respects_limit() {
        let store = synthetic_passage_store(2_000, 8);
        let index = PassageAnnIndex::build(&store, Deadline::unbounded());
        let query = deterministic_unit_vector(999_999, 8);
        let hits = index.search(&store, &query, 15);
        assert_eq!(hits.len(), 15);
        for window in hits.windows(2) {
            assert!(window[0].1 >= window[1].1, "{hits:?}");
        }
    }

    #[test]
    fn top_matches_stays_exact_below_the_ann_threshold() {
        let dim = 6;
        let store = synthetic_passage_store(64, dim);
        let query = deterministic_unit_vector(777, dim);

        let expected = brute_force_top(&store, &query, 10);
        let actual = store.top_matches(&query, 10, Deadline::unbounded());
        assert_eq!(
            actual.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            expected.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
        );
        assert!(
            store.ann.lock().is_none(),
            "below threshold must never build the index"
        );
    }

    #[test]
    fn explain_style_calls_bypass_the_ann_index_even_above_the_threshold() {
        let dim = 12;
        let store = synthetic_passage_store(PASSAGE_ANN_THRESHOLD, dim);
        let query = deterministic_unit_vector(0x5EED, dim);

        let expected = brute_force_top(&store, &query, store.len());
        let actual = store.top_matches(&query, usize::MAX, Deadline::unbounded());
        assert_eq!(actual.len(), store.len());
        assert_eq!(
            actual.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            expected.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
        );
        assert!(
            store.ann.lock().is_none(),
            "limit >= len() must never touch the index"
        );
    }

    #[test]
    fn top_matches_falls_back_to_the_exact_sweep_when_the_deadline_is_already_spent() {
        let dim = 12;
        let store = synthetic_passage_store(PASSAGE_ANN_THRESHOLD, dim);
        let query = deterministic_unit_vector(0xFACE, dim);

        let expected = brute_force_top(&store, &query, 20);
        let spent = Deadline::after(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(1));
        assert!(spent.expired());

        let actual = store.top_matches(&query, 20, spent);
        assert_eq!(
            actual.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            expected.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
        );
        assert!(
            store.ann.lock().is_none(),
            "a spent deadline must skip building the index, not error"
        );
    }

    #[test]
    fn top_matches_uses_the_ann_index_at_the_threshold_and_finds_planted_matches() {
        let dim = 16;
        let mut store = synthetic_passage_store(PASSAGE_ANN_THRESHOLD, dim);
        let query = deterministic_unit_vector(0xA11CE, dim);
        // Ten scattered rows (including row 0, the deterministic first
        // centroid seed, and the last row) become exact copies of the
        // query: an unambiguous top-10 the index must surface in full,
        // not just "mostly". Spread relative to the threshold so the
        // fixture keeps covering the whole store when the threshold is
        // recalibrated.
        let planted: Vec<usize> = (0..10)
            .map(|k| k * (PASSAGE_ANN_THRESHOLD - 1) / 9)
            .collect();
        for &row in &planted {
            let start = row * dim;
            store.data[start..start + dim].copy_from_slice(&query);
        }
        let mut expected_keys: Vec<&PassageKey> =
            planted.iter().map(|&row| &store.keys[row]).collect();
        expected_keys.sort_by(|a, b| a.source.cmp(&b.source).then_with(|| a.index.cmp(&b.index)));

        let hits = store.top_matches(&query, planted.len(), Deadline::unbounded());
        assert!(
            store.ann.lock().is_some(),
            "a store this large, asked for far fewer rows than it has, must engage the index"
        );
        assert_eq!(
            hits.iter().map(|&(k, _)| k).collect::<Vec<_>>(),
            expected_keys,
            "every planted exact match must surface"
        );
        for &(_, score) in &hits {
            assert!(score > 0.999, "{score}");
        }
    }

    #[test]
    fn top_matches_has_reasonable_recall_against_the_exact_sweep() {
        let dim = 16;
        let top = 50;
        let store = synthetic_passage_store(PASSAGE_ANN_THRESHOLD, dim);
        let query = deterministic_unit_vector(0xC0FFEE, dim);

        let expected: std::collections::HashSet<(String, u32)> =
            brute_force_top(&store, &query, top)
                .into_iter()
                .map(|(k, _)| (k.source.clone(), k.index))
                .collect();
        let actual = store.top_matches(&query, top, Deadline::unbounded());
        assert_eq!(actual.len(), top);
        let overlap = actual
            .iter()
            .filter(|&&(k, _)| expected.contains(&(k.source.clone(), k.index)))
            .count();
        // Observed recall on this deterministic fixture is 50/50; 80% is
        // a comfortable margin under that so this stays a "still high
        // recall" regression guard rather than a brittle exact pin.
        assert!(
            overlap * 100 >= top * 80,
            "recall too low against the exact sweep: {overlap}/{top}"
        );
    }

    #[test]
    #[ignore = "wall-clock comparison for manual inspection: cargo test -- --ignored --nocapture"]
    fn passage_ann_index_is_faster_than_the_exact_sweep_at_scale() {
        let dim = 32;
        let rows = 300_000;
        let store = synthetic_passage_store(rows, dim);
        let query = deterministic_unit_vector(0xBEEF, dim);

        let started = std::time::Instant::now();
        let exact = store.exact_top_matches(&query, 50);
        let exact_elapsed = started.elapsed();

        // Warm-up: the first call pays the one-time build cost, which
        // this benchmark is not measuring.
        store.top_matches(&query, 50, Deadline::unbounded());
        let started = std::time::Instant::now();
        let approx = store.top_matches(&query, 50, Deadline::unbounded());
        let approx_elapsed = started.elapsed();

        eprintln!(
            "exact sweep: {exact_elapsed:?} for {rows} rows; ann search: {approx_elapsed:?} (after a one-time build)"
        );
        assert_eq!(exact.len(), approx.len());
        assert!(
            approx_elapsed < exact_elapsed,
            "expected the index to beat a full sweep at {rows} rows: exact={exact_elapsed:?} ann={approx_elapsed:?}"
        );
    }

    /// The data [`PASSAGE_ANN_THRESHOLD`] was calibrated on: exact
    /// sweep vs ANN (one-time build + per-query search) wall clock and
    /// ANN recall, swept across corpus sizes bracketing the threshold,
    /// at realistic embedding dimensions, over both data shapes that
    /// bound reality — topic-clustered rows and structureless uniform
    /// ones. Queries are perturbed copies of stored rows (the
    /// paraphrase case the lane exists for). Run it in release or the
    /// numbers lie:
    /// `cargo test --release passage_ann_threshold_calibration -- --ignored --nocapture`
    #[test]
    #[ignore = "calibration sweep for manual inspection: cargo test --release -- --ignored --nocapture"]
    fn passage_ann_threshold_calibration_sweep() {
        const QUERIES: usize = 20;
        // lane_pool(5): what the semantic lane actually asks for at the
        // default serve limit.
        const SERVE_POOL: usize = 50;
        for &dim in &[768usize, 1536] {
            for &rows in &[5_000usize, 10_000, 20_000, 50_000] {
                for clustered in [true, false] {
                    let store = if clustered {
                        clustered_passage_store(rows, dim)
                    } else {
                        synthetic_passage_store(rows, dim)
                    };
                    let queries: Vec<Vec<f32>> = (0..QUERIES)
                        .map(|j| {
                            let row = (j * rows) / QUERIES;
                            let base = &store.data[row * dim..(row + 1) * dim];
                            let noise = deterministic_unit_vector(0x9_000_000 + j as u64, dim);
                            unit(base.iter().zip(&noise).map(|(b, n)| b + 0.6 * n).collect())
                        })
                        .collect();

                    let mut exact_total = Duration::ZERO;
                    let truths: Vec<Vec<(&PassageKey, f32)>> = queries
                        .iter()
                        .map(|q| {
                            let at = std::time::Instant::now();
                            let hits = store.exact_top_matches(q, SERVE_POOL);
                            exact_total += at.elapsed();
                            hits
                        })
                        .collect();

                    let at = std::time::Instant::now();
                    let index = PassageAnnIndex::build(&store, Deadline::unbounded());
                    let build = at.elapsed();

                    let mut ann_total = Duration::ZERO;
                    let mut hit10 = 0usize;
                    let mut hit50 = 0usize;
                    for (q, truth) in queries.iter().zip(&truths) {
                        let at = std::time::Instant::now();
                        let hits = index.search(&store, q, SERVE_POOL);
                        ann_total += at.elapsed();
                        let id = |key: &PassageKey| (key.source.clone(), key.index);
                        let want10: std::collections::HashSet<_> =
                            truth.iter().take(10).map(|&(k, _)| id(k)).collect();
                        let want50: std::collections::HashSet<_> =
                            truth.iter().map(|&(k, _)| id(k)).collect();
                        hit10 += hits
                            .iter()
                            .take(10)
                            .filter(|&&(k, _)| want10.contains(&id(k)))
                            .count();
                        hit50 += hits
                            .iter()
                            .filter(|&&(k, _)| want50.contains(&id(k)))
                            .count();
                    }
                    eprintln!(
                        "dim {dim:>4} rows {rows:>6} {} | exact {:>10.1?}/q | ann {:>9.1?}/q build {:>10.1?} | recall@10 {:>5.1}% @50 {:>5.1}%",
                        if clustered { "clustered" } else { "uniform  " },
                        exact_total / QUERIES as u32,
                        ann_total / QUERIES as u32,
                        build,
                        hit10 as f64 * 100.0 / (10 * QUERIES) as f64,
                        hit50 as f64 * 100.0 / (SERVE_POOL * QUERIES) as f64,
                    );
                }
            }
        }
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn proptest_config() -> ProptestConfig {
            let cases = std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            ProptestConfig {
                cases,
                ..ProptestConfig::default()
            }
        }

        fn finite_f32_strategy() -> impl Strategy<Value = f32> {
            -1000.0f32..1000.0f32
        }

        fn name_strategy() -> impl Strategy<Value = String> {
            prop_oneof![
                Just("りんご"),
                Just("バナナ"),
                Just("好き"),
                Just("嫌い"),
                Just("concept-a"),
                Just("concept-b"),
                Just("label-x"),
            ]
            .prop_map(|s| s.to_string())
        }

        fn vector_table_strategy() -> impl Strategy<Value = VectorTable> {
            prop::collection::hash_map(
                name_strategy(),
                (
                    any::<u64>(),
                    prop::collection::vec(finite_f32_strategy(), 0..8),
                ),
                0..8,
            )
        }

        fn vector_store_strategy() -> impl Strategy<Value = VectorStore> {
            (
                "[a-z0-9-]{1,12}",
                vector_table_strategy(),
                vector_table_strategy(),
            )
                .prop_map(|(model, concepts, labels)| VectorStore {
                    model,
                    concepts,
                    labels,
                })
        }

        fn passage_key_strategy() -> impl Strategy<Value = PassageKey> {
            (
                "[a-z/.]{1,12}",
                any::<u32>(),
                any::<u64>(),
                prop::option::of(any::<u64>()),
            )
                .prop_map(|(source, index, hash, question_hash)| PassageKey {
                    source,
                    index,
                    hash,
                    question_hash,
                })
        }

        /// Every row in one store shares a dimension (as `push` enforces),
        /// so the dimension is fixed once and then threaded through every
        /// generated row.
        fn passage_rows_strategy() -> impl Strategy<Value = Vec<(PassageKey, Vec<f32>)>> {
            (1usize..6).prop_flat_map(|dim| {
                prop::collection::vec(
                    (
                        passage_key_strategy(),
                        prop::collection::vec(finite_f32_strategy(), dim..=dim),
                    ),
                    0..8,
                )
            })
        }

        proptest! {
            #![proptest_config(proptest_config())]

            #[test]
            fn vector_store_round_trips_through_bytes(store in vector_store_strategy()) {
                let bytes = store.to_bytes();
                let loaded = VectorStore::from_bytes(&bytes)
                    .expect("a freshly serialized store must always decode");
                prop_assert_eq!(loaded.model, store.model);
                prop_assert_eq!(loaded.concepts, store.concepts);
                prop_assert_eq!(loaded.labels, store.labels);
            }

            #[test]
            fn vector_store_from_bytes_never_panics_on_arbitrary_bytes(
                bytes in prop::collection::vec(any::<u8>(), 0..512),
            ) {
                let _ = VectorStore::from_bytes(&bytes);
            }

            #[test]
            fn vector_store_from_bytes_never_panics_on_mutated_valid_bytes(
                store in vector_store_strategy(),
                mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..16),
            ) {
                let mut bytes = store.to_bytes();
                for (pick, value) in mutations {
                    *pick.get_mut(&mut bytes) = value;
                }
                let _ = VectorStore::from_bytes(&bytes);
            }

            #[test]
            fn passage_vector_store_round_trips_through_bytes(rows in passage_rows_strategy()) {
                let mut store = PassageVectorStore::new("test-model");
                for (key, vector) in rows.clone() {
                    store.push(key, vector);
                }
                prop_assert_eq!(store.len(), rows.len(), "matching dims never get dropped");

                let bytes = store.to_bytes();
                let loaded = PassageVectorStore::from_bytes(&bytes)
                    .expect("a freshly serialized store must always decode");
                prop_assert_eq!(&loaded.model, "test-model");
                let actual: Vec<(PassageKey, Vec<f32>)> = loaded
                    .iter()
                    .map(|(key, vector)| (key.clone(), vector.to_vec()))
                    .collect();
                prop_assert_eq!(actual, rows);
            }

            #[test]
            fn passage_vector_store_from_bytes_never_panics_on_arbitrary_bytes(
                bytes in prop::collection::vec(any::<u8>(), 0..512),
            ) {
                let _ = PassageVectorStore::from_bytes(&bytes);
            }

            #[test]
            fn passage_vector_store_from_bytes_never_panics_on_mutated_valid_bytes(
                rows in passage_rows_strategy(),
                mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..16),
            ) {
                let mut store = PassageVectorStore::new("test-model");
                for (key, vector) in rows {
                    store.push(key, vector);
                }
                let mut bytes = store.to_bytes();
                for (pick, value) in mutations {
                    *pick.get_mut(&mut bytes) = value;
                }
                let _ = PassageVectorStore::from_bytes(&bytes);
            }
        }
    }
}
