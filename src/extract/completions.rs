//! `Completions`: the seam every extraction completion goes through
//! (ADR 0031 §3.1). Holds the run's identity and attempt counter —
//! moved off `ChatClient`, which stays a thin `/chat/completions`
//! transport that `taguru communities`/`consolidation` also use
//! directly and unaffected by any of this. A later issue (#818) adds
//! a `replay: Option<&ReplayIndex>` field and has `complete` try it
//! before falling through to the client; today `Completions` always
//! calls the client, byte-for-byte what `ChatClient` did on its own.

use super::*;

pub(super) struct Completions {
    client: ChatClient,
    /// ADR 0023: the identity of the run this value serves — 16 hex
    /// characters from the OS random source, minted once per
    /// `Completions` (one per `taguru extract` invocation), never
    /// derived from any input.
    run_id: String,
    /// ADR 0023: how many extraction completions this run has
    /// numbered so far — [`Completions::next_attempt`]'s counter.
    /// Shared across `--parallel` workers through the `&Completions`
    /// they already share.
    attempts: std::sync::atomic::AtomicU64,
}

impl Completions {
    pub(super) fn new(client: ChatClient) -> Self {
        Self {
            client,
            run_id: mint_run_id(),
            attempts: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub(super) fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The next extraction completion's identity (ADR 0023 §3.2):
    /// taken by the three extraction call sites immediately before
    /// their [`Completions::complete`], so the number names that call
    /// whether it succeeds or fails. Deliberately not taken inside
    /// `complete` itself: the ADR 0021 probe and `taguru communities`
    /// call `ChatClient::complete` directly and are not extraction
    /// attempts.
    pub(super) fn next_attempt(&self) -> AttemptRef {
        AttemptRef {
            run_id: self.run_id.clone(),
            attempt_seq: self
                .attempts
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1,
        }
    }

    pub(super) fn complete(
        &self,
        messages: &[serde_json::Value],
        options: &RequestOptions,
    ) -> Result<ChatCompletion, ChatError> {
        self.client.complete(messages, options)
    }
}
