//! `Completions`: the seam every extraction completion goes through
//! (ADR 0031 §3.1). Holds the run's identity and attempt counter —
//! moved off `ChatClient`, which stays a thin `/chat/completions`
//! transport that `taguru communities`/`consolidation` also use
//! directly and unaffected by any of this — plus, per document
//! (`Completions::begin_document`), the `ReplayIndex` a replay run
//! consults before ever reaching `client`.

use super::*;

pub(super) struct Completions {
    /// `None` exactly under `--replay strict` with no
    /// `TAGURU_EXTRACT_URL` (ADR 0031 §3.7/§3.8) — every other run
    /// always has one; `--dry-run` never builds a `Completions` at
    /// all (extract.rs).
    client: Option<ChatClient>,
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
    /// This document's index (ADR 0031 §3.1/§3.4), set by
    /// [`Completions::begin_document`] before its pieces are
    /// dispatched — `None` off `--replay` and, briefly, before the
    /// first document of a replaying run begins.
    replay: Option<ReplayIndex>,
    replay_mode: ReplayMode,
    /// This document's completion counts (reset by
    /// [`Completions::begin_document`]) — the source of the
    /// `replayed N/M completions (K live)` line and the
    /// `replay_summary` record.
    replayed: std::sync::atomic::AtomicU64,
    live: std::sync::atomic::AtomicU64,
}

impl Completions {
    pub(super) fn new(client: Option<ChatClient>) -> Self {
        Self {
            client,
            run_id: mint_run_id(),
            attempts: std::sync::atomic::AtomicU64::new(0),
            replay: None,
            replay_mode: ReplayMode::Off,
            replayed: std::sync::atomic::AtomicU64::new(0),
            live: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub(super) fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Resets this value for the document about to run: installs its
    /// `ReplayIndex` (or none, off `--replay`) and zeroes the
    /// replayed/live counters. Called from `Run::extract_document`
    /// (`&mut self` there — sequential across documents, so no
    /// `--parallel` worker of a PRIOR document can still be reading
    /// the index it replaces).
    pub(super) fn begin_document(&mut self, replay: Option<ReplayIndex>, mode: ReplayMode) {
        self.replay = replay;
        self.replay_mode = mode;
        self.replayed.store(0, std::sync::atomic::Ordering::Relaxed);
        self.live.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// ADR 0031 §3.6: this document's system-prompt pin decision —
    /// `NoRecord` off `--replay` (nothing recorded to pin from).
    pub(super) fn pinned_system(&self) -> SystemPinDecision<'_> {
        self.replay
            .as_ref()
            .map(ReplayIndex::pinned_system)
            .unwrap_or(SystemPinDecision::NoRecord)
    }

    /// This document's `(replayed, live)` counts so far.
    pub(super) fn document_counts(&self) -> (u64, u64) {
        (
            self.replayed.load(std::sync::atomic::Ordering::Relaxed),
            self.live.load(std::sync::atomic::Ordering::Relaxed),
        )
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

    /// Tries this document's `ReplayIndex` first (a hit needs no HTTP
    /// call at all); on a miss, `--replay auto` falls through to
    /// `client` exactly as an unreplayed run would, and `--replay
    /// strict` fails this completion instead — reported on stderr with
    /// the index's own miss diagnostic, and as a `ChatError` so the
    /// caller's ordinary failure handling (ADR 0001 §7) takes it from
    /// there. `piece_id` is `ReplayIndex::lookup`'s own key, never
    /// part of the match itself (ADR 0031 §3.2).
    pub(super) fn complete(
        &self,
        piece_id: &str,
        messages: &[serde_json::Value],
        options: &RequestOptions,
    ) -> Result<ChatCompletion, ChatError> {
        if let Some(replay) = &self.replay {
            match replay.lookup(piece_id, messages, options.max_tokens) {
                ReplayLookup::Hit(outcome) => {
                    self.replayed
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return outcome;
                }
                ReplayLookup::Miss(diagnostic) => {
                    if self.replay_mode == ReplayMode::Strict {
                        let message = format!("--replay strict: {}", describe_miss(&diagnostic));
                        eprintln!("taguru: extract: {message}");
                        return Err(ChatError::new(ChatFailure::Transport, message));
                    }
                    // --replay auto: fall through to a live call below,
                    // exactly as an unreplayed run would.
                }
            }
        }
        self.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match &self.client {
            Some(client) => client.complete(messages, options),
            None => Err(ChatError::new(
                ChatFailure::Transport,
                "no model endpoint is configured and replay did not satisfy this completion \
                 (--replay strict with no TAGURU_EXTRACT_URL)"
                    .to_string(),
            )),
        }
    }
}

/// `MissDiagnostic`'s stderr/error wording — "this piece has N
/// recorded attempts, none match" plus the first turn that differs,
/// when a comparison was possible (ADR 0031 §3.2's diagnosability
/// point). Names each side's turn by role and `sha256` only, never by
/// content: this message reaches stderr, a failed document's
/// `ChatError`, and from there the diagnostics sidecar's
/// `parse_error` — all metadata by design (ADR 0001 §10). The
/// document text stays exclusively in the attempts log's own
/// `messages`.
fn describe_miss(diagnostic: &MissDiagnostic) -> String {
    let piece_id = &diagnostic.piece_id;
    let mut message = if diagnostic.recorded == 0 {
        format!("piece {piece_id} has no recorded attempts")
    } else {
        format!(
            "piece {piece_id} has {} recorded attempt(s), none match",
            diagnostic.recorded
        )
    };
    if let Some(difference) = &diagnostic.first_difference {
        message.push_str(&format!(
            " — first differs at turn {}: recorded {}sha256:{}, requested {}sha256:{}",
            difference.turn_index,
            difference
                .recorded_role
                .as_deref()
                .map(|role| format!("{role} "))
                .unwrap_or_default(),
            difference.recorded_digest.as_deref().unwrap_or("<none>"),
            difference
                .requested_role
                .as_deref()
                .map(|role| format!("{role} "))
                .unwrap_or_default(),
            difference.requested_digest.as_deref().unwrap_or("<none>"),
        ));
    }
    message
}
