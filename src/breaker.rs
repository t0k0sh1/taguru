//! A minimal circuit breaker shared by every optional network provider
//! (embeddings, and — ADR 0006 §12 — the evidence reranker): open after
//! [`DEFAULT_THRESHOLD`] consecutive attempt failures, then one probe
//! per [`DEFAULT_COOLDOWN`] until a success closes it. Exists so a dead
//! provider costs each request a refusal instead of a full timeout —
//! the lanes behind it already know how to degrade; this makes the
//! degradation cheap.
//!
//! `subject` names the provider family in every client-facing refusal
//! string ("the embedding provider circuit is open…", "the reranker
//! provider circuit is open…") and in nothing else — it never carries
//! a URL, model name, or credential.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Consecutive provider-attempt failures that open the breaker. Counted
/// per ATTEMPT, not per call, so one full retry ladder against a dead
/// provider is enough to open it — the next caller fails fast instead
/// of paying the timeout again.
pub(crate) const DEFAULT_THRESHOLD: u32 = 3;
/// How long an open breaker refuses before letting one probe through.
pub(crate) const DEFAULT_COOLDOWN: Duration = Duration::from_secs(30);
/// A floor under how long a claimed half-open probe sits unsettled
/// before [`ProviderBreaker::admit`] reclaims it (its caller panicked,
/// or was otherwise dropped without `record`/`abandon`) — independent
/// of `cooldown` itself, which tests routinely set to [`Duration::ZERO`]
/// purely to make `Open`'s OWN recovery instant. Without this floor,
/// that same zero would also make a probe reclaimable the instant it
/// is claimed, breaking "one probe in flight at a time" for any
/// zero-cooldown breaker.
const MIN_PROBE_RECLAIM: Duration = Duration::from_millis(50);

/// Shared by handle: a provider holds one and the registry reads the
/// same one for /metrics and its own fast-fail gate.
#[derive(Clone)]
pub(crate) struct ProviderBreaker(Arc<BreakerInner>);

struct BreakerInner {
    subject: &'static str,
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
    /// by the probe's outcome (or [`ProviderBreaker::abandon`]) —
    /// or, if neither runs (a panic between [`ProviderBreaker::admit`]
    /// and its outcome), reclaimed once `since.elapsed()` exceeds
    /// `cooldown` (`admit`), the same recovery window `Open` already
    /// gives itself. Unlike `Open`, a stuck `HalfOpen` has no other
    /// path back to admitting calls, so this claim needs its own clock
    /// too, not the one it copied off `Open` when it was born.
    HalfOpen {
        probing: bool,
        since: Instant,
    },
}

/// What [`ProviderBreaker::admit`] granted: an ordinary call, or the
/// one half-open probe whose outcome decides the breaker's next state.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Admission {
    Normal,
    Probe,
}

/// One scrape's view of the breaker, rendered under
/// `taguru_{subject}_breaker_*` on /metrics.
pub(crate) struct BreakerSnapshot {
    /// 0 = closed, 1 = open, 2 = half-open.
    pub(crate) state: u8,
    pub(crate) consecutive_failures: u64,
    pub(crate) opened_total: u64,
    pub(crate) short_circuits_total: u64,
}

impl ProviderBreaker {
    /// The production policy is [`DEFAULT_THRESHOLD`]/[`DEFAULT_COOLDOWN`].
    pub(crate) fn new(subject: &'static str) -> Self {
        Self::with_policy(subject, DEFAULT_THRESHOLD, DEFAULT_COOLDOWN)
    }

    /// Tests shrink the cooldown to keep the half-open walk fast.
    pub(crate) fn with_policy(subject: &'static str, threshold: u32, cooldown: Duration) -> Self {
        Self(Arc::new(BreakerInner {
            subject,
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
    pub(crate) fn admit(&self) -> Result<Admission, String> {
        let mut state = self.0.state.lock();
        match &mut *state {
            BreakerState::Closed { .. } => Ok(Admission::Normal),
            BreakerState::Open { since } => {
                let elapsed = since.elapsed();
                if elapsed >= self.0.cooldown {
                    *state = BreakerState::HalfOpen {
                        probing: true,
                        since: Instant::now(),
                    };
                    Ok(Admission::Probe)
                } else {
                    self.0.short_circuits_total.fetch_add(1, Ordering::Relaxed);
                    Err(self.open_refusal(self.0.cooldown - elapsed))
                }
            }
            BreakerState::HalfOpen { probing, since } => {
                if *probing && since.elapsed() < self.probe_reclaim_after() {
                    self.0.short_circuits_total.fetch_add(1, Ordering::Relaxed);
                    Err(self.probe_busy_refusal())
                } else {
                    // Either idle, or a claimed probe nobody ever
                    // settled (its caller panicked, or was otherwise
                    // dropped without `record`/`abandon`) and has sat
                    // past its own cooldown — reclaim the slot rather
                    // than wedge this breaker half-open for the rest
                    // of the process.
                    *probing = true;
                    *since = Instant::now();
                    Ok(Admission::Probe)
                }
            }
        }
    }

    /// The same refusal arms as [`Self::admit`], WITHOUT claiming the
    /// probe slot — a pre-flight gate, so a short-circuit never lands
    /// in a provider-latency histogram as a ~0s sample. `None` means
    /// "call the provider" (whose own `admit` then claims the probe if
    /// one is due).
    pub(crate) fn refusal(&self) -> Option<String> {
        let state = self.0.state.lock();
        let refusal = match &*state {
            BreakerState::Closed { .. } => None,
            BreakerState::Open { since } => {
                let elapsed = since.elapsed();
                (elapsed < self.0.cooldown).then(|| self.open_refusal(self.0.cooldown - elapsed))
            }
            BreakerState::HalfOpen { probing, since } => (*probing
                && since.elapsed() < self.probe_reclaim_after())
            .then(|| self.probe_busy_refusal()),
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
    pub(crate) fn record(&self, admission: Admission, ok: bool) {
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

    /// Releases an admitted attempt that produced NO outcome (e.g. a
    /// shutdown abandon): a claimed probe slot must not wedge the
    /// breaker half-open forever, and an abandoned wait is no evidence
    /// about the provider either way.
    pub(crate) fn abandon(&self, admission: Admission) {
        if let Admission::Probe = admission
            && let BreakerState::HalfOpen { probing, .. } = &mut *self.0.state.lock()
        {
            *probing = false;
        }
    }

    pub(crate) fn snapshot(&self) -> BreakerSnapshot {
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

    /// How long a claimed half-open probe may sit unsettled before
    /// [`Self::admit`]/[`Self::refusal`] treat it as abandoned and
    /// reclaim it — [`MIN_PROBE_RECLAIM`], or `cooldown` itself when
    /// that is the longer of the two (production's 30s cooldown
    /// dominates; a test's zero cooldown does not).
    fn probe_reclaim_after(&self) -> Duration {
        self.0.cooldown.max(MIN_PROBE_RECLAIM)
    }

    /// The open-state refusal. Client-facing (it rides 502 bodies and
    /// explain reasons), so it names the recovery path, never the URL.
    fn open_refusal(&self, retry_in: Duration) -> String {
        format!(
            "the {} circuit is open after {} consecutive failures; \
             the next probe is allowed in {}s",
            self.0.subject,
            self.0.threshold,
            retry_in.as_secs().max(1)
        )
    }

    fn probe_busy_refusal(&self) -> String {
        format!(
            "the {} circuit is half-open with a probe already in flight; \
             not adding load until it settles",
            self.0.subject
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state machine itself: a success resets the consecutive
    /// count, one probe flies at a time, a failed probe re-opens, an
    /// abandoned probe releases its slot without judging the provider.
    #[test]
    fn the_breaker_state_machine_counts_probes_and_resets() {
        let breaker = ProviderBreaker::with_policy("test provider", 2, Duration::from_secs(1000));
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

        let breaker = ProviderBreaker::with_policy("test provider", 1, Duration::ZERO);
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

    /// A claimed probe that nobody ever settles (the caller panicked,
    /// or was otherwise dropped between `admit` and `record`/`abandon`)
    /// must not wedge the breaker half-open for the rest of the
    /// process — `HalfOpen` has no other clock back to admitting calls
    /// the way `Open` does, so it needs its own cooldown-bounded
    /// self-heal.
    #[test]
    fn an_unsettled_probe_is_reclaimed_after_its_own_cooldown() {
        // Duration::ZERO only fast-forwards `Open`'s own cooldown
        // (matching every other test's convention); `probe_reclaim_after`
        // still floors the half-open reclaim window at
        // `MIN_PROBE_RECLAIM` regardless.
        let breaker = ProviderBreaker::with_policy("test provider", 1, Duration::ZERO);
        let only = breaker.admit().unwrap();
        breaker.record(only, false);
        // Claims the sole half-open probe slot, then is simply
        // dropped — simulating a panic mid-attempt: neither `record`
        // nor `abandon` ever runs.
        let stuck = breaker.admit().unwrap();
        assert!(matches!(stuck, Admission::Probe));
        assert!(
            breaker.admit().is_err(),
            "still busy immediately after the leak"
        );
        assert!(breaker.refusal().is_some(), "the pre-flight gate agrees");

        std::thread::sleep(MIN_PROBE_RECLAIM * 2);

        let reclaimed = breaker
            .admit()
            .expect("past its own reclaim window, the stuck slot is reclaimed");
        assert!(matches!(reclaimed, Admission::Probe));
        // Reclaiming makes THIS call the new probe — a third
        // concurrent caller still sees it as busy, exactly as if the
        // original probe had never gone missing.
        assert!(
            breaker.refusal().is_some(),
            "the reclaimed probe is itself in flight"
        );
        breaker.record(reclaimed, true);
        assert_eq!(breaker.snapshot().state, 0, "the new probe closes it");
        assert!(breaker.refusal().is_none(), "closed, nothing in flight");
    }

    /// The refusal strings name the subject a caller configured this
    /// breaker for — the point of parameterizing it — never a URL or
    /// credential.
    #[test]
    fn refusal_strings_name_the_configured_subject() {
        let breaker = ProviderBreaker::with_policy("reranker provider", 1, Duration::from_secs(30));
        let attempt = breaker.admit().unwrap();
        breaker.record(attempt, false);
        let refusal = breaker.refusal().expect("open");
        assert!(
            refusal.contains("reranker provider circuit is open"),
            "{refusal}"
        );

        let breaker = ProviderBreaker::with_policy("reranker provider", 1, Duration::ZERO);
        let attempt = breaker.admit().unwrap();
        breaker.record(attempt, false);
        // The next admit claims the sole probe slot; a second concurrent
        // caller sees the half-open busy refusal.
        let _probe = breaker.admit().unwrap();
        let busy = breaker.admit().unwrap_err();
        assert!(
            busy.contains("reranker provider circuit is half-open"),
            "{busy}"
        );
    }
}
