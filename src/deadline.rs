//! Request-scoped time budget, threaded through the `block_in_place`
//! sections a `tokio::time::timeout` race around the handler cannot
//! preempt once entered (see [`crate::limits::enforce_timeout`]). A
//! [`Deadline`] is stamped onto the request as an `Extension` when the
//! budget starts and carried by value from there — `Copy`, like the
//! `Instant`/`Duration` it wraps.

use std::time::{Duration, Instant};

/// A point in time a piece of work should give up by, or no point at
/// all. `unbounded()` is for callers with no request budget to inherit
/// — the CLI binaries, which run one command to completion.
#[derive(Clone, Copy, Debug)]
pub struct Deadline(Option<Instant>);

impl Deadline {
    /// A deadline `budget` from now. A `budget` too large to represent
    /// as a future `Instant` becomes unbounded rather than panicking —
    /// the caller asked for "practically forever", and that is what
    /// `unbounded()` means.
    pub fn after(budget: Duration) -> Self {
        Self(Instant::now().checked_add(budget))
    }

    /// No deadline: `expired()` is always false and `remaining()` is
    /// always `Duration::MAX`.
    pub const fn unbounded() -> Self {
        Self(None)
    }

    /// Time left, floored at zero. `Duration::MAX` for an unbounded
    /// deadline.
    pub fn remaining(self) -> Duration {
        match self.0 {
            Some(deadline) => deadline.saturating_duration_since(Instant::now()),
            None => Duration::MAX,
        }
    }

    /// True once `Instant::now()` has reached the deadline. Always
    /// false for an unbounded deadline.
    pub fn expired(self) -> bool {
        self.0.is_some_and(|deadline| deadline <= Instant::now())
    }
}

/// A [`Deadline`] elapsed before an operation that checks it could
/// finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeadlineExceeded;

impl std::fmt::Display for DeadlineExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request deadline exceeded")
    }
}

impl std::error::Error for DeadlineExceeded {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_never_expires_and_has_max_remaining() {
        let deadline = Deadline::unbounded();
        assert!(!deadline.expired());
        assert_eq!(deadline.remaining(), Duration::MAX);
    }

    #[test]
    fn a_deadline_in_the_future_has_not_expired() {
        let deadline = Deadline::after(Duration::from_secs(60));
        assert!(!deadline.expired());
        assert!(deadline.remaining() > Duration::from_secs(1));
    }

    #[test]
    fn a_zero_budget_is_already_expired() {
        let deadline = Deadline::after(Duration::ZERO);
        // The clock advances between `after` and `expired` even under
        // load, so a zero budget is expired the instant it is observed.
        std::thread::sleep(Duration::from_millis(1));
        assert!(deadline.expired());
        assert_eq!(deadline.remaining(), Duration::ZERO);
    }

    #[test]
    fn a_budget_too_large_to_represent_becomes_unbounded() {
        let deadline = Deadline::after(Duration::MAX);
        assert!(!deadline.expired());
        assert_eq!(deadline.remaining(), Duration::MAX);
    }
}
