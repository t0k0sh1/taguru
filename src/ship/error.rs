//! Why a shipping cycle stopped early: retryable I/O vs. terminal
//! fencing.

use super::*;

/// Why a cycle stopped early. Transient errors surface as `Io` and the
/// next cycle retries from the recorded cursors; `Fenced` is terminal
/// by design.
#[derive(Debug)]
pub(crate) enum ShipError {
    /// A newer generation claimed the bucket: fail-stop.
    Fenced {
        newer_generation: u64,
    },
    Io(io::Error),
}

impl std::fmt::Display for ShipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fenced { newer_generation } => write!(
                f,
                "fenced: generation {newer_generation} claimed the bucket after ours"
            ),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl From<io::Error> for ShipError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The replica tailer speaks `io::Error`; `Fenced` cannot reach it (a
/// replica never ships, so nothing ever answers it with a fence), but
/// the conversion stays total rather than panicking on the impossible.
impl From<ShipError> for io::Error {
    fn from(error: ShipError) -> Self {
        match error {
            ShipError::Io(error) => error,
            ShipError::Fenced { .. } => io::Error::other(error.to_string()),
        }
    }
}

pub(super) fn store_error(context: &str, error: object_store::Error) -> ShipError {
    ShipError::Io(io::Error::other(format!("{context}: {error}")))
}
