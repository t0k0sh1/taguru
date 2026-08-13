//! Why a shipping cycle stopped early: retryable I/O vs. a
//! likely-permanent store error vs. terminal fencing.

use super::*;

/// Why a cycle stopped early. `Io` is transient by default assumption
/// — the next cycle retries from the recorded cursors; `Permanent` is
/// still retried (credentials can be rotated back, a bucket policy
/// can be relaxed) but is loud where `Io` is quiet, because it will
/// not self-heal on its own the way a network blip does; `Fenced` is
/// terminal by design.
#[derive(Debug)]
pub(crate) enum ShipError {
    /// A newer generation claimed the bucket: fail-stop.
    Fenced {
        newer_generation: u64,
    },
    Io(io::Error),
    /// object_store reported a condition a bare retry cannot fix by
    /// itself: bad/expired credentials, an operation this backend
    /// refuses to support, or a config key it does not recognize. See
    /// [`store_error`]'s classification.
    Permanent(io::Error),
}

impl std::fmt::Display for ShipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fenced { newer_generation } => write!(
                f,
                "fenced: generation {newer_generation} claimed the bucket after ours"
            ),
            Self::Io(error) | Self::Permanent(error) => error.fmt(f),
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
            ShipError::Io(error) | ShipError::Permanent(error) => error,
            ShipError::Fenced { .. } => io::Error::other(error.to_string()),
        }
    }
}

/// Classifies an `object_store` failure. Most variants (`Generic`,
/// `NotFound`-shaped-but-not-`NotFound`, `InvalidPath`, `Precondition`,
/// `NotModified`, transport errors folded into `Generic`) are treated
/// as transient by default — that has always been the safe assumption
/// here. The four variants below are different in kind: they name a
/// condition (credentials, an unsupported operation, an unrecognized
/// config key) that describes the DEPLOYMENT, not a momentary hiccup,
/// so a caller that only warns-and-retries would loop forever without
/// ever telling anyone why.
pub(super) fn store_error(context: &str, error: object_store::Error) -> ShipError {
    let permanent = matches!(
        error,
        object_store::Error::PermissionDenied { .. }
            | object_store::Error::Unauthenticated { .. }
            | object_store::Error::NotSupported { .. }
            | object_store::Error::UnknownConfigurationKey { .. }
    );
    let wrapped = io::Error::other(format!("{context}: {error}"));
    if permanent {
        ShipError::Permanent(wrapped)
    } else {
        ShipError::Io(wrapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed_source() -> Box<dyn std::error::Error + Send + Sync + 'static> {
        Box::new(io::Error::other("injected"))
    }

    /// The four deployment-shaped variants — the ones a bare retry
    /// cannot fix — classify as `Permanent`, not `Io`.
    #[test]
    fn deployment_shaped_errors_classify_as_permanent() {
        let cases: Vec<(&str, object_store::Error)> = vec![
            (
                "PermissionDenied",
                object_store::Error::PermissionDenied {
                    path: "p".to_string(),
                    source: boxed_source(),
                },
            ),
            (
                "Unauthenticated",
                object_store::Error::Unauthenticated {
                    path: "p".to_string(),
                    source: boxed_source(),
                },
            ),
            (
                "NotSupported",
                object_store::Error::NotSupported {
                    source: boxed_source(),
                },
            ),
            (
                "UnknownConfigurationKey",
                object_store::Error::UnknownConfigurationKey {
                    store: "s3",
                    key: "bogus".to_string(),
                },
            ),
        ];
        for (name, error) in cases {
            assert!(
                matches!(store_error("op", error), ShipError::Permanent(_)),
                "{name} must classify as Permanent"
            );
        }
    }

    /// `Display` must actually render each variant's message, not
    /// just carry the right data — `Fenced` names the generation,
    /// `Io`/`Permanent` both delegate to the wrapped `io::Error`'s own
    /// text.
    #[test]
    fn display_renders_each_variant() {
        let fenced = ShipError::Fenced {
            newer_generation: 7,
        };
        assert!(fenced.to_string().contains("generation 7"), "{fenced}");

        let io = ShipError::Io(io::Error::other("transient boom"));
        assert!(io.to_string().contains("transient boom"), "{io}");

        let permanent = ShipError::Permanent(io::Error::other("permanent boom"));
        assert!(
            permanent.to_string().contains("permanent boom"),
            "{permanent}"
        );
    }

    /// Everything else stays `Io` — the safe, transient-by-default
    /// assumption this classifier has always made, unchanged by this
    /// fix.
    #[test]
    fn other_errors_stay_transient() {
        let cases: Vec<(&str, object_store::Error)> = vec![
            (
                "Generic",
                object_store::Error::Generic {
                    store: "s3",
                    source: boxed_source(),
                },
            ),
            (
                "NotFound",
                object_store::Error::NotFound {
                    path: "p".to_string(),
                    source: boxed_source(),
                },
            ),
            (
                "Precondition",
                object_store::Error::Precondition {
                    path: "p".to_string(),
                    source: boxed_source(),
                },
            ),
            (
                "NotModified",
                object_store::Error::NotModified {
                    path: "p".to_string(),
                    source: boxed_source(),
                },
            ),
        ];
        for (name, error) in cases {
            assert!(
                matches!(store_error("op", error), ShipError::Io(_)),
                "{name} must stay Io"
            );
        }
    }
}
