//! SHA-256: the binary crate's one content-hash primitive for
//! identity, not the FNV-1a fold `hash.rs` owns for change detection.
//! `extract`'s manifest and checkpoints, `benchmark`'s
//! documents/configs/hostnames, and `evaluate`'s thresholds-file
//! stamp all hash through this one function rather than a
//! second implementation that could drift. `evaluate` in particular
//! must not import `crate::extract` (ADR 0004 §12, enforced by
//! `evaluate/tests.rs`'s `evaluate_module_never_names_an_extraction_or_embedding_seam`),
//! which is why this lives here rather than on `extract` itself.

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against the published SHA-256 test vector for `"abc"`,
    /// independent of `sha256_hex` itself — every other test in the
    /// tree computes both sides through this same function, which
    /// would not catch a regression in the digest algorithm or the
    /// hex formatting.
    #[test]
    fn sha256_hex_matches_the_published_test_vector_for_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
