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
