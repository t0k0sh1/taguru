//! FNV-1a: the crate's one content-hash primitive, shared by
//! gloss-change detection ([`fnv1a`]), the BM25 sidecar's drift folds,
//! and paragraph hashing. Stability across builds matters (std's
//! `DefaultHasher` promises none) — a changed function would silently
//! re-embed every name and re-upsert every source, so the constants
//! live in exactly one place.
//!
//! Like `crc32c`, this file is included by BOTH crates (`mod hash;` in
//! main.rs for the folds above, and in lib.rs for the community
//! fingerprints) — a hash primitive is not worth widening the
//! library's public surface.

/// The FNV-1a offset basis — the canonical starting digest for
/// [`fnv1a_fold`] chains.
pub(crate) const FNV1A_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// One FNV-1a continuation over `bytes` from an existing digest — the
/// shared primitive behind [`fnv1a`] and the BM25 drift folds, so the
/// constants live in exactly one place. Stability across builds
/// matters (std's DefaultHasher promises none): a changed function
/// would silently re-embed every name and re-upsert every source.
pub(crate) fn fnv1a_fold(digest: u64, bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut digest = digest;
    for byte in bytes {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    digest
}

/// FNV-1a: a tiny content hash for gloss-change detection.
// The library's inclusion only needs the fold; this whole-string form
// is the binary crate's.
#[allow(dead_code)]
pub(crate) fn fnv1a(text: &str) -> u64 {
    fnv1a_fold(FNV1A_OFFSET, text.bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_is_pinned_to_the_published_test_vectors() {
        // Gloss-change detection stores these hashes in the vector
        // sidecar; any drift would silently re-embed every name. Pin
        // the function to the official FNV-1a 64-bit vectors.
        assert_eq!(fnv1a(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a("foobar"), 0x8594_4171_f739_67e8);
    }
}
