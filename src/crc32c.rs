//! CRC-32C (Castagnoli): the checksum under every integrity footer —
//! the image's, the passage snapshot's, and each WAL record's. One
//! hand-rolled table keeps the dependency list as it is; Castagnoli
//! rather than the zlib polynomial because its error-detection bounds
//! at these block sizes are the reason iSCSI/ext4 settled on it.
//!
//! This file is deliberately included by BOTH crates (`mod crc32c;` in
//! lib.rs for the image format, and again in main.rs for the WAL and
//! passage formats) — the binaries' modules cannot see the library's
//! private items, and a checksum primitive is not worth widening the
//! library's public surface (which is `context` alone).

/// The Castagnoli polynomial, reflected form.
const POLYNOMIAL: u32 = 0x82F6_3B78;

const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut index = 0;
    while index < 256 {
        let mut crc = index as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLYNOMIAL
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
};

/// The CRC-32C of `bytes`, standard parameters (init `!0`, final xor
/// `!0`, reflected in and out) — the value `crc32c(1)` in RFC 3720's
/// vocabulary, comparable with any other implementation's output.
pub(crate) fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc = (crc >> 8) ^ TABLE[((crc ^ byte as u32) & 0xFF) as usize];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RFC 3720 / kernel test vectors — pinning the polynomial and
    /// the reflection parameters, so "matches other CRC-32C tools"
    /// stays true.
    #[test]
    fn matches_the_published_castagnoli_vectors() {
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
    }

    #[test]
    fn a_single_flipped_bit_changes_the_sum() {
        let clean = crc32c(b"the same bytes");
        let mut corrupt = b"the same bytes".to_vec();
        corrupt[3] ^= 1;
        assert_ne!(clean, crc32c(&corrupt));
    }
}
