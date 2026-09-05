#!/usr/bin/env python3
"""Generate the JIS X 0208 ideograph bitset for src/extract/jis_x0208.rs.

Membership comes from Python's `shift_jis` codec (JIS X 0208:1997 +
JIS X 0201): a CJK Unified Ideograph (U+4E00..U+9FFF) that the codec
can encode is in the set. Deterministic — the same table on every run.
"""
LO, HI = 0x4E00, 0x9FFF
words = []
count = 0
for base in range(LO, HI + 1, 64):
    w = 0
    for i in range(64):
        cp = base + i
        if cp > HI: break
        try:
            chr(cp).encode('shift_jis'); w |= 1 << i; count += 1
        except UnicodeEncodeError:
            pass
    words.append(w)
print(f"// {count} ideographs, {len(words)} words")
lines = []
for i in range(0, len(words), 4):
    lines.append("    " + ", ".join(f"0x{w:016x}" for w in words[i:i+4]) + ",")
print(f"pub(super) const JIS_X0208_IDEOGRAPHS: [u64; {len(words)}] = [")
print("\n".join(lines)); print("];")
