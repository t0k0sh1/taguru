"""Small text/heading helpers shared by more than one connector — first
extracted out of ``html.py`` (issue #349) when ``docx.py`` (issue #350)
needed the same paragraph-sanitization and breadcrumb-fitting behavior for
its own heading hierarchy. Not a connector itself, and not part of the
protocol (`protocol.py`) — purely internal plumbing.
"""

from __future__ import annotations

import io
import re
import zipfile
import zlib
from typing import Final

# Collapses an interior blank-line run to a single `\n` — mirrors exactly
# what a connector's paragraph splitting must never let slip through: a
# stray blank line INSIDE what the connector considers one paragraph (e.g.
# two consecutive `<br><br>`, or a `<w:br/>` pair) would otherwise silently
# register as an extra paragraph boundary once
# `taguru_langchain._extract.split_paragraphs` re-derives paragraphs from
# the final `"\n\n".join(...)` text, offsetting every locator/section
# paragraph index after it.
_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n+")


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def sanitize_paragraph_text(text: str) -> str:
    return _BLANK_RUN_RE.sub("\n", text).strip()


def fit_breadcrumb(crumbs: list[str], *, separator: str, max_bytes: int) -> str | None:
    """Joins ``crumbs`` (outermost first) with ``separator``, dropping the
    outermost (least specific) ancestor first until the result fits within
    ``max_bytes`` — ``None`` if even the innermost crumb alone doesn't fit
    (or ``crumbs`` is empty)."""
    while crumbs:
        candidate = separator.join(crumbs)
        if byte_len(candidate) <= max_bytes:
            return candidate
        crumbs = crumbs[1:]
    return None


# Deflate decompression, streamed so a hostile ratio cannot exhaust memory.
# `zlib.decompressobj(-MAX_WBITS)` is raw deflate (no zlib/gzip header — the
# form zip members store), and `.decompress(data, max_length)` is the seam
# that keeps output bounded: it emits at most `max_length` bytes and stashes
# the still-compressed remainder in `.unconsumed_tail` for the next call.
_INFLATE_OUTPUT_CHUNK: Final = 1 << 18


def _member_data_offset(source: object, info: zipfile.ZipInfo) -> int | None:
    """Byte offset of ``info``'s compressed data, parsed from its LOCAL
    header (whose name/extra lengths can differ from the central
    directory's), or ``None`` if the local signature isn't there."""
    source.seek(info.header_offset)  # type: ignore[attr-defined]
    local = source.read(30)  # type: ignore[attr-defined]
    if local[:4] != b"PK\x03\x04":
        return None
    name_len = int.from_bytes(local[26:28], "little")
    extra_len = int.from_bytes(local[28:30], "little")
    return info.header_offset + 30 + name_len + extra_len


def _inflate_capped(
    source: object, info: zipfile.ZipInfo, cap: int
) -> tuple[bytearray | None, int]:
    """Decompress one member, stopping once output exceeds ``cap``.

    Returns ``(data, produced)``: ``data`` is the decompressed bytes when
    they fit within ``cap`` (else ``None``, signalling "over cap"), and
    ``produced`` is the byte count actually generated — enough for a
    size-only caller even when ``data`` is dropped. Reads the compressed
    input bounded by ``compress_size`` (the true on-disk length), the exact
    bytes python-docx/-pptx would later feed their own parser, so the cost
    measured here is the cost the real read would incur.
    """
    offset = _member_data_offset(source, info)
    if offset is None:
        return None, 0
    source.seek(offset)  # type: ignore[attr-defined]
    if info.compress_type == zipfile.ZIP_STORED:
        if info.compress_size > cap:
            return None, info.compress_size
        return bytearray(source.read(info.compress_size)), info.compress_size  # type: ignore[attr-defined]
    if info.compress_type != zipfile.ZIP_DEFLATED:
        # An encoding this connector will not parse anyway — trust the
        # declared size rather than guess at its expansion.
        return (None, info.file_size) if info.file_size > cap else (bytearray(), info.file_size)
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    out = bytearray()
    produced = 0
    data = source.read(info.compress_size)  # type: ignore[attr-defined]
    while data:
        chunk = decompressor.decompress(data, _INFLATE_OUTPUT_CHUNK)
        produced += len(chunk)
        if produced > cap:
            return None, produced
        out += chunk
        data = decompressor.unconsumed_tail
        if not chunk and not data:
            break
    return out, produced


def decompressed_size_within(raw: bytes, cap: int) -> int | None:
    """Total decompressed size across every zip member, computed by ACTUALLY
    decompressing each entry rather than trusting ``ZipInfo.file_size`` —
    central-directory metadata an attacker can forge to a small value while
    the deflate stream still expands to gigabytes. Stops the moment the
    running total exceeds ``cap`` and returns that partial total, so a
    decompression bomb is detected without ever being materialized.
    ``None`` when ``raw`` is not a readable zip — the ``corrupt`` path's
    business, not this cap's.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            source = archive.fp
            if source is None:
                return None
            total = 0
            for info in archive.infolist():
                _data, produced = _inflate_capped(source, info, cap - total)
                total += produced
                if total > cap:
                    return total
            return total
    except (zipfile.BadZipFile, OSError, EOFError, zlib.error):
        return None


def read_member_within(raw: bytes, name: str, cap: int) -> bytes | None:
    """Decompressed bytes of the single named member, or ``None`` if it is
    absent, ``raw`` is not a zip, or it would decompress past ``cap`` —
    bounded-memory, on the same no-trust-``file_size`` footing as
    :func:`decompressed_size_within`.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            source = archive.fp
            if source is None:
                return None
            info = archive.getinfo(name)
            data, _produced = _inflate_capped(source, info, cap)
            return None if data is None else bytes(data)
    except (KeyError, zipfile.BadZipFile, OSError, EOFError, zlib.error):
        return None
