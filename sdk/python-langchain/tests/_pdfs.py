"""Minimal PDF byte-builders for :mod:`taguru_langchain.ingest_connectors.
pdf` tests (issue #348) — hand-assembled rather than shipped as binary
fixtures, the same "synthesize at test time" convention every other
connector test in this repo already follows (no binary fixture exists
anywhere under ``sdk/python-langchain/tests``).

Each page's full text is emitted as ONE literal-string ``Tj`` operand, with
its own ``\\n``/``\\n\\n`` embedded verbatim in the PDF string (legal per
the PDF spec: a CR/LF inside a literal string is content, not a line-break
instruction to the writer) — this sidesteps pypdf's own text-extraction
layout heuristics entirely, which is deliberate: those heuristics decide
line breaks from Td/Tm/T* *positioning*, not from any blank-line-preserving
rule, and cannot be coaxed into reproducing a paragraph gap through
positioning alone. Embedding the paragraph breaks directly makes what
``PdfReader.extract_text()`` returns exactly what each test asserts on.
"""

from __future__ import annotations

import io
from collections.abc import Sequence


class _Builder:
    """The smallest correct PDF container: sequential objects, each
    recorded at its own byte offset, followed by a plain (uncompressed)
    xref table and a one-entry trailer — enough for pypdf to open, and
    nothing pypdf-specific, so these fixtures don't depend on pypdf's own
    writer producing byte-identical output across versions."""

    def __init__(self) -> None:
        self._objects: dict[int, bytes] = {}
        self._next = 1

    def alloc(self) -> int:
        number = self._next
        self._next += 1
        return number

    def add(self, number: int, body: bytes) -> None:
        self._objects[number] = body

    def render(self, root: int) -> bytes:
        buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: dict[int, int] = {}
        for number in sorted(self._objects):
            offsets[number] = len(buf)
            buf += f"{number} 0 obj\n".encode()
            buf += self._objects[number]
            buf += b"\nendobj\n"
        xref_offset = len(buf)
        highest = max(self._objects) if self._objects else 0
        buf += f"xref\n0 {highest + 1}\n".encode()
        buf += b"0000000000 65535 f \n"
        for number in range(1, highest + 1):
            offset = offsets.get(number)
            line = f"{offset:010d} 00000 n \n" if offset is not None else "0000000000 00000 f \n"
            buf += line.encode()
        buf += b"trailer\n"
        buf += f"<< /Size {highest + 1} /Root {root} 0 R >>\n".encode()
        buf += f"startxref\n{xref_offset}\n%%EOF".encode()
        return bytes(buf)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(pages: Sequence[str], *, outline: Sequence[tuple[str, int]] = ()) -> bytes:
    """A PDF whose page ``i`` extracts to exactly ``pages[i]`` (embedded
    ``\\n``/``\\n\\n`` preserved verbatim — see the module docstring).
    ``outline`` is a flat list of ``(title, page_index)`` bookmarks, each
    pointing at ``pages[page_index]`` via an explicit ``/Dest``."""
    builder = _Builder()
    catalog_num = builder.alloc()
    pages_num = builder.alloc()
    font_num = builder.alloc()
    builder.add(font_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_nums: list[int] = []
    for page_text in pages:
        page_num = builder.alloc()
        content_num = builder.alloc()
        page_nums.append(page_num)
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape(page_text)}) Tj ET".encode("latin-1")
        builder.add(
            content_num, f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        builder.add(
            page_num,
            (
                f"<< /Type /Page /Parent {pages_num} 0 R "
                f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
                f"/MediaBox [0 0 612 792] /Contents {content_num} 0 R >>"
            ).encode(),
        )

    kids = " ".join(f"{number} 0 R" for number in page_nums)
    builder.add(pages_num, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_nums)} >>".encode())

    outlines_ref = ""
    if outline:
        outlines_num = builder.alloc()
        item_nums = [builder.alloc() for _ in outline]
        for index, ((title, page_index), item_num) in enumerate(
            zip(outline, item_nums, strict=True)
        ):
            parts = [
                f"/Title ({_escape(title)})",
                f"/Parent {outlines_num} 0 R",
                f"/Dest [{page_nums[page_index]} 0 R /Fit]",
            ]
            if index > 0:
                parts.append(f"/Prev {item_nums[index - 1]} 0 R")
            if index + 1 < len(item_nums):
                parts.append(f"/Next {item_nums[index + 1]} 0 R")
            builder.add(item_num, ("<< " + " ".join(parts) + " >>").encode())
        builder.add(
            outlines_num,
            (
                f"<< /Type /Outlines /First {item_nums[0]} 0 R "
                f"/Last {item_nums[-1]} 0 R /Count {len(item_nums)} >>"
            ).encode(),
        )
        outlines_ref = f" /Outlines {outlines_num} 0 R"

    builder.add(catalog_num, f"<< /Type /Catalog /Pages {pages_num} 0 R{outlines_ref} >>".encode())
    return builder.render(catalog_num)


def scanned_pdf(page_count: int) -> bytes:
    """A structurally valid PDF whose every page has an empty content
    stream — the "photographed page, no text layer" shape ``ocr_required``
    (ADR 0007 §10) exists to name."""
    return text_pdf([""] * page_count)


def encrypted_pdf(
    pages: Sequence[str] = ("Secret text.",),
    *,
    user_password: str = "secret",
    owner_password: str | None = None,
) -> bytes:
    """``text_pdf(pages)``, re-encrypted with pypdf's own writer. A blank
    ``user_password`` (the default an "owner restrictions only" PDF uses)
    still requires importing pypdf here, so this helper — unlike
    :func:`text_pdf`/:func:`scanned_pdf`/:func:`corrupt_pdf` — is not
    import-safe without it; every caller already runs behind
    ``pytest.importorskip("pypdf")``."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(text_pdf(pages))))
    writer.encrypt(user_password=user_password, owner_password=owner_password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def corrupt_pdf() -> bytes:
    """A truncated PDF: a real xref/trailer starts to parse and then runs
    out of bytes — pypdf raises ``PdfStreamError`` (a ``PyPdfError``) on
    this, not a bare ``ValueError``, so it also exercises the connector's
    narrower exception branch."""
    data = text_pdf(["This document will be cut off before its xref table."])
    return data[: len(data) // 2]
