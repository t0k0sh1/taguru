"""Minimal OOXML (``.docx``) byte-builders for
:mod:`taguru_langchain.ingest_connectors.docx` tests (issue #350) —
hand-assembled rather than shipped as binary fixtures, the same
"synthesize at test time" convention ``tests/_pdfs.py`` already established
for :class:`~taguru_langchain.ingest_connectors.pdf.PdfConnector`.

Assembled as a raw zip of exactly the parts ``DocxConnector`` needs
(``[Content_Types].xml``, ``_rels/.rels``, ``word/document.xml``,
``word/styles.xml``, ``word/_rels/document.xml.rels``,
``docProps/core.xml``) rather than through ``python-docx``'s own writer —
this is what lets a fixture place XML ``python-docx`` has no create-API for
at all (a ``w:footnoteReference``, a ``w:txbxContent``) exactly where a test
needs it, and keeps every fixture's exact byte shape under this module's own
control rather than whatever a future ``python-docx`` release happens to
write. ``test_docx_connector.py`` separately builds one document through
``python-docx``'s real ``Document()``/``.save()`` to prove this connector
reads real Word output, not only this module's own dialect.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from collections.abc import Sequence

_CONTENT_TYPES = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
 ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType=
 "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType=
 "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType=
 "application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

_PACKAGE_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type=
 "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
 Target="word/document.xml"/>
<Relationship Id="rId2" Type=
 "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"
 Target="docProps/core.xml"/>
</Relationships>"""

_DOCUMENT_RELS = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type=
 "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
 Target="styles.xml"/>
</Relationships>"""

# Heading1..Heading6, each named exactly what `BabelFish.internal2ui` (real
# python-docx) would produce for a genuine Word document ("Heading N") — a
# heading built with `heading()` below is indistinguishable, from
# `DocxConnector`'s point of view, from one a real Word install produced.
# `MonTitre` is deliberately NOT a recognized heading name at all — it
# exists only so `outline_heading()` can prove the `w:outlineLvl` fallback
# fires for a document using a custom/localized style.
_STYLES = (
    b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
"""
    + b"".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{n}">'
        f'<w:name w:val="Heading {n}"/><w:basedOn w:val="Normal"/></w:style>'.encode()
        for n in range(1, 7)
    )
    + b"""<w:style w:type="paragraph" w:styleId="MonTitre">
<w:name w:val="MonTitre"/><w:basedOn w:val="Normal"/></w:style>
</w:styles>"""
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{_escape(text)}</w:t></w:r>'


def para(text: str, *, style: str | None = None) -> str:
    """One plain (or styled) paragraph."""
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{_run(text)}</w:p>"


def heading(text: str, level: int) -> str:
    """A paragraph styled ``HeadingN`` — the primary, style-name-based
    signal ``DocxConnector`` looks for first."""
    return para(text, style=f"Heading{level}")


def outline_heading(text: str, level: int) -> str:
    """A paragraph carrying only ``w:outlineLvl`` (0-based; `level` is
    1-based) and a style name (`MonTitre`) that isn't recognized as a
    heading at all — exercises ``DocxConnector``'s fallback for a document
    whose heading styles aren't named the way ``BabelFish`` expects."""
    return (
        f'<w:p><w:pPr><w:pStyle w:val="MonTitre"/>'
        f'<w:outlineLvl w:val="{level - 1}"/></w:pPr>{_run(text)}</w:p>'
    )


def table(rows: Sequence[Sequence[str]]) -> str:
    """A plain table (no nested content) — ``rows`` is document order,
    outer-to-inner cells left-to-right."""
    row_xml = "".join(
        "<w:tr>"
        + "".join(f"<w:tc><w:tcPr/><w:p>{_run(cell)}</w:p></w:tc>" for cell in row)
        + "</w:tr>"
        for row in rows
    )
    return f"<w:tbl><w:tblPr/><w:tblGrid/>{row_xml}</w:tbl>"


def table_with_nested_cell(
    rows: Sequence[Sequence[str]], *, at: tuple[int, int], nested_rows: Sequence[Sequence[str]]
) -> str:
    """``table(rows)``, except cell ``at`` (0-based ``(row, col)``) holds a
    nested ``table(nested_rows)`` instead of its own plain text — a ``w:tc``
    must still end in a paragraph after a nested ``w:tbl`` (OOXML's own
    content model), so an empty one follows it, exactly as real Word emits
    for this shape."""
    return table_with_nested_cells(rows, nested={at: nested_rows})


def table_with_nested_cells(
    rows: Sequence[Sequence[str]],
    *,
    nested: dict[tuple[int, int], Sequence[Sequence[str]]],
) -> str:
    """``table(rows)``, except each cell keyed in ``nested`` (0-based
    ``(row, col)``) holds its own nested ``table(...)`` instead of plain
    text — one parent table with a nested table in more than one of its own
    cells, the shape a locator-numbering bug can only surface against
    (issue #350 review: a nested table's ordinal must count across every
    cell of its parent, not reset per cell)."""
    row_xml = []
    for row_index, row in enumerate(rows):
        cells = []
        for col_index, cell in enumerate(row):
            nested_rows = nested.get((row_index, col_index))
            if nested_rows is not None:
                cells.append(f"<w:tc><w:tcPr/>{table(nested_rows)}<w:p/></w:tc>")
            else:
                cells.append(f"<w:tc><w:tcPr/><w:p>{_run(cell)}</w:p></w:tc>")
        row_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return f"<w:tbl><w:tblPr/><w:tblGrid/>{''.join(row_xml)}</w:tbl>"


def footnote_reference_para(text: str, *, note_id: int = 2) -> str:
    """A paragraph referencing footnote ``note_id`` — the reference alone,
    without ever declaring a ``word/footnotes.xml`` part: enough for
    ``DocxConnector``'s own marker scan (it never reads the footnote text
    itself, only detects that a reference exists in the body)."""
    return f'<w:p>{_run(text)}<w:r><w:footnoteReference w:id="{note_id}"/></w:r></w:p>'


def endnote_reference_para(text: str, *, note_id: int = 2) -> str:
    return f'<w:p>{_run(text)}<w:r><w:endnoteReference w:id="{note_id}"/></w:r></w:p>'


def comment_reference_para(text: str, *, comment_id: int = 0) -> str:
    return f'<w:p>{_run(text)}<w:r><w:commentReference w:id="{comment_id}"/></w:r></w:p>'


def textbox_para(box_text: str) -> str:
    """A paragraph whose only content is a text box — real Word wraps this
    in ``w:drawing``/DrawingML shape markup; only the ``w:txbxContent``
    wrapper itself matters to ``DocxConnector``'s marker scan, so the
    surrounding shape XML is trimmed to the minimum lxml will still parse as
    well-formed."""
    return (
        "<w:p><w:r><w:drawing><wps:txbx xmlns:wps="
        '"http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        f"<w:txbxContent>{para(box_text)}</w:txbxContent>"
        "</wps:txbx></w:drawing></w:r></w:p>"
    )


def docx_bytes(body_xml: str, *, title: str | None = None) -> bytes:
    """Assembles a complete, valid ``.docx`` whose ``word/document.xml``
    body is exactly ``body_xml`` (already-built ``para()``/``heading()``/
    ``table()`` fragments, concatenated) — real headings/tables/paragraphs,
    exactly as ``DocxConnector.read`` will see them."""
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_xml}</w:body></w:document>"
    ).encode()
    title_xml = f"<dc:title>{_escape(title)}</dc:title>" if title else ""
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
        'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"{title_xml}</cp:coreProperties>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _PACKAGE_RELS)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", _STYLES)
        archive.writestr("word/_rels/document.xml.rels", _DOCUMENT_RELS)
        archive.writestr("docProps/core.xml", core.encode())
    return buf.getvalue()


def zip_bomb_docx(declared_size: int = 2 * 1024 * 1024) -> bytes:
    """A small, high-ratio zip whose one entry declares ``declared_size``
    zero bytes but compresses to almost nothing — the shape
    `DocxConnector`'s `max_decompressed_bytes` cap exists to refuse before
    python-docx ever decompresses it. Never assembled into a fully valid
    docx package (no ``[Content_Types].xml``, etc.): the cap is checked,
    and this document refused, before python-docx would ever open it."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * declared_size)
    return buf.getvalue()


def forged_size_zip_bomb_docx(payload_size: int = 32 * 1024 * 1024) -> bytes:
    """A zip bomb that DEFEATS a metadata-only check: its one entry really
    decompresses to ``payload_size`` bytes, but every ``file_size`` field
    (local header and central directory) is forged down to 50 so
    ``ZipInfo.file_size`` reports 50. Only decompressing for real reveals
    the true size — the exact bypass `decompressed_size_within` closes.
    The CRC is repaired for the 50-byte prefix so a truncating reader
    accepts it too."""
    payload = b"\0" * payload_size
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", payload)
    raw = bytearray(buf.getvalue())
    central = raw.find(b"PK\x01\x02")
    local = raw.find(b"PK\x03\x04")
    forged = 50
    struct.pack_into("<I", raw, central + 24, forged)  # central: uncompressed size
    struct.pack_into("<I", raw, local + 22, forged)  # local: uncompressed size
    crc = zlib.crc32(payload[:forged])
    struct.pack_into("<I", raw, central + 16, crc)
    struct.pack_into("<I", raw, local + 14, crc)
    return bytes(raw)


def corrupt_docx(kind: str = "not_zip") -> bytes:
    """Three distinct corruption shapes ``DocxConnector`` must all report as
    `corrupt`, never raise: ``"not_zip"`` (not a zip container at all),
    ``"missing_part"`` (a valid zip missing ``word/document.xml``
    entirely), and ``"malformed_xml"`` (the part exists but its XML doesn't
    parse)."""
    # `docProps/core.xml` is included in every variant below (empty core
    # properties, no `<dc:title>`) even though it isn't what's being tested
    # — `_PACKAGE_RELS` declares a relationship to it, and python-docx's
    # package loader walks every declared relationship eagerly, so a
    # missing `docProps/core.xml` would itself raise a `KeyError` before
    # `word/document.xml` (the part actually under test) is ever reached.
    core = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
        b'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"/>'
    )
    if kind == "not_zip":
        return b"this is not a zip file at all, just plain bytes"
    if kind == "missing_part":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
            archive.writestr("_rels/.rels", _PACKAGE_RELS)
            archive.writestr("docProps/core.xml", core)
        return buf.getvalue()
    if kind == "malformed_xml":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
            archive.writestr("_rels/.rels", _PACKAGE_RELS)
            archive.writestr("word/document.xml", b"<w:document><w:body><w:p not closed")
            archive.writestr("word/styles.xml", _STYLES)
            archive.writestr("word/_rels/document.xml.rels", _DOCUMENT_RELS)
            archive.writestr("docProps/core.xml", core)
        return buf.getvalue()
    raise ValueError(f"unknown corrupt_docx kind: {kind!r}")


def encrypted_docx() -> bytes:
    """The first 8 bytes any MS-OFFCRYPTO password-protected Office file
    starts with (the OLE2/Compound File Binary container signature) —
    enough for ``DocxConnector``'s own magic-byte check, which never parses
    past this header."""
    return b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504
