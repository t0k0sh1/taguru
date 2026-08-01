"""The PDF connector (ADR 0007 §7/§8/§10, issue #348): ``.pdf`` files into
:class:`~taguru_langchain.ingest_connectors.document.ConnectorDocument`,
one ``{"kind": "page", "value": ...}`` locator per paragraph and, when the
document carries one, its outline (bookmarks) as ``sections``.

The thing ``docs/local-rag-walkthrough.html`` used to require of every
caller — "Converting a PDF to text is this pipeline's job — plain
``pypdf``... before any Taguru code runs at all" — done once, here,
against the normalized document contract instead of a bare string.

No OCR engine ships in this connector (ADR 0007 §10): a page whose
extracted text has fewer than ``min_chars_per_page`` non-whitespace
characters is treated as having no usable text layer and is named in an
``ocr_required`` diagnostic — never silently passed through as low-quality
text. Absent an external OCR adapter (#352), that diagnostic is terminal
for the pages it names.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from taguru import Locator

try:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError
except ImportError as _pypdf_import_error:  # pragma: no cover - exercised only without pypdf
    PdfReader = None  # type: ignore[assignment, misc]
    PyPdfError = Exception  # type: ignore[assignment, misc]
    _PYPDF_IMPORT_ERROR: ImportError | None = _pypdf_import_error
else:
    _PYPDF_IMPORT_ERROR = None

from .._extract import MAX_PASSAGE_BYTES, split_paragraphs
from .document import (
    MAX_SECTION_BYTES,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    DiagnosticCode,
    FingerprintInputs,
    LocatorEntry,
    SectionEntry,
    options_digest,
)
from .sources import check_source_id, file_source_id

PARSER_NAME: Final = "taguru-pdf-connector"
PARSER_VERSION: Final = "1.0.0"

_SUPPORTED_SUFFIXES: Final = frozenset({".pdf"})

# The raw-file cap, checked before any parsing — independent of
# MAX_PASSAGE_BYTES (checked again below, against the EXTRACTED text): a
# huge scanned PDF can stay well under the text cap, and a huge but
# well-compressed text PDF can still exceed it, so a PDF connector needs
# both, unlike TextFileConnector where the file bytes ARE the text.
_DEFAULT_MAX_FILE_BYTES: Final = 64 * 1024 * 1024

# ADR 0007 §10's own words: "a connector-defined, documented threshold —
# e.g. fewer than N extractable characters per page after whitespace
# normalization." 16 was picked as comfortably above what a stray running
# header/footer or page number alone would produce, comfortably below a
# single real sentence.
_DEFAULT_MIN_CHARS_PER_PAGE: Final = 16

# How many page numbers one diagnostic message lists — a document with
# hundreds of scanned pages must not make one diagnostic's message
# unbounded.
_MAX_LISTED_PAGES: Final = 20

_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _page_char_count(text: str) -> int:
    """Non-whitespace character count — the OCR-required threshold (ADR
    0007 §10) is judged on this "after whitespace normalization", so a
    page of only line breaks/blank runs (a common near-empty-page
    artifact, e.g. a section divider) is correctly treated as having no
    usable text layer."""
    return sum(1 for char in text if not char.isspace())


def _list_pages(pages: list[int]) -> str:
    shown = ", ".join(str(page) for page in pages[:_MAX_LISTED_PAGES])
    remainder = len(pages) - _MAX_LISTED_PAGES
    if remainder > 0:
        shown += f", … and {remainder} more"
    return shown


class PdfConnector:
    """Reference connector for ``.pdf`` files (issue #348).

    ``extract_outline`` (default ``True``) controls whether the PDF's own
    outline (bookmarks) becomes ``sections`` and the document ``title`` —
    set ``False`` to reproduce a plain paginated passage with neither.
    ``min_chars_per_page`` (default 16) is the ADR 0007 §10-required,
    connector-documented per-page text-layer threshold. ``max_file_bytes``
    (default 64 MiB) caps the raw file size checked before any parsing.
    """

    def __init__(
        self,
        *,
        extract_outline: bool = True,
        min_chars_per_page: int = _DEFAULT_MIN_CHARS_PER_PAGE,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        if PdfReader is None:
            raise ImportError(
                'PdfConnector requires pypdf — install with pip install "langchain-taguru[pdf]"'
            ) from _PYPDF_IMPORT_ERROR
        self._extract_outline = extract_outline
        self._min_chars_per_page = min_chars_per_page
        self._max_file_bytes = max_file_bytes

    @property
    def parser(self) -> str:
        return PARSER_NAME

    @property
    def parser_version(self) -> str:
        return PARSER_VERSION

    def supports(self, reference: str) -> bool:
        return Path(reference).suffix.lower() in _SUPPORTED_SUFFIXES

    def _options_digest(self) -> str:
        return options_digest(
            {
                "extract_outline": self._extract_outline,
                "min_chars_per_page": self._min_chars_per_page,
                "max_file_bytes": self._max_file_bytes,
            }
        )

    def _fingerprint(self, raw_content_sha256: str) -> FingerprintInputs:
        return FingerprintInputs(
            raw_content_sha256=raw_content_sha256,
            parser=self.parser,
            parser_version=self.parser_version,
            parse_options_digest=self._options_digest(),
        )

    def _failure(
        self,
        *,
        source: str,
        display_name: str,
        code: DiagnosticCode,
        message: str,
        raw_content_sha256: str,
    ) -> ConnectorDocument:
        return ConnectorDocument(
            source=source,
            text="",
            metadata=ConnectorMetadata(
                origin_uri=source, display_name=display_name, content_type="application/pdf"
            ),
            fingerprint_inputs=self._fingerprint(raw_content_sha256),
            diagnostics=(Diagnostic(code=code, message=message, source=source),),
        )

    def read(self, reference: str) -> ConnectorDocument:
        # As in TextFileConnector: `reference` itself is the source id
        # (ADR 0007 §6.1); `Path` is only ever used for filesystem access
        # below, never for identity.
        path = Path(reference)
        source = file_source_id(reference)
        display_name = path.name

        def failure(
            code: DiagnosticCode, message: str, raw_content_sha256: str = _EMPTY_SHA256
        ) -> ConnectorDocument:
            return self._failure(
                source=source,
                display_name=display_name,
                code=code,
                message=message,
                raw_content_sha256=raw_content_sha256,
            )

        source_diagnostic = check_source_id(source)
        if source_diagnostic is not None:
            return failure(source_diagnostic.code, source_diagnostic.message)

        if not self.supports(reference):
            return failure(
                "unsupported_format", f"unsupported extension {path.suffix!r} (only .pdf)"
            )

        try:
            size = path.stat().st_size
        except OSError as error:
            return failure("unreadable", str(error))
        if size > self._max_file_bytes:
            return failure(
                "content_too_large",
                f"{size} bytes exceeds the {self._max_file_bytes}-byte file cap",
            )

        try:
            raw = path.read_bytes()
        except OSError as error:
            return failure("unreadable", str(error))
        # The raw bytes, before parsing — ADR 0007 §5's connector-native
        # checkpoint fingerprint, deliberately distinct from any hash of
        # the extracted text (§6.2).
        raw_content_sha256 = hashlib.sha256(raw).hexdigest()

        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted and reader.decrypt("") == 0:
                return failure(
                    "encrypted",
                    "the document requires a password this connector does not have",
                    raw_content_sha256,
                )
            page_count = len(reader.pages)
        except PyPdfError as error:
            return failure("corrupt", str(error), raw_content_sha256)
        except Exception as error:  # noqa: BLE001 - pypdf can raise plain
            # ValueError/struct.error/KeyError for some malformed inputs
            # that never subclass PyPdfError; every such failure is still
            # this document's structure failing to parse.
            return failure("corrupt", str(error), raw_content_sha256)

        if page_count == 0:
            return failure("corrupt", "the document has no pages", raw_content_sha256)

        page_texts: list[str] = []
        failed_pages: list[int] = []
        for index in range(page_count):
            try:
                page_texts.append(reader.pages[index].extract_text())
            except Exception:  # noqa: BLE001 - one page's own decode failure
                # must not fail the whole document; it is reported as
                # partial_extraction below instead.
                failed_pages.append(index + 1)
                page_texts.append("")

        empty_pages = [
            index + 1
            for index, text in enumerate(page_texts)
            if index + 1 not in failed_pages and _page_char_count(text) < self._min_chars_per_page
        ]

        paragraphs: list[str] = []
        locators: list[LocatorEntry] = []
        page_paragraph_starts: list[int | None] = [None] * page_count
        for index, page_text in enumerate(page_texts):
            page_number = index + 1
            if page_number in failed_pages or page_number in empty_pages:
                continue
            page_paragraphs = split_paragraphs(page_text)
            if not page_paragraphs:
                continue
            page_paragraph_starts[index] = len(paragraphs)
            for paragraph in page_paragraphs:
                locators.append(
                    LocatorEntry(
                        paragraph=len(paragraphs),
                        locator=Locator(kind="page", value=str(page_number)),
                    )
                )
                paragraphs.append(paragraph)

        if not paragraphs:
            if empty_pages:
                unusable = sorted({*empty_pages, *failed_pages})
                return failure(
                    "ocr_required",
                    f"no extractable text layer on page(s) {_list_pages(unusable)}",
                    raw_content_sha256,
                )
            return failure(
                "corrupt",
                f"failed to extract text on page(s) {_list_pages(failed_pages)}",
                raw_content_sha256,
            )

        text = "\n\n".join(paragraphs)
        if _byte_len(text) > MAX_PASSAGE_BYTES:
            return failure(
                "content_too_large",
                f"{_byte_len(text)} bytes of extracted text exceeds the "
                f"{MAX_PASSAGE_BYTES}-byte passage cap",
                raw_content_sha256,
            )

        sections: list[SectionEntry] = []
        title: str | None = None
        if self._extract_outline:
            sections, title = self._outline_sections(reader, page_paragraph_starts)

        diagnostics: list[Diagnostic] = []
        if empty_pages:
            diagnostics.append(
                Diagnostic(
                    code="ocr_required",
                    message=f"no extractable text layer on page(s) {_list_pages(empty_pages)}",
                    source=source,
                )
            )
        if failed_pages:
            diagnostics.append(
                Diagnostic(
                    code="partial_extraction",
                    message=f"failed to extract text on page(s) {_list_pages(failed_pages)}",
                    source=source,
                )
            )

        return ConnectorDocument(
            source=source,
            text=text,
            locators=tuple(locators),
            sections=tuple(sections),
            metadata=ConnectorMetadata(
                origin_uri=source,
                display_name=display_name,
                title=title,
                content_type="application/pdf",
            ),
            fingerprint_inputs=self._fingerprint(raw_content_sha256),
            diagnostics=tuple(diagnostics),
        )

    def _outline_sections(
        self, reader: PdfReader, page_paragraph_starts: list[int | None]
    ) -> tuple[list[SectionEntry], str | None]:
        """One :class:`SectionEntry` per outline (bookmark) entry that
        resolves to a page carrying at least one paragraph — at the FIRST
        paragraph of that page, since an outline entry names a page, never
        a paragraph (ADR 0007 §7.4's "derived from the document's own
        structure"). An entry whose page has no usable text (scanned,
        failed, or simply blank) falls forward to the next page that does;
        an entry with nothing to fall forward to is dropped, silently —
        the same degrade-quietly posture ``TextFileConnector._headings``
        already takes for a heading that doesn't parse."""
        entries: list[SectionEntry] = []
        seen_paragraphs: set[int] = set()
        titles: list[str] = []
        try:
            outline = reader.outline
        except Exception:  # noqa: BLE001 - a malformed outline degrades to none
            outline = []
        for raw_title, page_index in self._flatten_outline(reader, outline):
            titles.append(raw_title)
            if page_index is None:
                continue
            paragraph = self._first_paragraph_at_or_after(page_paragraph_starts, page_index)
            if paragraph is None or paragraph in seen_paragraphs:
                continue
            label = raw_title.strip()
            if not label or _byte_len(label) > MAX_SECTION_BYTES:
                continue
            seen_paragraphs.add(paragraph)
            entries.append(SectionEntry(paragraph=paragraph, section=label))
        entries.sort(key=lambda entry: entry.paragraph)
        return entries, self._document_title(reader, titles)

    def _flatten_outline(
        self, reader: PdfReader, items: object
    ) -> Iterator[tuple[str, int | None]]:
        """Depth-first walk of pypdf's nested outline shape (a list whose
        elements are either a ``Destination`` or a nested list of the
        children under the previous sibling) into ``(title, page_index)``
        pairs, document order, one entry per bookmark regardless of
        nesting depth — ADR 0007 §7 locators/sections carry no hierarchy
        of their own."""
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, list):
                yield from self._flatten_outline(reader, item)
                continue
            title = getattr(item, "title", None)
            if not isinstance(title, str):
                continue
            try:
                page_index = reader.get_destination_page_number(item)
            except Exception:  # noqa: BLE001 - a malformed /Dest degrades to "unplaced"
                page_index = None
            yield title, page_index

    @staticmethod
    def _first_paragraph_at_or_after(starts: list[int | None], page_index: int) -> int | None:
        for candidate in starts[max(page_index, 0) :]:
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _document_title(reader: PdfReader, outline_titles: list[str]) -> str | None:
        try:
            metadata_title = reader.metadata.title if reader.metadata is not None else None
        except Exception:  # noqa: BLE001 - malformed /Info degrades to "no title"
            metadata_title = None
        if isinstance(metadata_title, str) and metadata_title.strip():
            return metadata_title.strip()
        if outline_titles:
            return outline_titles[0].strip() or None
        return None
