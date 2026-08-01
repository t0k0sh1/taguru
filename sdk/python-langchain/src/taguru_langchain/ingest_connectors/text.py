"""The reference connector (ADR 0007 §5, issue #347): ``.md``/``.txt`` files into
:class:`~taguru_langchain.ingest_connectors.document.ConnectorDocument`.

The minimal implementation the issue asks for — proof that the normalized
document contract reaches ``TaguruIngester`` end to end, not a full Markdown
parser. Heading extraction is a single-line ATX-heading heuristic (``#`` to
``######``, plus an optional closing sequence per the CommonMark rule that
it must be preceded by a space); anything more (setext headings, headings
sharing a paragraph with body text because a blank line is missing) is
simply not recognized as a section boundary, which degrades quietly, never
incorrectly — a paragraph that doesn't parse as a heading just carries no
``section``, exactly like ``taguru extract`` itself has never emitted
sections.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

from .._extract import MAX_PASSAGE_BYTES, split_paragraphs
from .document import (
    MAX_SECTION_BYTES,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    DiagnosticCode,
    FingerprintInputs,
    SectionEntry,
    options_digest,
)
from .sources import check_source_id, file_source_id

PARSER_NAME: Final = "taguru-text-connector"
PARSER_VERSION: Final = "1.0.0"

_SUPPORTED_SUFFIXES: Final = frozenset({".md", ".txt"})

# A single-line ATX heading: 1-6 '#', required whitespace, the title, and an
# optional closing '#' run that (per CommonMark) must itself be preceded by
# whitespace to count as a closer rather than literal trailing text (so
# "# C#" keeps its trailing '#', but "## Heading ##" does not).
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?$")

_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


class TextFileConnector:
    """Reference connector for ``.md``/``.txt`` files.

    ``extract_headings`` (default ``True``) controls whether ``.md`` ATX
    headings are emitted as ``sections`` — set ``False`` to reproduce a
    plain-text passage with no section metadata even for Markdown input.
    ``.txt`` never carries sections regardless of this flag: it has no
    heading syntax to extract.
    """

    def __init__(self, *, extract_headings: bool = True) -> None:
        self._extract_headings = extract_headings

    @property
    def parser(self) -> str:
        return PARSER_NAME

    @property
    def parser_version(self) -> str:
        return PARSER_VERSION

    def supports(self, reference: str) -> bool:
        return Path(reference).suffix.lower() in _SUPPORTED_SUFFIXES

    def _options_digest(self) -> str:
        return options_digest({"extract_headings": self._extract_headings})

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
        content_type: str | None,
        code: DiagnosticCode,
        message: str,
        raw_content_sha256: str,
    ) -> ConnectorDocument:
        return ConnectorDocument(
            source=source,
            text="",
            metadata=ConnectorMetadata(
                origin_uri=source, display_name=display_name, content_type=content_type
            ),
            fingerprint_inputs=self._fingerprint(raw_content_sha256),
            diagnostics=(Diagnostic(code=code, message=message, source=source),),
        )

    def read(self, reference: str) -> ConnectorDocument:
        path = Path(reference)
        source = file_source_id(str(path))
        display_name = path.name
        is_markdown = path.suffix.lower() == ".md"
        content_type = "text/markdown" if is_markdown else "text/plain"

        source_diagnostic = check_source_id(source)
        if source_diagnostic is not None:
            return self._failure(
                source=source,
                display_name=display_name,
                content_type=content_type,
                code=source_diagnostic.code,
                message=source_diagnostic.message,
                raw_content_sha256=_EMPTY_SHA256,
            )

        if not self.supports(reference):
            return self._failure(
                source=source,
                display_name=display_name,
                content_type=content_type,
                code="unsupported_format",
                message=f"unsupported extension {path.suffix!r} (only .md/.txt)",
                raw_content_sha256=_EMPTY_SHA256,
            )

        try:
            size = path.stat().st_size
        except OSError as error:
            return self._failure(
                source=source,
                display_name=display_name,
                content_type=content_type,
                code="unreadable",
                message=str(error),
                raw_content_sha256=_EMPTY_SHA256,
            )
        if size > MAX_PASSAGE_BYTES:
            return self._failure(
                source=source,
                display_name=display_name,
                content_type=content_type,
                code="content_too_large",
                message=f"{size} bytes exceeds the {MAX_PASSAGE_BYTES}-byte passage cap",
                raw_content_sha256=_EMPTY_SHA256,
            )

        try:
            raw = path.read_bytes()
        except OSError as error:
            return self._failure(
                source=source,
                display_name=display_name,
                content_type=content_type,
                code="unreadable",
                message=str(error),
                raw_content_sha256=_EMPTY_SHA256,
            )
        raw_content_sha256 = hashlib.sha256(raw).hexdigest()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            return self._failure(
                source=source,
                display_name=display_name,
                content_type=content_type,
                code="corrupt",
                message=str(error),
                raw_content_sha256=raw_content_sha256,
            )

        # paragraph.rs::split (and its Python twin, split_paragraphs) do no
        # normalization at all — a leading BOM would otherwise land inside
        # paragraph 0 verbatim. Mirrors src/extract.rs's read_document.
        text = text.removeprefix("﻿")

        sections: list[SectionEntry] = (
            self._sections(text) if is_markdown and self._extract_headings else []
        )
        title = self._title(text) if is_markdown and self._extract_headings else None

        return ConnectorDocument(
            source=source,
            text=text,
            sections=tuple(sections),
            metadata=ConnectorMetadata(
                origin_uri=source,
                display_name=display_name,
                title=title,
                content_type=content_type,
            ),
            fingerprint_inputs=self._fingerprint(raw_content_sha256),
        )

    def _sections(self, text: str) -> list[SectionEntry]:
        """One entry per paragraph whose ENTIRE content is a single ATX
        heading line — never a heading sharing a paragraph with body text,
        since that would require guessing where the heading ends without a
        blank-line boundary to trust. Paragraph indices are
        ``split_paragraphs(text)``'s own indices, the exact contract ADR
        0007 §5 requires (a connector never numbers its own paragraphs)."""
        entries: list[SectionEntry] = []
        for index, paragraph in enumerate(split_paragraphs(text)):
            match = _HEADING_RE.match(paragraph)
            if match is None:
                continue
            label = match.group(2).strip()
            if not label or _byte_len(label) > MAX_SECTION_BYTES:
                continue
            entries.append(SectionEntry(paragraph=index, section=label))
        return entries

    def _title(self, text: str) -> str | None:
        """The first H1 (``# ...``) heading, if any — ``None`` otherwise.
        Independent of :meth:`_sections`'s paragraph-indexed entries: a
        title is metadata, not a citation locator."""
        for paragraph in split_paragraphs(text):
            match = _HEADING_RE.match(paragraph)
            if match is not None and len(match.group(1)) == 1:
                label = match.group(2).strip()
                if label:
                    return label
        return None
