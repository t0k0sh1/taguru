"""Normalized document contract (ADR 0007 §5, issue #347): the one shape
every connector — present or future — produces, and the one shape
:class:`~taguru_langchain.ingest.TaguruIngester` consumes from a connector
instead of reading files directly.

A field-for-field port of the ADR's JSON example. ``text`` is plain
paragraph-joined text — the same blank-line convention
``taguru_langchain._extract.split_paragraphs`` (mirroring src/paragraph.rs)
already defines; a connector never numbers its own paragraphs, it produces
text and lets that splitter decide where paragraphs begin and end.
``locators``/``sections`` index into whatever that splitter yields, exactly
as ``taguru extract`` already assumes for its own question/section lines.

Degradation posture throughout: ``__post_init__`` raises on a document that
would violate a wire-level invariant the server enforces unconditionally
(an empty locator/section field, an oversized tag) — a connector bug should
fail loudly and locally rather than ship a document the server would refuse
anyway. What the server instead drops-and-counts (a paragraph index outside
the document's own range) is deliberately NOT validated here; that decision
is the server's alone (``filter_markers``, src/passages.rs:203) and a
connector cannot know the range without re-running the same paragraph split
the server runs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeVar, cast, get_args

from taguru import Locator

CONNECTOR_DOCUMENT_VERSION: Final = 1
"""Checked for equality, like ``taguru``'s own ``BATCH_VERSION``/
``GROUP_VERSION`` (src/ingest.rs) — a protocol-breaking change bumps this,
never loosens the check to a range."""

# Mirrors the caps src/api.rs enforces server-side (MAX_NAME_BYTES,
# MAX_LOCATOR_KIND_BYTES, MAX_LOCATOR_VALUE_BYTES, MAX_SECTION_BYTES,
# MAX_TAG_BYTES, MAX_TAGS_PER_SOURCE) — checked here too so a connector bug
# fails at construction time, not only after a round trip to the server.
MAX_NAME_BYTES: Final = 1024
MAX_LOCATOR_KIND_BYTES: Final = 64
MAX_LOCATOR_VALUE_BYTES: Final = 512
MAX_SECTION_BYTES: Final = 512
MAX_TAG_BYTES: Final = 128
MAX_TAGS_PER_SOURCE: Final = 32

DiagnosticCode = Literal[
    "unreadable",
    "unsupported_format",
    "encrypted",
    "corrupt",
    "ocr_required",
    "source_id_too_long",
    "content_too_large",
    "partial_extraction",
]
"""ADR 0007 §8's closed, versioned diagnostic vocabulary. Adding a new code
is additive (compatible, ADR 0005 §4); renaming or repurposing an existing
one is breaking — the same posture src/api.rs's ``ErrorCode`` documents for
itself."""

DIAGNOSTIC_CODES: Final[frozenset[str]] = frozenset(get_args(DiagnosticCode))

_T = TypeVar("_T")


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _check_nonempty_size(field_name: str, value: str, max_bytes: int) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if _byte_len(value) > max_bytes:
        raise ValueError(f"{field_name} exceeds {max_bytes} bytes")


def _parse_list(items: object, parser: Callable[[object], _T | None]) -> list[_T] | None:
    """Parses a JSON array field where a single unparseable element
    invalidates the whole list — the same "one bad unit invalidates the
    file" posture ``checkpoints._DocumentCheckpoints._parse`` already
    takes, applied per-field instead of per-file."""
    if not isinstance(items, list):
        return None
    result: list[_T] = []
    for item in items:
        parsed = parser(item)
        if parsed is None:
            return None
        result.append(parsed)
    return result


@dataclass(slots=True, frozen=True, kw_only=True)
class Diagnostic:
    """One structured diagnostic (ADR 0007 §8) — the same three fields
    ``taguru extract``'s own diagnostics sidecar uses for its own per-chunk/
    per-document records (``DiagnosticsSink``/``AttemptRecord``,
    src/extract.rs:2372/:2566), so a downstream tool that already parses one
    JSONL diagnostics shape does not need a second parser."""

    code: DiagnosticCode
    message: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "source": self.source}


def _parse_diagnostic(data: object) -> Diagnostic | None:
    if not isinstance(data, dict):
        return None
    code = data.get("code")
    if not isinstance(code, str) or code not in DIAGNOSTIC_CODES:
        return None
    try:
        return Diagnostic(
            code=cast(DiagnosticCode, code),
            message=str(data["message"]),
            source=str(data["source"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


@dataclass(slots=True, frozen=True, kw_only=True)
class LocatorEntry:
    """One paragraph-indexed typed locator (ADR 0007 §7): a page, slide,
    sheet, table, or other position. Independent of :class:`SectionEntry` —
    unlike a section, a locator does not extend to the next paragraph; a
    connector emits one locator line per paragraph it needs to place."""

    paragraph: int
    locator: Locator

    def to_dict(self) -> dict[str, object]:
        return {
            "paragraph": self.paragraph,
            "locator": {"kind": self.locator.kind, "value": self.locator.value},
        }


def _parse_locator_entry(data: object) -> LocatorEntry | None:
    if not isinstance(data, dict):
        return None
    locator_data = data.get("locator")
    if not isinstance(locator_data, dict):
        return None
    try:
        return LocatorEntry(
            paragraph=int(data["paragraph"]),
            locator=Locator(kind=str(locator_data["kind"]), value=str(locator_data["value"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


@dataclass(slots=True, frozen=True, kw_only=True)
class SectionEntry:
    """One paragraph-indexed free-text section heading — extends to the
    next marker, exactly like a batch file's ``{"paragraph": N, "section":
    ...}`` line (src/ingest.rs's ``SectionLine``). ``section``'s meaning is
    unchanged by ADR 0007 §7.1: prose, never overloaded with positional
    metadata."""

    paragraph: int
    section: str

    def to_dict(self) -> dict[str, object]:
        return {"paragraph": self.paragraph, "section": self.section}


def _parse_section_entry(data: object) -> SectionEntry | None:
    if not isinstance(data, dict):
        return None
    try:
        return SectionEntry(paragraph=int(data["paragraph"]), section=str(data["section"]))
    except (KeyError, TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(slots=True, frozen=True, kw_only=True)
class ConnectorMetadata:
    """ADR 0007 §5's ``metadata`` object — only what ``Citation``/
    ``SourceEntry`` can already represent or what §7 adds. A connector must
    not invent metadata the contract doesn't declare."""

    origin_uri: str
    display_name: str
    title: str | None = None
    canonical_url: str | None = None
    tags: tuple[str, ...] = ()
    content_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "origin_uri": self.origin_uri,
            "display_name": self.display_name,
            "title": self.title,
            "canonical_url": self.canonical_url,
            "tags": list(self.tags),
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: object) -> ConnectorMetadata | None:
        if not isinstance(data, dict):
            return None
        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            return None
        try:
            return cls(
                origin_uri=str(data["origin_uri"]),
                display_name=str(data["display_name"]),
                title=_optional_str(data.get("title")),
                canonical_url=_optional_str(data.get("canonical_url")),
                tags=tuple(raw_tags),
                content_type=_optional_str(data.get("content_type")),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(slots=True, frozen=True, kw_only=True)
class FingerprintInputs:
    """ADR 0007 §5/§6.3's connector-native checkpoint key material.

    ``raw_content_sha256`` hashes the RAW bytes fetched/opened from the
    origin, before parsing — deliberately not ``text`` — so a connector can
    detect "the object is byte-identical to last time" without re-parsing
    at all. This is a different, deliberately non-interchangeable question
    from ``taguru extract``'s own manifest fingerprint, which hashes the
    parsed ``text`` (ADR 0007 §5's ``fingerprint_inputs`` note)."""

    raw_content_sha256: str
    parser: str
    parser_version: str
    parse_options_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_content_sha256": self.raw_content_sha256,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "parse_options_digest": self.parse_options_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> FingerprintInputs | None:
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                raw_content_sha256=str(data["raw_content_sha256"]),
                parser=str(data["parser"]),
                parser_version=str(data["parser_version"]),
                parse_options_digest=str(data["parse_options_digest"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def options_digest(options: dict[str, object]) -> str:
    """A canonical sha256 of a connector's own effective config —
    ``fingerprint_inputs.parse_options_digest`` (ADR 0007 §5). Key order
    never matters (``sort_keys=True``); this is not itself a wire format,
    only a stable fingerprint of one."""
    canonical = json.dumps(options, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True, frozen=True, kw_only=True)
class ConnectorDocument:
    """The one shape every connector — present or future — produces (ADR
    0007 §5), and the one shape a connector hands to
    :func:`taguru_langchain.ingest_connectors.ingest_connector_document`
    instead of a bare string.

    ``diagnostics`` is empty on a clean parse; a non-empty ``diagnostics``
    with an empty ``text`` is the required encoding of "nothing usable was
    extracted" (scanned PDF, encrypted file, unsupported format) — never a
    silently empty ``text`` with no diagnostic, which is exactly the
    "quietly degraded output" #217 forbids and ``__post_init__`` refuses to
    construct.
    """

    source: str
    text: str
    locators: tuple[LocatorEntry, ...] = ()
    sections: tuple[SectionEntry, ...] = ()
    metadata: ConnectorMetadata
    fingerprint_inputs: FingerprintInputs
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("ConnectorDocument.source must not be empty")
        if _byte_len(self.source) > MAX_NAME_BYTES and self.text:
            raise ValueError(
                f"source exceeds {MAX_NAME_BYTES} bytes (MAX_NAME_BYTES, src/api.rs:1137) "
                "— refuse with a 'source_id_too_long' diagnostic and empty text instead of "
                "a source id the server would reject anyway"
            )
        if not self.text and not self.diagnostics:
            raise ValueError(
                "an empty text must carry at least one diagnostic explaining why "
                "(ADR 0007 §5/§10: never a silently empty passage)"
            )
        for entry in self.locators:
            if entry.paragraph < 0:
                raise ValueError("locator paragraph index must be >= 0")
            _check_nonempty_size("locator.kind", entry.locator.kind, MAX_LOCATOR_KIND_BYTES)
            _check_nonempty_size("locator.value", entry.locator.value, MAX_LOCATOR_VALUE_BYTES)
        for section in self.sections:
            if section.paragraph < 0:
                raise ValueError("section paragraph index must be >= 0")
            _check_nonempty_size("section", section.section, MAX_SECTION_BYTES)
        if len(self.metadata.tags) > MAX_TAGS_PER_SOURCE:
            raise ValueError(f"more than {MAX_TAGS_PER_SOURCE} tags (MAX_TAGS_PER_SOURCE)")
        for tag in self.metadata.tags:
            _check_nonempty_size("tag", tag, MAX_TAG_BYTES)

    def to_dict(self) -> dict[str, object]:
        return {
            "connector_document": CONNECTOR_DOCUMENT_VERSION,
            "source": self.source,
            "text": self.text,
            "locators": [entry.to_dict() for entry in self.locators],
            "sections": [entry.to_dict() for entry in self.sections],
            "metadata": self.metadata.to_dict(),
            "fingerprint_inputs": self.fingerprint_inputs.to_dict(),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, data: object) -> ConnectorDocument | None:
        """``None`` on any structural mismatch, including a
        ``connector_document`` version this reader does not recognize —
        the same degrade-never-raise posture every ``from_dict`` in this
        SDK already takes."""
        if not isinstance(data, dict):
            return None
        if data.get("connector_document") != CONNECTOR_DOCUMENT_VERSION:
            return None
        metadata = ConnectorMetadata.from_dict(data.get("metadata"))
        fingerprint_inputs = FingerprintInputs.from_dict(data.get("fingerprint_inputs"))
        if metadata is None or fingerprint_inputs is None:
            return None
        locators = _parse_list(data.get("locators", []), _parse_locator_entry)
        sections = _parse_list(data.get("sections", []), _parse_section_entry)
        diagnostics = _parse_list(data.get("diagnostics", []), _parse_diagnostic)
        if locators is None or sections is None or diagnostics is None:
            return None
        try:
            return cls(
                source=str(data["source"]),
                text=str(data["text"]),
                locators=tuple(locators),
                sections=tuple(sections),
                metadata=metadata,
                fingerprint_inputs=fingerprint_inputs,
                diagnostics=tuple(diagnostics),
            )
        except (KeyError, TypeError, ValueError):
            return None
