"""ConnectorDocument and its component types (ADR 0007 §5, issue #347)."""

from __future__ import annotations

import pytest
from taguru import Locator

from taguru_langchain.ingest_connectors import (
    CONNECTOR_DOCUMENT_VERSION,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    FingerprintInputs,
    LocatorEntry,
    SectionEntry,
    options_digest,
)


def _metadata(
    *,
    origin_uri: str = "doc.md",
    display_name: str = "doc.md",
    title: str | None = None,
    canonical_url: str | None = None,
    tags: tuple[str, ...] = (),
    content_type: str | None = None,
) -> ConnectorMetadata:
    return ConnectorMetadata(
        origin_uri=origin_uri,
        display_name=display_name,
        title=title,
        canonical_url=canonical_url,
        tags=tags,
        content_type=content_type,
    )


def _fingerprint(
    *,
    raw_content_sha256: str = "deadbeef",
    parser: str = "taguru-text-connector",
    parser_version: str = "1.0.0",
    parse_options_digest: str = "cafef00d",
) -> FingerprintInputs:
    return FingerprintInputs(
        raw_content_sha256=raw_content_sha256,
        parser=parser,
        parser_version=parser_version,
        parse_options_digest=parse_options_digest,
    )


def _document(
    *,
    source: str = "doc.md",
    text: str = "paragraph one.",
    locators: tuple[LocatorEntry, ...] = (),
    sections: tuple[SectionEntry, ...] = (),
    metadata: ConnectorMetadata | None = None,
    fingerprint_inputs: FingerprintInputs | None = None,
    diagnostics: tuple[Diagnostic, ...] = (),
) -> ConnectorDocument:
    return ConnectorDocument(
        source=source,
        text=text,
        locators=locators,
        sections=sections,
        metadata=metadata if metadata is not None else _metadata(),
        fingerprint_inputs=fingerprint_inputs if fingerprint_inputs is not None else _fingerprint(),
        diagnostics=diagnostics,
    )


def test_to_dict_round_trips_through_from_dict() -> None:
    document = _document(
        text="paragraph one.\n\nparagraph two.",
        locators=(LocatorEntry(paragraph=1, locator=Locator(kind="page", value="12")),),
        sections=(SectionEntry(paragraph=0, section="導入"),),
        metadata=_metadata(title="タイトル", tags=("quarterly",), content_type="text/markdown"),
    )
    restored = ConnectorDocument.from_dict(document.to_dict())
    assert restored == document


def test_from_dict_rejects_an_unrecognized_connector_document_version() -> None:
    payload = _document().to_dict()
    payload["connector_document"] = 2
    assert ConnectorDocument.from_dict(payload) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("metadata"),
        lambda payload: payload.pop("fingerprint_inputs"),
        lambda payload: payload["locators"].append({"paragraph": "not-an-int", "locator": {}}),
        lambda payload: payload.pop("source"),
        lambda payload: payload.__setitem__("diagnostics", [{"code": "not-a-real-code"}]),
    ],
    ids=[
        "missing_metadata",
        "missing_fingerprint",
        "bad_locator",
        "missing_source",
        "bad_diagnostic",
    ],
)
def test_from_dict_returns_none_on_structural_mismatch(mutate: object) -> None:
    payload = _document(
        locators=(LocatorEntry(paragraph=0, locator=Locator(kind="page", value="1")),),
    ).to_dict()
    mutate(payload)  # type: ignore[operator]
    assert ConnectorDocument.from_dict(payload) is None


def test_from_dict_rejects_non_dict_input() -> None:
    assert ConnectorDocument.from_dict(["not", "a", "dict"]) is None
    assert ConnectorDocument.from_dict(None) is None


def test_empty_text_requires_at_least_one_diagnostic() -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        _document(text="")


def test_empty_text_with_a_diagnostic_is_allowed() -> None:
    document = _document(
        text="", diagnostics=(Diagnostic(code="unreadable", message="boom", source="doc.md"),)
    )
    assert document.text == ""
    assert document.diagnostics[0].code == "unreadable"


def test_source_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="source"):
        _document(source="")


def test_oversized_source_is_only_allowed_when_text_is_empty() -> None:
    long_source = "x" * 1025
    with pytest.raises(ValueError, match="1024"):
        _document(source=long_source)
    # With an empty text and a diagnostic, the same oversized source is
    # exactly what a connector must be able to report (ADR 0007 §6.1).
    document = _document(
        source=long_source,
        text="",
        diagnostics=(
            Diagnostic(code="source_id_too_long", message="too long", source=long_source),
        ),
    )
    assert document.source == long_source


@pytest.mark.parametrize(
    ("kind", "value"),
    [("", "12"), ("page", ""), ("p" * 65, "12"), ("page", "v" * 513)],
    ids=["empty_kind", "empty_value", "kind_too_long", "value_too_long"],
)
def test_locator_kind_and_value_are_size_checked(kind: str, value: str) -> None:
    with pytest.raises(ValueError):
        _document(locators=(LocatorEntry(paragraph=0, locator=Locator(kind=kind, value=value)),))


def test_locator_paragraph_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="paragraph"):
        _document(locators=(LocatorEntry(paragraph=-1, locator=Locator(kind="page", value="1")),))


@pytest.mark.parametrize("section", ["", "s" * 513], ids=["empty", "too_long"])
def test_section_label_is_size_checked(section: str) -> None:
    with pytest.raises(ValueError):
        _document(sections=(SectionEntry(paragraph=0, section=section),))


def test_more_than_max_tags_is_rejected() -> None:
    with pytest.raises(ValueError, match="tags"):
        _document(metadata=_metadata(tags=tuple(f"tag{i}" for i in range(33))))


def test_oversized_tag_is_rejected() -> None:
    with pytest.raises(ValueError):
        _document(metadata=_metadata(tags=("t" * 129,)))


def test_options_digest_is_independent_of_key_order() -> None:
    assert options_digest({"a": 1, "b": 2}) == options_digest({"b": 2, "a": 1})


def test_options_digest_differs_on_value_change() -> None:
    assert options_digest({"extract_headings": True}) != options_digest({"extract_headings": False})


def test_connector_document_version_constant_is_one() -> None:
    # Checked for equality (not a range) — a protocol-breaking change bumps
    # this, never loosens the check (ADR 0007 §5, mirrors BATCH_VERSION).
    assert CONNECTOR_DOCUMENT_VERSION == 1
