"""The connector-to-TaguruIngester bridge (ADR 0007, issue #347) — the
"end to end" wiring the issue asks for."""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import AsyncTaguru, Locator, Taguru

from taguru_langchain import TaguruIngester
from taguru_langchain.ingest_connectors import (
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    FingerprintInputs,
    LocatorEntry,
    SectionEntry,
    aingest_connector_document,
    aingest_connector_documents,
    ingest_connector_document,
    ingest_connector_documents,
)

from .conftest import FakeServer

EMPTY_ANSWER = json.dumps({"associations": [], "aliases": [], "questions": []})


def _fingerprint() -> FingerprintInputs:
    return FingerprintInputs(
        raw_content_sha256="deadbeef",
        parser="taguru-text-connector",
        parser_version="1.0.0",
        parse_options_digest="cafef00d",
    )


def _document(
    *,
    source: str = "doc.md",
    text: str = "paragraph one.\n\nparagraph two.",
    sections: tuple[SectionEntry, ...] = (),
    locators: tuple[LocatorEntry, ...] = (),
    diagnostics: tuple[Diagnostic, ...] = (),
) -> ConnectorDocument:
    return ConnectorDocument(
        source=source,
        text=text,
        sections=sections,
        locators=locators,
        metadata=ConnectorMetadata(origin_uri=source, display_name=source),
        fingerprint_inputs=_fingerprint(),
        diagnostics=diagnostics,
    )


def _ingester(sync_client: Taguru, async_client: AsyncTaguru) -> TaguruIngester:
    return TaguruIngester(
        context="sake",
        llm=FakeListChatModel(responses=[EMPTY_ANSWER, EMPTY_ANSWER]),
        client=sync_client,
        async_client=async_client,
    )


def test_ingest_connector_document_carries_sections_and_locators_onto_the_wire(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    document = _document(
        sections=(SectionEntry(paragraph=0, section="導入"),),
        locators=(LocatorEntry(paragraph=1, locator=Locator(kind="page", value="1")),),
    )
    outcome = ingest_connector_document(_ingester(sync_client, async_client), document)

    assert outcome.ok
    assert outcome.source == document.source
    lines = [json.loads(line) for line in fake_server.imported[0].strip().split("\n")]
    assert {"paragraph": 0, "section": "導入"} in lines
    assert {"paragraph": 1, "locator": {"kind": "page", "value": "1"}} in lines


async def test_aingest_connector_document_carries_sections_and_locators_onto_the_wire(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    document = _document(
        sections=(SectionEntry(paragraph=0, section="導入"),),
    )
    outcome = await aingest_connector_document(_ingester(sync_client, async_client), document)
    assert outcome.ok
    lines = [json.loads(line) for line in fake_server.imported[0].strip().split("\n")]
    assert {"paragraph": 0, "section": "導入"} in lines


def test_a_diagnostics_only_document_never_reaches_the_network(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    document = _document(
        text="",
        diagnostics=(Diagnostic(code="unreadable", message="boom", source="doc.md"),),
    )
    outcome = ingest_connector_document(_ingester(sync_client, async_client), document)

    assert outcome.ok is False
    assert outcome.error is not None and "unreadable" in outcome.error and "boom" in outcome.error
    assert fake_server.imported == []
    assert fake_server.calls == []


async def test_a_diagnostics_only_document_never_reaches_the_network_async(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    document = _document(
        text="",
        diagnostics=(Diagnostic(code="corrupt", message="bad bytes", source="doc.md"),),
    )
    outcome = await aingest_connector_document(_ingester(sync_client, async_client), document)
    assert outcome.ok is False
    assert fake_server.imported == []


def test_include_passage_false_drops_sections_and_locators(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester = TaguruIngester(
        context="sake",
        llm=FakeListChatModel(responses=[EMPTY_ANSWER]),
        client=sync_client,
        async_client=async_client,
        include_passage=False,
    )
    document = _document(
        sections=(SectionEntry(paragraph=0, section="導入"),),
        locators=(LocatorEntry(paragraph=1, locator=Locator(kind="page", value="1")),),
    )
    outcome = ingest_connector_document(ingester, document)
    assert outcome.ok
    ndjson = fake_server.imported[0]
    assert "section" not in ndjson
    assert "locator" not in ndjson


def test_ingest_outcome_counters_come_from_the_servers_import_outcome(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.import_result_override = {
        "context": "sake",
        "source": "doc.md",
        "created": True,
        "retracted": 0,
        "associations": 0,
        "aliases": 0,
        "passage_stored": True,
        "passage_dropped": False,
        "questions_stored": 0,
        "questions_dropped": 0,
        "sections_stored": 1,
        "sections_dropped": 0,
        "locators_stored": 1,
        "locators_dropped": 1,
        "association_paragraphs_dropped": 0,
    }
    document = _document(
        sections=(SectionEntry(paragraph=0, section="導入"),),
        locators=(LocatorEntry(paragraph=1, locator=Locator(kind="page", value="1")),),
    )
    outcome = ingest_connector_document(_ingester(sync_client, async_client), document)
    assert outcome.sections_stored == 1
    assert outcome.locators_stored == 1
    assert outcome.locators_dropped == 1


def test_ingest_connector_documents_continues_past_a_diagnostics_only_document(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    documents = [
        _document(source="a.md"),
        _document(
            source="b.md",
            text="",
            diagnostics=(Diagnostic(code="unreadable", message="boom", source="b.md"),),
        ),
        _document(source="c.md"),
    ]
    outcomes = ingest_connector_documents(_ingester(sync_client, async_client), documents)
    assert [outcome.source for outcome in outcomes] == ["a.md", "b.md", "c.md"]
    assert [outcome.ok for outcome in outcomes] == [True, False, True]
    assert len(fake_server.imported) == 2  # a.md and c.md only


async def test_aingest_connector_documents_continues_past_a_diagnostics_only_document(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    documents = [
        _document(source="a.md"),
        _document(
            source="b.md",
            text="",
            diagnostics=(Diagnostic(code="unreadable", message="boom", source="b.md"),),
        ),
    ]
    outcomes = await aingest_connector_documents(_ingester(sync_client, async_client), documents)
    assert [outcome.ok for outcome in outcomes] == [True, False]
