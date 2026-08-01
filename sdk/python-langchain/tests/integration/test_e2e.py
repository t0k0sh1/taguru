"""Retriever + Ingester against the real server (LLM stays a deterministic fake)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import Locator, Taguru

from taguru_langchain import TaguruIngester, TaguruRetriever
from taguru_langchain.ingest_connectors import (
    LocatorEntry,
    TextFileConnector,
    ingest_connector_document,
)


def test_retriever_serves_both_lanes_from_the_seeded_context(
    client: Taguru, server: object, seeded: str
) -> None:
    retriever = TaguruRetriever(context=seeded, client=client, k=8)
    documents = retriever.invoke("青嶺酒造")

    assert documents
    graph_backed = [d for d in documents if "graph" in d.metadata["lane"]]
    assert graph_backed, "the graph lane must surface the seeded facts"
    # The 杜氏 fact is paragraph-located, so its Document carries the verbatim
    # paragraph text as page_content.
    located = [d for d in documents if d.metadata.get("paragraph") == 1]
    assert located
    assert "高瀬" in located[0].page_content
    assert located[0].metadata["source"] == "docs/aomine.md"
    labels = {a["label"] for d in graph_backed for a in d.metadata.get("associations", [])}
    assert "杜氏" in labels


async def test_retriever_async_lane(client: Taguru, seeded: str) -> None:
    retriever = TaguruRetriever(context=seeded, k=4)  # env-based connection
    documents = await retriever.ainvoke("青嶺酒造")
    assert documents
    assert len(documents) <= 4


def test_retriever_text_lane_catches_answer_shaped_queries(client: Taguru, seeded: str) -> None:
    retriever = TaguruRetriever(context=seeded, include_graph=False, k=3)
    documents = retriever.invoke("1907年に創業した")
    assert documents
    assert documents[0].metadata["lane"] == "text"
    assert "1907年" in documents[0].page_content


INGEST_ANSWER = json.dumps(
    {
        "associations": [
            {
                "subject": "月白堂",
                "label": "kind",
                "object": "和菓子屋",
                "weight": 1.0,
                "paragraph": 0,
            },
            {
                "subject": "月白堂",
                "label": "名物",
                "object": "栗きんとん",
                "weight": 1.0,
                "paragraph": 1,
            },
        ],
        "aliases": [{"alias": "Geppakudo", "canonical": "月白堂", "kind": "concept"}],
        "questions": [{"paragraph": 1, "question": "名物は何ですか?"}],
    },
    ensure_ascii=False,
)

SHOP_DOC = "月白堂は架空の和菓子屋である。\n\n名物は栗きんとんである。"


def test_ingester_end_to_end_and_idempotent(client: Taguru, server: object) -> None:
    llm = FakeListChatModel(responses=[INGEST_ANSWER, INGEST_ANSWER, INGEST_ANSWER])
    ingester = TaguruIngester(
        context="wagashi",
        llm=llm,
        client=client,
        create_context=True,
        context_description="和菓子屋の知識",
        questions=2,
    )

    # Dry run first: rendered, validated, nothing applied.
    dry = ingester.ingest_text(SHOP_DOC, source="docs/geppakudo.md", dry_run=True)
    assert dry.ok and dry.ndjson is not None
    assert not client.contexts.exists("wagashi")

    outcomes = ingester.ingest_documents(
        [Document(page_content=SHOP_DOC, metadata={"source": "docs/geppakudo.md"})]
    )
    assert outcomes[0].ok
    assert outcomes[0].created
    assert outcomes[0].associations == 2
    assert outcomes[0].aliases == 1
    assert outcomes[0].passage_stored
    assert outcomes[0].questions_stored == 1
    # 501 (no embedding provider in tests) must stay silent.
    assert outcomes[0].embeddings_refresh_warning is None

    ctx = client.context("wagashi")
    match = ctx.query(subject="月白堂", label="名物").matches[0]
    assert match.object == "栗きんとん"
    assert match.attributions[0].paragraph == 1
    assert ctx.resolve("Geppakudo")[0].name == "月白堂"
    assert "栗きんとん" in ctx.lookup_passages(["docs/geppakudo.md"]).passages["docs/geppakudo.md"]

    # Re-ingesting the same document is a per-source replace, never a
    # double-count.
    before = ctx.query(subject="月白堂", label="名物").matches[0]
    again = ingester.ingest_text(SHOP_DOC, source="docs/geppakudo.md")
    assert again.ok and again.retracted > 0
    after = ctx.query(subject="月白堂", label="名物").matches[0]
    assert after.weight == before.weight
    assert after.count == before.count

    # The ingested knowledge is immediately retrievable through the retriever.
    retriever = TaguruRetriever(context="wagashi", client=client, k=4)
    documents = retriever.invoke("月白堂")
    assert any("栗きんとん" in d.page_content for d in documents)

    client.contexts.delete("wagashi")


def test_connector_document_round_trips_sections_and_locators_to_citations(
    client: Taguru, tmp_path: Path
) -> None:
    """ADR 0007 (issue #347)'s own acceptance bar, against the real server
    rather than the unit-test FakeServer mock: a connector's normalized
    document reaches TaguruIngester end to end (``taguru_langchain.
    ingest_connectors.TextFileConnector`` -> ``ingest_connector_document``
    -> ``POST /import``), and both its section heading and a typed locator
    survive storage and the ``/citations`` response round trip (ADR 0007
    §7, landed by #346/PR #360)."""
    text = "# 藍染工房\n\n藍染めの技法を伝える工房である。\n\n代表作は暖簾である。\n"
    path = tmp_path / "aizome.md"
    path.write_text(text, encoding="utf-8")

    document = TextFileConnector().read(str(path))
    assert document.diagnostics == ()
    assert len(document.sections) == 1
    assert document.sections[0].paragraph == 0
    assert document.sections[0].section == "藍染工房"

    # TextFileConnector never produces a locator (no natural page/slide in
    # .md/.txt) — a synthetic one exercises the same wire path #348-#351's
    # real page/slide/sheet connectors will use.
    document = dataclasses.replace(
        document, locators=(LocatorEntry(paragraph=2, locator=Locator(kind="page", value="1")),)
    )

    llm = FakeListChatModel(
        responses=[json.dumps({"associations": [], "aliases": [], "questions": []})]
    )
    ingester = TaguruIngester(
        context="aizome",
        llm=llm,
        client=client,
        create_context=True,
        context_description="connector round trip (issue #347)",
    )
    try:
        outcome = ingest_connector_document(ingester, document)
        assert outcome.ok
        assert outcome.sections_stored == 1
        assert outcome.locators_stored == 1

        ctx = client.context("aizome")
        heading_citation = ctx.cite_passage(document.source, 0)
        assert heading_citation.section == "藍染工房"

        located_citation = ctx.cite_passage(document.source, 2)
        assert located_citation.locator == Locator(kind="page", value="1")
    finally:
        # Guaranteed even if an assertion above fails — otherwise "aizome"
        # is left behind on the shared real server and the next run starts
        # from create_context=True colliding with an already-existing
        # context, turning one failure into a second, unrelated one.
        if client.contexts.exists("aizome"):
            client.contexts.delete("aizome")


def test_pdf_connector_document_round_trips_page_locators_to_citations(
    client: Taguru, tmp_path: Path
) -> None:
    """The PDF connector's own acceptance bar (issue #348), against the
    real server: unlike the synthetic locator #347's own round-trip test
    above attaches by hand, this locator comes straight from
    ``PdfConnector.read`` — the PDF's own page boundaries — and still
    reaches ``Citation.locator`` unchanged."""
    pytest.importorskip("pypdf")
    from taguru_langchain.ingest_connectors import PdfConnector

    from .._pdfs import text_pdf

    # ASCII only: the hand-built test PDF (tests/_pdfs.py) uses the base14
    # Helvetica font with no embedded CJK glyphs/ToUnicode CMap, unlike
    # this suite's other fixtures.
    path = tmp_path / "workshop.pdf"
    path.write_bytes(
        text_pdf(
            [
                "Indigo Workshop\n\nThe workshop preserves indigo dyeing techniques.",
                "Its best known product is a noren curtain.",
            ]
        )
    )

    document = PdfConnector().read(str(path))
    assert document.diagnostics == ()
    assert len(document.locators) == 3
    assert document.locators[2].locator == Locator(kind="page", value="2")

    llm = FakeListChatModel(
        responses=[json.dumps({"associations": [], "aliases": [], "questions": []})]
    )
    ingester = TaguruIngester(
        context="indigo-workshop",
        llm=llm,
        client=client,
        create_context=True,
        context_description="PDF connector round trip (issue #348)",
    )
    try:
        outcome = ingest_connector_document(ingester, document)
        assert outcome.ok
        assert outcome.locators_stored == len(document.locators)

        ctx = client.context("indigo-workshop")
        located_citation = ctx.cite_passage(document.source, 2)
        assert located_citation.locator == Locator(kind="page", value="2")
    finally:
        if client.contexts.exists("indigo-workshop"):
            client.contexts.delete("indigo-workshop")


def test_html_connector_document_round_trips_fragment_locators_to_citations(
    client: Taguru,
) -> None:
    """The HTML connector's own acceptance bar (issue #349), against the
    real server: a fetched page's own heading `id` becomes a `fragment`
    locator, and its heading hierarchy becomes a breadcrumb `section` —
    both surviving storage and the `/citations` response round trip
    unchanged, the same bar #348's PDF-page-locator test above clears for
    page numbers instead."""
    from taguru_langchain.ingest_connectors import HtmlConnector

    from .._httpd import Route, serve

    body = b"""<html><head><title>Weaving Studio</title></head><body><main>
    <h1 id="top">Weaving Studio</h1>
    <p>The studio preserves traditional loom weaving.</p>
    <h2 id="products">Products</h2>
    <p>Its best known product is an obi sash.</p>
    </main></body></html>"""

    with serve({"/studio": Route(body=body)}) as httpd:
        document = HtmlConnector(allow_private_networks=True).read(f"{httpd.base_url}/studio")

    assert document.diagnostics == ()
    assert [(s.paragraph, s.section) for s in document.sections] == [
        (0, "Weaving Studio"),
        (2, "Weaving Studio > Products"),
    ]
    assert document.locators[2].locator == Locator(kind="fragment", value="products")

    llm = FakeListChatModel(
        responses=[json.dumps({"associations": [], "aliases": [], "questions": []})]
    )
    ingester = TaguruIngester(
        context="weaving-studio",
        llm=llm,
        client=client,
        create_context=True,
        context_description="HTML connector round trip (issue #349)",
    )
    try:
        outcome = ingest_connector_document(ingester, document)
        assert outcome.ok
        assert outcome.sections_stored == len(document.sections)
        assert outcome.locators_stored == len(document.locators)

        ctx = client.context("weaving-studio")
        heading_citation = ctx.cite_passage(document.source, 2)
        assert heading_citation.section == "Weaving Studio > Products"
        assert heading_citation.locator == Locator(kind="fragment", value="products")
    finally:
        if client.contexts.exists("weaving-studio"):
            client.contexts.delete("weaving-studio")


def test_docx_connector_document_round_trips_table_locators_to_citations(
    client: Taguru, tmp_path: Path
) -> None:
    """The DOCX connector's own acceptance bar (issue #350), against the
    real server: a table's own document position becomes a `table`
    locator — unlike #349's HTML `fragment` locator (spent on the
    surrounding heading instead), this is the one DOCX has no page/anchor
    of its own to compete with (ADR 0007 §7.2) — and both it and the
    heading hierarchy's breadcrumb `section` survive storage and the
    `/citations` response round trip unchanged, the same bar #348's
    PDF-page-locator test above clears for page numbers instead."""
    pytest.importorskip("docx")
    from taguru_langchain.ingest_connectors import DocxConnector

    from .._docx import docx_bytes, heading, para, table

    body = (
        heading("Pottery Studio", 1)
        + para("The studio preserves traditional raku firing.")
        + heading("Products", 2)
        + table([["Item", "Price"], ["Tea bowl", "3000"]])
    )
    path = tmp_path / "pottery.docx"
    path.write_bytes(docx_bytes(body))

    document = DocxConnector().read(str(path))
    assert document.diagnostics == ()
    assert [(s.paragraph, s.section) for s in document.sections] == [
        (0, "Pottery Studio"),
        (2, "Pottery Studio > Products"),
    ]
    assert document.locators[0].locator == Locator(kind="table", value="1")

    llm = FakeListChatModel(
        responses=[json.dumps({"associations": [], "aliases": [], "questions": []})]
    )
    ingester = TaguruIngester(
        context="pottery-studio",
        llm=llm,
        client=client,
        create_context=True,
        context_description="DOCX connector round trip (issue #350)",
    )
    try:
        outcome = ingest_connector_document(ingester, document)
        assert outcome.ok
        assert outcome.sections_stored == len(document.sections)
        assert outcome.locators_stored == len(document.locators)

        ctx = client.context("pottery-studio")
        table_paragraph = document.locators[0].paragraph
        table_citation = ctx.cite_passage(document.source, table_paragraph)
        assert table_citation.section == "Pottery Studio > Products"
        assert table_citation.locator == Locator(kind="table", value="1")
    finally:
        if client.contexts.exists("pottery-studio"):
            client.contexts.delete("pottery-studio")
