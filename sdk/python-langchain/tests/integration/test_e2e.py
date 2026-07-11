"""Retriever + Ingester against the real server (LLM stays a deterministic fake)."""

from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import Taguru

from taguru_langchain import TaguruIngester, TaguruRetriever


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
