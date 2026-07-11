"""TaguruRetriever against a routed fake server (no LLM involved)."""

from __future__ import annotations

from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruRetriever

from .conftest import FakeServer


def make_retriever(
    sync_client: Taguru, async_client: AsyncTaguru, **kwargs: object
) -> TaguruRetriever:
    return TaguruRetriever(context="sake", client=sync_client, async_client=async_client, **kwargs)


def test_both_lanes_merge_and_dedup(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    retriever = make_retriever(sync_client, async_client)
    documents = retriever.invoke("青嶺酒造")

    # The (docs/aomine.md, 1) paragraph surfaced in BOTH lanes → one Document.
    merged = [d for d in documents if d.metadata.get("lane") == "graph+text"]
    assert len(merged) == 1
    top = merged[0]
    assert top.page_content == "杜氏は高瀬である。"
    assert top.metadata["source"] == "docs/aomine.md"
    assert top.metadata["paragraph"] == 1
    assert top.metadata["section"] == "人物"  # graph lane brought the citation's section
    assert top.metadata["associations"][0]["label"] == "杜氏"
    assert top.metadata["lanes"]["bm25"]["rank"] == 0
    # Dual-lane evidence outranks everything single-lane.
    assert documents[0] is top

    # The graph-only fact (no stored passage) still became a Document.
    facts = [d for d in documents if d.metadata["paragraph"] is None]
    assert len(facts) == 1
    assert facts[0].page_content == "青嶺酒造 創業年 1907年"
    assert facts[0].metadata["lane"] == "graph"
    assert facts[0].metadata["source"] == "口伝"

    # The text-only hit is present and tagged.
    text_only = [d for d in documents if d.metadata.get("lane") == "text"]
    assert [d.metadata["source"] for d in text_only] == ["docs/other.md"]


async def test_async_lane_matches_sync(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    retriever = make_retriever(sync_client, async_client)
    sync_documents = retriever.invoke("青嶺酒造")
    async_documents = await retriever.ainvoke("青嶺酒造")
    assert [d.page_content for d in async_documents] == [d.page_content for d in sync_documents]
    assert [d.metadata["lane"] for d in async_documents] == [
        d.metadata["lane"] for d in sync_documents
    ]


def test_lane_toggles(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    graph_only = make_retriever(sync_client, async_client, include_text=False)
    documents = graph_only.invoke("青嶺酒造")
    assert all("graph" in d.metadata["lane"] for d in documents)
    assert not any(path.endswith("/sources/search") for path, _ in fake_server.calls)

    fake_server.calls.clear()
    text_only = make_retriever(sync_client, async_client, include_graph=False)
    documents = text_only.invoke("青嶺酒造")
    assert all(d.metadata["lane"] == "text" for d in documents)
    assert not any(path.endswith("/resolve") for path, _ in fake_server.calls)


def test_unresolvable_cue_still_serves_the_text_lane(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    retriever = make_retriever(sync_client, async_client)
    documents = retriever.invoke("無関係な話題")
    assert documents
    assert all(d.metadata["lane"] == "text" for d in documents)


def test_k_truncates(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    retriever = make_retriever(sync_client, async_client, k=1)
    documents = retriever.invoke("青嶺酒造")
    assert len(documents) == 1


def test_graph_only_facts_can_be_switched_off(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    retriever = make_retriever(sync_client, async_client, include_graph_only_facts=False)
    documents = retriever.invoke("青嶺酒造")
    assert all(d.metadata["paragraph"] is not None for d in documents)
