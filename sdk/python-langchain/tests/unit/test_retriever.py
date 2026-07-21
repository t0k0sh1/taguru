"""TaguruRetriever against a routed fake server (no LLM involved)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruRetriever

from .conftest import FakeServer


def make_retriever(
    sync_client: Taguru, async_client: AsyncTaguru, **kwargs: Any
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


def test_a_target_is_required(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    with pytest.raises(PydanticValidationError, match="name a target"):
        TaguruRetriever(client=sync_client, async_client=async_client)


def test_cross_contexts_tag_documents_and_share_one_text_call(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    retriever = TaguruRetriever(
        contexts=["sake", "tea"], client=sync_client, async_client=async_client
    )
    documents = retriever.invoke("青嶺酒造")

    # Every Document names the context it came from.
    assert all(d.metadata["context"] in {"sake", "tea"} for d in documents)
    assert {d.metadata["context"] for d in documents} == {"sake", "tea"}

    # The graph lane ran per context; the text lane rode the server's own
    # cross-context search — one top-level call naming both targets.
    resolves = [path for path, _ in fake_server.calls if path.endswith("/resolve")]
    assert resolves == ["/contexts/sake/resolve", "/contexts/tea/resolve"]
    cross_searches = [body for path, body in fake_server.calls if path == "/sources/search"]
    assert cross_searches == [{"contexts": ["sake", "tea"], "query": "青嶺酒造", "limit": 5}]

    # Same source id in two contexts stays two Documents (keys carry context).
    graph_backed = [d for d in documents if "graph" in d.metadata["lane"]]
    by_context = {d.metadata["context"] for d in graph_backed}
    assert by_context == {"sake", "tea"}


def test_groups_resolve_to_members_nested_children_included(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    retriever = TaguruRetriever(groups=["parent"], client=sync_client, async_client=async_client)
    documents = retriever.invoke("青嶺酒造")

    # parent reaches sake directly and tea through its child group.
    fetched = [path for path, _ in fake_server.calls if path.startswith("/groups/")]
    assert set(fetched) == {"/groups/parent", "/groups/childg"}
    cross_searches = [body for path, body in fake_server.calls if path == "/sources/search"]
    assert cross_searches == [{"contexts": ["sake", "tea"], "query": "青嶺酒造", "limit": 5}]
    assert {d.metadata["context"] for d in documents} == {"sake", "tea"}


def test_still_resolves_the_groups_it_can_when_one_group_fails_to_fetch(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    retriever = TaguruRetriever(
        groups=["parent", "no-such-group"], client=sync_client, async_client=async_client
    )
    documents = retriever.invoke("青嶺酒造")
    # parent's members (sake, tea) still come back even though the
    # sibling group 404s.
    assert {d.metadata["context"] for d in documents} == {"sake", "tea"}


async def test_async_still_resolves_the_groups_it_can_when_one_group_fails_to_fetch(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    retriever = TaguruRetriever(
        groups=["parent", "no-such-group"], client=sync_client, async_client=async_client
    )
    documents = await retriever.ainvoke("青嶺酒造")
    assert {d.metadata["context"] for d in documents} == {"sake", "tea"}


def test_keeps_a_healthy_targets_graph_docs_when_another_targets_graph_lane_errors(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.fail_contexts.add("tea")
    retriever = TaguruRetriever(
        contexts=["sake", "tea"], client=sync_client, async_client=async_client
    )
    documents = retriever.invoke("青嶺酒造")

    # sake's graph lane never touched the failing context, so its docs
    # still show up.
    graph_docs = [d for d in documents if "graph" in d.metadata["lane"]]
    assert graph_docs
    assert all(d.metadata["context"] == "sake" for d in graph_docs)
    # tea's cross-context text hit isn't a per-context call, so it
    # still shows up despite tea's graph lane failing.
    assert any(d.metadata["context"] == "tea" for d in documents)


async def test_async_keeps_a_healthy_targets_graph_docs_when_another_targets_graph_lane_errors(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.fail_contexts.add("tea")
    retriever = TaguruRetriever(
        contexts=["sake", "tea"], client=sync_client, async_client=async_client
    )
    documents = await retriever.ainvoke("青嶺酒造")

    graph_docs = [d for d in documents if "graph" in d.metadata["lane"]]
    assert graph_docs
    assert all(d.metadata["context"] == "sake" for d in graph_docs)
    assert any(d.metadata["context"] == "tea" for d in documents)


def test_keeps_the_graph_lanes_docs_when_the_text_lane_errors(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.fail_text_search = True
    retriever = TaguruRetriever(
        contexts=["sake", "tea"], client=sync_client, async_client=async_client
    )
    documents = retriever.invoke("青嶺酒造")

    # The one cross-context /sources/search call 500s, but it is not a
    # per-target call — both targets' already-fetched graph lanes must
    # survive it rather than being wiped out along with the text hits.
    assert documents
    assert all("graph" in d.metadata["lane"] for d in documents)
    assert {d.metadata["context"] for d in documents} == {"sake", "tea"}


async def test_async_keeps_the_graph_lanes_docs_when_the_text_lane_errors(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.fail_text_search = True
    retriever = TaguruRetriever(
        contexts=["sake", "tea"], client=sync_client, async_client=async_client
    )
    documents = await retriever.ainvoke("青嶺酒造")

    assert documents
    assert all("graph" in d.metadata["lane"] for d in documents)
    assert {d.metadata["context"] for d in documents} == {"sake", "tea"}


async def test_async_cross_matches_sync(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    retriever = TaguruRetriever(
        contexts=["sake"], groups=["childg"], client=sync_client, async_client=async_client
    )
    sync_documents = retriever.invoke("青嶺酒造")
    async_documents = await retriever.ainvoke("青嶺酒造")
    assert [d.page_content for d in async_documents] == [d.page_content for d in sync_documents]
    assert [d.metadata["context"] for d in async_documents] == [
        d.metadata["context"] for d in sync_documents
    ]


def test_close_leaves_a_caller_supplied_client_open(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    retriever = make_retriever(sync_client, async_client)
    retriever.close()
    assert not sync_client._http.is_closed


async def test_aclose_leaves_caller_supplied_clients_open(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    retriever = make_retriever(sync_client, async_client)
    await retriever.aclose()
    assert not sync_client._http.is_closed
    assert not async_client._http.is_closed


def test_close_closes_a_self_built_client() -> None:
    retriever = TaguruRetriever(context="sake", base_url="http://test")
    client = retriever.client
    assert client is not None
    retriever.close()
    assert client._http.is_closed


def test_close_also_closes_the_self_built_async_client() -> None:
    """close() has no event loop to await aclose() with, but must not leak
    the async client's connections when ainvoke was ever used before it."""
    retriever = TaguruRetriever(context="sake", base_url="http://test")
    async_client_ = retriever.async_client
    assert async_client_ is not None
    retriever.close()
    assert async_client_._http.is_closed


async def test_aclose_closes_both_self_built_clients() -> None:
    retriever = TaguruRetriever(context="sake", base_url="http://test")
    client, async_client_ = retriever.client, retriever.async_client
    assert client is not None
    assert async_client_ is not None
    await retriever.aclose()
    assert client._http.is_closed
    assert async_client_._http.is_closed


def test_sync_context_manager_closes_the_self_built_client_on_exit() -> None:
    with TaguruRetriever(context="sake", base_url="http://test") as retriever:
        client, async_client_ = retriever.client, retriever.async_client
        assert client is not None
        assert async_client_ is not None
        assert not client._http.is_closed
    assert client._http.is_closed
    assert async_client_._http.is_closed


async def test_async_context_manager_closes_self_built_clients_on_exit() -> None:
    async with TaguruRetriever(context="sake", base_url="http://test") as retriever:
        client, async_client_ = retriever.client, retriever.async_client
        assert client is not None
        assert async_client_ is not None
    assert client._http.is_closed
    assert async_client_._http.is_closed
