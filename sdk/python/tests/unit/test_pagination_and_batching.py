"""Auto-pagination iterators and association batch chunking."""

from __future__ import annotations

import json

import httpx

from taguru import AliasEntry
from taguru._shared import chunk_associations, dumps_compact
from taguru._types import AssocOp

from .conftest import ok_response, sync_client


def test_contexts_iter_walks_pages_with_keyset_cursor() -> None:
    rows = [
        {
            "name": name,
            "description": "",
            "pinned": False,
            "loaded": False,
            "stats": {
                "associations": 0,
                "concepts": 0,
                "labels": 0,
                "sources": 0,
                "footprint_bytes": 0,
                "top_concepts": [],
                "label_sample": [],
            },
            "usage": {
                "reads": 0,
                "empty_reads": 0,
                "writes": 0,
                "last_read_epoch": 0,
                "last_write_epoch": 0,
            },
        }
        for name in ["a", "b", "c"]
    ]
    cursors: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        after = req.url.params.get("after")
        cursors.append(after)
        if after is None:
            return ok_response({"total": 3, "contexts": rows[:2]})
        if after == "b":
            return ok_response({"total": 3, "contexts": rows[2:]})
        raise AssertionError(after)

    client = sync_client(handler)
    names = [entry.name for entry in client.contexts.iter(limit=2)]
    assert names == ["a", "b", "c"]
    # Page of 1 < limit of 2 stops without an extra empty-page request.
    assert cursors == [None, "b"]


def test_iter_labels_stops_on_short_page() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get("after") is None
        return ok_response({"total": 1, "labels": ["代表銘柄"]})

    client = sync_client(handler)
    assert list(client.context("sake").iter_labels(limit=10)) == ["代表銘柄"]


def test_iter_aliases_flattens_both_namespaces_and_advances_cursor() -> None:
    cursors: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        after = req.url.params.get("after")
        cursors.append(after)
        if after is None:
            return ok_response(
                {"total": 3, "concepts": {"Aomine": "青嶺酒造", "青嶺": "青嶺酒造"}, "labels": {}}
            )
        if after == "concept:青嶺":
            return ok_response({"total": 3, "concepts": {}, "labels": {"brand": "代表銘柄"}})
        raise AssertionError(after)

    client = sync_client(handler)
    entries = list(client.context("sake").iter_aliases(limit=2))
    assert entries == [
        AliasEntry(namespace="concept", alias="Aomine", canonical="青嶺酒造"),
        AliasEntry(namespace="concept", alias="青嶺", canonical="青嶺酒造"),
        AliasEntry(namespace="label", alias="brand", canonical="代表銘柄"),
    ]
    # The second page came back short (1 < limit 2), so iteration stopped
    # without an extra empty-page request.
    assert cursors == [None, "concept:青嶺"]


def _op(i: int) -> AssocOp:
    return {"subject": f"s{i}", "label": "l", "object": "o", "weight": 1.0}


def test_chunk_associations_splits_by_count() -> None:
    ops = [_op(i) for i in range(5)]
    chunks = list(chunk_associations(ops, 2, 10**9))
    assert [len(c) for c in chunks] == [2, 2, 1]


def test_chunk_associations_splits_by_bytes_matching_wire_serialization() -> None:
    ops = [_op(i) for i in range(4)]
    one = len(dumps_compact(_op(0)))
    # Room for exactly two ops per chunk: [op,op] = 2 + one + 1 + one bytes.
    budget = 2 + one + 1 + one
    chunks = list(chunk_associations(ops, 10_000, budget))
    assert [len(c) for c in chunks] == [2, 2]
    for chunk in chunks:
        assert len(dumps_compact(chunk)) <= budget


def test_chunk_associations_yields_oversized_single_op_alone() -> None:
    ops = [_op(0), _op(1)]
    chunks = list(chunk_associations(ops, 10_000, 3))
    assert [len(c) for c in chunks] == [1, 1]


def test_add_associations_batched_sums_applied_counts() -> None:
    batches: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ops = json.loads(req.content)
        batches.append(len(ops))
        return ok_response(len(ops))

    client = sync_client(handler)
    result = client.context("sake").add_associations_batched(
        [_op(i) for i in range(5)], chunk_size=2
    )
    assert result.applied == 5
    assert result.chunks == 3
    assert batches == [2, 2, 1]
