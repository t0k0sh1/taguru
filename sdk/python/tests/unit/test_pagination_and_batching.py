"""Auto-pagination iterators and association batching."""

from __future__ import annotations

import json

import httpx

from taguru import AliasEntry
from taguru._shared import batch_associations, dumps_compact
from taguru._types import AssocOp

from .conftest import ok_response, sync_client


def test_contexts_iter_walks_pages_with_keyset_cursor() -> None:
    rows = [
        {
            "id": f"id-{name}",
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
                "dead_edges": 0,
                "dead_attributions": 0,
                "arena_slack": 0,
                "unsourced_edges": 0,
                "unsourced_weight": 0.0,
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
    cursors: list[tuple[str | None, str | None]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        after = req.url.params.get("after")
        after_id = req.url.params.get("after_id")
        cursors.append((after, after_id))
        if after is None:
            return ok_response({"total": 3, "contexts": rows[:2]})
        if after == "b":
            return ok_response({"total": 3, "contexts": rows[2:]})
        if after == "c":
            return ok_response({"total": 3, "contexts": []})
        raise AssertionError(after)

    client = sync_client(handler)
    names = [entry.name for entry in client.contexts.iter(limit=2)]
    assert names == ["a", "b", "c"]
    # A short page is not the last one — a concurrent delete could shorten it
    # while later rows remain — so iteration pages on past the short final page
    # and stops only on the empty page after it. The cursor carries the
    # (name, id) pair — names are not unique (#964).
    assert cursors == [(None, None), ("b", "id-b"), ("c", "id-c")]


def test_iter_labels_pages_until_an_empty_page() -> None:
    cursors: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        after = req.url.params.get("after")
        cursors.append(after)
        if after is None:
            return ok_response({"total": 1, "labels": ["代表銘柄"]})
        if after == "代表銘柄":
            return ok_response({"total": 1, "labels": []})
        raise AssertionError(after)

    client = sync_client(handler)
    assert list(client.context("sake").iter_labels(limit=10)) == ["代表銘柄"]
    # A short page (1 < limit 10) is not the end: iteration continues to the
    # empty page after it rather than stopping early.
    assert cursors == [None, "代表銘柄"]


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
        if after == "label:brand":
            return ok_response({"total": 3, "concepts": {}, "labels": {}})
        raise AssertionError(after)

    client = sync_client(handler)
    entries = list(client.context("sake").iter_aliases(limit=2))
    assert entries == [
        AliasEntry(namespace="concept", alias="Aomine", canonical="青嶺酒造"),
        AliasEntry(namespace="concept", alias="青嶺", canonical="青嶺酒造"),
        AliasEntry(namespace="label", alias="brand", canonical="代表銘柄"),
    ]
    # The short second page (1 < limit 2) is not the end: iteration pages on
    # to the empty page after it, advancing the cursor to the last alias.
    assert cursors == [None, "concept:青嶺", "label:brand"]


def _op(i: int) -> AssocOp:
    return {"subject": f"s{i}", "label": "l", "object": "o", "weight": 1.0}


def test_batch_associations_splits_by_count() -> None:
    ops = [_op(i) for i in range(5)]
    batches = list(batch_associations(ops, 2, 10**9))
    assert [len(b) for b in batches] == [2, 2, 1]


def test_batch_associations_splits_by_bytes_matching_wire_serialization() -> None:
    ops = [_op(i) for i in range(4)]
    one = len(dumps_compact(_op(0)))
    # Room for exactly two ops per batch: [op,op] = 2 + one + 1 + one bytes.
    budget = 2 + one + 1 + one
    batches = list(batch_associations(ops, 10_000, budget))
    assert [len(b) for b in batches] == [2, 2]
    for batch in batches:
        assert len(dumps_compact(batch)) <= budget


def test_batch_associations_yields_oversized_single_op_alone() -> None:
    ops = [_op(0), _op(1)]
    batches = list(batch_associations(ops, 10_000, 3))
    assert [len(b) for b in batches] == [1, 1]


def test_batch_associations_charges_the_separating_comma_exactly() -> None:
    """One byte under the exact two-op serialization must split — the comma
    between elements is real wire bytes, and undercounting it by one would
    ship batches that overshoot ``max_batch_bytes``."""
    ops = [_op(0), _op(1)]
    one = len(dumps_compact(_op(0)))
    exactly_two = 2 + one + 1 + one
    assert [len(b) for b in batch_associations(ops, 10_000, exactly_two)] == [2]
    assert [len(b) for b in batch_associations(ops, 10_000, exactly_two - 1)] == [1, 1]


def test_add_associations_batched_sums_applied_counts() -> None:
    batches: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ops = json.loads(req.content)
        batches.append(len(ops))
        return ok_response(len(ops))

    client = sync_client(handler)
    result = client.context("sake").add_associations_batched(
        [_op(i) for i in range(5)], batch_size=2
    )
    assert result.applied == 5
    assert result.batches == 3
    assert result.issues == []
    assert result.schema_violations == 0
    assert batches == [2, 2, 1]


def test_add_associations_surfaces_the_warn_mode_envelope_carrier() -> None:
    """ADR 0009 §8.3: a ``warn``-mode write's violations ride the success
    envelope beside ``result`` — the SDK must hand them to the caller, not
    strip them with the envelope."""
    issue = {
        "path": "associations[0].object",
        "kind": "range",
        "expected": "one of [Brewery]",
        "actual": "Prefecture",
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": 1,
                "status": "ok",
                "time": 0.001,
                "issues": [issue],
                "schema_violations": 3,
            },
        )

    client = sync_client(handler)
    outcome = client.context("sake").add_associations([_op(0)])
    assert outcome.applied == 1
    assert outcome.schema_violations == 3
    assert [(i.path, i.kind, i.expected, i.actual) for i in outcome.issues] == [
        (issue["path"], issue["kind"], issue["expected"], issue["actual"])
    ]

    # Batched aggregation preserves batch order and sums the true counts.
    # Each response is derived from the BATCH THE REQUEST CARRIED, not
    # from handler call order — so a reordered transmission, a dropped
    # batch, or a double-count would all be visible.
    violations_for = {"s0": 1, "s1": 2}

    def batched_handler(req: httpx.Request) -> httpx.Response:
        ops = json.loads(req.content)
        subject = ops[0]["subject"]
        return httpx.Response(
            200,
            json={
                "result": len(ops),
                "status": "ok",
                "time": 0.001,
                "issues": [dict(issue, path=f"associations[0].{subject}")],
                "schema_violations": violations_for[subject],
            },
        )

    batched_client = sync_client(batched_handler)
    batched = batched_client.context("sake").add_associations_batched(
        [_op(0), _op(1)], batch_size=1
    )
    assert batched.applied == 2
    assert batched.schema_violations == 1 + 2
    assert [i.path for i in batched.issues] == [
        "associations[0].s0",
        "associations[0].s1",
    ]
