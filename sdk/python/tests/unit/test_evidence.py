"""``assemble_evidence`` (#216, #305, #306): request construction and
decoding, against an in-memory mock transport. The golden-fixture
decode itself is covered by ``test_wire_contract.py``; this file covers
what that one cannot — argument normalization and omission.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .conftest import ok_response, sync_client

MIXED_LANES_RESULT: dict[str, Any] = {
    "budget": {
        "bytes_used": 1066,
        "items_used": 2,
        "limits": {"max_bytes": 65536, "max_items": 40, "max_tokens": 4000},
        "tokens_used": 284,
    },
    "citations": [
        {
            "citation": {
                "section": None,
                "source": "docs/kura.md",
                "text": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。",
            },
            "paragraph": 0,
            "source": "docs/kura.md",
        }
    ],
    "items": [
        {
            "association": {
                "attributions": [
                    {
                        "count": 1,
                        "paragraph": 0,
                        "section": None,
                        "source": "docs/kura.md",
                        "weight": 1.0,
                    }
                ],
                "count": 1,
                "label": "杜氏",
                "object": "高瀬",
                "subject": "青嶺酒造",
                "weight": 1.0,
            },
            "bytes": 512,
            "candidate_id": "association\u0000青嶺酒造\u0000杜氏\u0000高瀬",
            "citation_refs": [{"paragraph": 0, "source": "docs/kura.md"}],
            "corroboration": {
                "attributions": [{"paragraph": 0, "source": "docs/kura.md"}],
                "sources": ["docs/kura.md"],
            },
            "estimated_tokens": 132,
            "fused_rank": 1,
            "kind": "association",
            "lane_ranks": [{"lane": "graph_activate", "rank": 1}],
        },
        {
            "bytes": 367,
            "candidate_id": "passage\u0000sake\u0000docs/kura.md\u00000",
            "citation_refs": [],
            "estimated_tokens": 99,
            "fused_rank": 2,
            "kind": "passage",
            "lane_ranks": [{"lane": "passage_bm25", "rank": 1}],
            "passage": {
                "lanes": {"bm25": {"rank": 1, "score": 0.86304635}},
                "paragraph": 0,
                "score": 0.86304635,
                "source": "docs/kura.md",
                "text": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。",
            },
        },
    ],
    "omitted": [],
    "omitted_by_reason": {},
    "omitted_total": 0,
    "plan": {
        "lanes": {
            "activate": {"ran": True},
            "citations": {"ran": True},
            "communities": {"ran": False, "reason": "include_communities was false"},
            "passages": {"ran": True},
            "query": {"ran": False, "reason": "no 'labels' given"},
            "resolve": {"ran": True},
        },
        "reranker": {"configured": False, "ran": False},
        "selection": {"contradiction_groups": 0, "dedup_dropped": 0, "diversity_tier_width": 10},
    },
}


def test_assemble_evidence_posts_to_the_evidence_route_with_a_bare_origins_call() -> None:
    calls: list[tuple[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.path, json.loads(req.content) if req.content else None))
        return ok_response(MIXED_LANES_RESULT)

    client = sync_client(handler, retries=0)
    package = client.context("sake").assemble_evidence("青嶺酒造")

    assert calls == [("/contexts/sake/evidence", {"origins": ["青嶺酒造"]})]
    assert len(package.items) == 2
    assert package.items[0].kind == "association"
    assert package.items[0].association is not None
    assert package.items[0].corroboration is not None
    assert package.items[0].corroboration.sources == ["docs/kura.md"]
    assert package.items[0].contradicts == []
    assert package.items[1].kind == "passage"
    assert package.items[1].passage is not None
    assert package.items[1].corroboration is None
    assert package.budget.limits.max_items == 40
    assert package.omitted == []
    assert package.omitted_by_reason == {}
    assert package.plan.lanes.communities.ran is False
    assert package.plan.lanes.communities.reason == "include_communities was false"
    assert package.plan.reranker.configured is False


def test_assemble_evidence_normalizes_a_single_origin_string() -> None:
    calls: list[tuple[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.path, json.loads(req.content) if req.content else None))
        return ok_response(MIXED_LANES_RESULT)

    client = sync_client(handler, retries=0)
    client.context("sake").assemble_evidence("青嶺酒造", labels="杜氏")

    _, body = calls[0]
    assert body["origins"] == ["青嶺酒造"]
    assert body["labels"] == ["杜氏"]


def test_assemble_evidence_omits_unset_options_from_the_body() -> None:
    """No field the caller didn't pass should ever reach the wire —
    including `include_communities`, whose server-side default is
    `false` but which must still be *omitted*, not sent as `false`,
    when the caller never named it."""
    calls: list[tuple[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.path, json.loads(req.content) if req.content else None))
        return ok_response(MIXED_LANES_RESULT)

    client = sync_client(handler, retries=0)
    client.context("sake").assemble_evidence(["青嶺酒造", "高瀬"])

    _, body = calls[0]
    assert body == {"origins": ["青嶺酒造", "高瀬"]}


def test_assemble_evidence_passes_budget_and_rerank_through_as_nested_objects() -> None:
    calls: list[tuple[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.path, json.loads(req.content) if req.content else None))
        return ok_response(MIXED_LANES_RESULT)

    client = sync_client(handler, retries=0)
    client.context("sake").assemble_evidence(
        "青嶺酒造",
        include_communities=True,
        budget={"max_items": 10, "max_bytes": 8192},
        rerank={"model": "bge-reranker"},
    )

    _, body = calls[0]
    assert body["include_communities"] is True
    assert body["budget"] == {"max_items": 10, "max_bytes": 8192}
    assert body["rerank"] == {"model": "bge-reranker"}


def test_assemble_evidence_decodes_a_minimal_response_with_every_optional_field_omitted() -> None:
    """Every `skip_serializing_if`-guarded field on the wire (a
    corroboration-less, contradiction-less, duplicate_of-less,
    reranker-reason-less package) must decode without a
    ``ResponseShapeError`` — the server never emits `null` for these,
    it omits the key entirely."""
    minimal_result: dict[str, Any] = {
        "budget": {
            "bytes_used": 10,
            "items_used": 1,
            "limits": {"max_bytes": 65536, "max_items": 40, "max_tokens": 4000},
            "tokens_used": 3,
        },
        "citations": [],
        "items": [
            {
                "bytes": 10,
                "candidate_id": "passage\u0000s\u00000",
                "citation_refs": [],
                "estimated_tokens": 3,
                "fused_rank": 1,
                "kind": "passage",
                "lane_ranks": [{"lane": "passage_bm25", "rank": 1}],
                "passage": {
                    "lanes": {"bm25": {"rank": 1, "score": 1.0}},
                    "paragraph": 0,
                    "score": 1.0,
                    "source": "s",
                    "text": "hello",
                },
                # no "corroboration", no "contradicts" — both skipped on the wire.
            }
        ],
        "omitted": [],
        "omitted_by_reason": {},
        "omitted_total": 0,
        "plan": {
            "lanes": {
                "activate": {"ran": True},
                "citations": {"ran": True},
                "communities": {"ran": False, "reason": "include_communities was false"},
                "passages": {"ran": True},
                "query": {"ran": False, "reason": "no 'labels' given"},
                "resolve": {"ran": True},
            },
            "reranker": {"configured": False, "ran": False},
            # no "model", no "reason" on reranker — both skipped.
            "selection": {
                "contradiction_groups": 0,
                "dedup_dropped": 0,
                "diversity_tier_width": 10,
            },
        },
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return ok_response(minimal_result)

    client = sync_client(handler, retries=0)
    package = client.context("sake").assemble_evidence("青嶺酒造")

    assert package.items[0].corroboration is None
    assert package.items[0].contradicts == []
    assert package.items[0].association is None
    assert package.items[0].community is None
    assert package.plan.reranker.model is None
    assert package.plan.reranker.reason is None
