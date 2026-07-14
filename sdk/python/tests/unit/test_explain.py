"""The resolve/search explain endpoints: request shape and typed decode."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from taguru import ResolveExplanation, SearchExplanation

from .conftest import async_client, ok_response, sync_client

RESOLVE_EXPLANATION: dict[str, Any] = {
    "verdict": "below_floor",
    "summary": "青嶺酒造 scored 0.42, below the 0.6 floor in effect.",
    "cue": "青嶺",
    "expected": "青嶺酒造",
    "in_vocabulary": True,
    "canonical": "青嶺酒造",
    "expected_kind": "exact",
    "lexical": {"score": 0.42, "kind": "containment", "floor": 0.6, "confident": False},
    "semantic": {"entered": False, "reason": "lexical tier not confident"},
    "ranking": {
        "rank": 3,
        "tier": "lexical",
        "score": 0.42,
        "limit": 5,
        "served": False,
        "limit_to_reach": 3,
    },
}

SEARCH_EXPLANATION: dict[str, Any] = {
    "verdict": "no_term_overlap",
    "summary": "query says 酒造, the paragraph spells 酒蔵.",
    "source": "docs/aomine.md",
    "paragraph": 1,
    "paragraphs": 3,
    "paragraph_named": True,
    "query_terms": ["酒造"],
    "paragraph_terms": ["酒蔵"],
    "bm25": {
        "score": 0.0,
        "terms": [{"term": "酒造", "tf": 0.0, "df": 5, "idf": 1.2, "contribution": 0.0}],
    },
    "vector": {"ran": False, "reason": "no embedding provider configured"},
    "ranking": {"fused": False, "ranked": 0, "limit": 5, "served": False},
}


def capturing_handler(payload: Any, calls: list[tuple[str, Any]]) -> Any:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.url.path, json.loads(req.content) if req.content else None))
        return ok_response(payload)

    return handler


def test_explain_resolve_posts_cue_and_expected_and_decodes() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(capturing_handler(RESOLVE_EXPLANATION, calls))
    verdict = client.context("aomine").explain_resolve("青嶺", "青嶺酒造", dice_floor=0.6)

    path, body = calls[0]
    assert path == "/contexts/aomine/resolve/explain"
    assert body == {"cue": "青嶺", "expected": "青嶺酒造", "dice_floor": 0.6}

    assert isinstance(verdict, ResolveExplanation)
    assert verdict.verdict == "below_floor"
    assert verdict.in_vocabulary is True
    # Nested optionals decode into their own frozen models.
    assert verdict.lexical is not None
    assert verdict.lexical.score == 0.42
    assert verdict.lexical.confident is False
    assert verdict.semantic is not None
    assert verdict.semantic.entered is False
    assert verdict.ranking is not None
    assert verdict.ranking.limit_to_reach == 3


def test_explain_resolve_label_routes_to_the_label_endpoint() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(capturing_handler(RESOLVE_EXPLANATION, calls))
    verdict = client.context("aomine").explain_resolve_label("杜氏", "杜氏長")

    path, body = calls[0]
    assert path == "/contexts/aomine/resolve_label/explain"
    # No overrides passed → only the two required args ride the body.
    assert body == {"cue": "杜氏", "expected": "杜氏長"}
    assert isinstance(verdict, ResolveExplanation)


def test_explain_search_passages_posts_query_source_and_decodes() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(capturing_handler(SEARCH_EXPLANATION, calls))
    verdict = client.context("aomine").explain_search_passages(
        "青嶺酒造の酒造", "docs/aomine.md", paragraph=1
    )

    path, body = calls[0]
    assert path == "/contexts/aomine/sources/search/explain"
    assert body == {"query": "青嶺酒造の酒造", "source": "docs/aomine.md", "paragraph": 1}

    assert isinstance(verdict, SearchExplanation)
    assert verdict.verdict == "no_term_overlap"
    assert verdict.query_terms == ["酒造"]
    assert verdict.bm25 is not None
    assert verdict.bm25.terms[0].df == 5
    assert verdict.vector is not None
    assert verdict.vector.ran is False


async def test_explain_resolve_parity_on_the_async_client() -> None:
    calls: list[tuple[str, Any]] = []
    client = async_client(capturing_handler(RESOLVE_EXPLANATION, calls))
    try:
        verdict = await client.context("aomine").explain_resolve("青嶺", "青嶺酒造")
    finally:
        await client.close()
    assert calls[0][0] == "/contexts/aomine/resolve/explain"
    assert isinstance(verdict, ResolveExplanation)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
