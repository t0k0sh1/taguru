"""The high-level retrieve() loop, composed against a routed mock server."""

from __future__ import annotations

import json
from typing import Any

import httpx

from taguru import citation_key

from .conftest import err_response, ok_response, sync_client

ASSOCIATION = {
    "subject": "青嶺酒造",
    "label": "杜氏",
    "object": "高瀬",
    "weight": 2.0,
    "count": 2,
    "attributions": [
        {"source": "docs/aomine.md", "weight": 2.0, "count": 2, "paragraph": 1, "section": None},
        {"source": "unstored.md", "weight": 1.0, "count": 1, "paragraph": 0, "section": None},
    ],
}


def routed_handler(calls: list[tuple[str, Any]]) -> Any:
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        body = json.loads(req.content) if req.content else None
        calls.append((path, body))
        if path.endswith("/resolve"):
            return ok_response(
                [{"name": "青嶺酒造", "score": 1.0, "tier": "lexical", "kind": "exact"}]
                if body["cue"] == "青嶺"
                else []
            )
        if path.endswith("/describe"):
            return ok_response(
                {
                    "concept": "青嶺酒造",
                    "as_subject": [{"label": "杜氏", "count": 1}],
                    "as_object": [],
                }
            )
        if path.endswith("/activate"):
            return ok_response(
                {
                    "total": 1,
                    "matches": [
                        {"strength": 0.9, "path": ["青嶺酒造"], "association": ASSOCIATION}
                    ],
                }
            )
        if path.endswith("/query"):
            return ok_response({"total": 1, "matches": [ASSOCIATION]})
        if path.endswith("/citations"):
            if body["source"] == "unstored.md":
                return err_response(404, "source 'unstored.md' has no stored passage")
            return ok_response(
                {"text": "杜氏は高瀬。", "source": body["source"], "section": "人物"}
            )
        if path.endswith("/sources/search"):
            return ok_response(
                [
                    {
                        "source": "docs/aomine.md",
                        "paragraph": 1,
                        "score": 3.2,
                        "text": "杜氏は高瀬。",
                        "lanes": {"bm25": {"rank": 0, "score": 3.2}},
                    }
                ]
            )
        raise AssertionError(path)

    return handler


def test_retrieve_runs_the_documented_loop() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(routed_handler(calls), retries=0)
    result = client.context("sake").retrieve("青嶺")

    assert [c.name for c in result.resolved["青嶺"]] == ["青嶺酒造"]
    assert result.outline["青嶺酒造"] is not None
    assert len(result.associations) == 1
    assert result.activations[0].strength == 0.9
    # The located citation was fetched; the unstored one was skipped, not fatal.
    assert result.citations[citation_key("docs/aomine.md", 1)].section == "人物"
    assert citation_key("unstored.md", 0) not in result.citations
    # Graph answered, so the text lane did not run by default.
    assert result.passage_hits == []
    assert [p for p, _ in calls] == [
        "/contexts/sake/resolve",
        "/contexts/sake/describe",
        "/contexts/sake/activate",
        "/contexts/sake/citations",
        "/contexts/sake/citations",
    ]


def test_retrieve_text_fallback_fires_only_when_graph_is_empty() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(routed_handler(calls), retries=0)
    ctx = client.context("sake")

    # Graph lane non-empty → fallback stays off even when a query is given.
    result = ctx.retrieve("青嶺", text_fallback_query="杜氏は高瀬である")
    assert result.passage_hits == []

    # Unresolvable cue → no anchors → empty graph → fallback fires.
    result = ctx.retrieve("無関係", text_fallback_query="杜氏は高瀬である")
    assert result.associations == []
    assert len(result.passage_hits) == 1
    assert result.passage_hits[0].lanes.bm25 is not None
    assert result.passage_hits[0].lanes.vector is None

    # only_if_empty=False → always runs.
    result = ctx.retrieve(
        "青嶺", text_fallback_query="杜氏は高瀬である", text_fallback_only_if_empty=False
    )
    assert len(result.passage_hits) == 1


def test_retrieve_auto_pick_false_uses_cues_verbatim() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve("青嶺", auto_pick=False, describe_first=False)
    activate_call = next(body for path, body in calls if path.endswith("/activate"))
    assert activate_call["origins"] == ["青嶺"]


def test_retrieve_labels_pins_query_facets() -> None:
    calls: list[tuple[str, Any]] = []
    client = sync_client(routed_handler(calls), retries=0)
    result = client.context("sake").retrieve(
        "青嶺", labels=["杜氏"], describe_first=False, fetch_citations=False
    )
    query_call = next(body for path, body in calls if path.endswith("/query"))
    assert query_call == {"subject": ["青嶺酒造"], "label": ["杜氏"]}
    # query + activate both returned the same triple — deduplicated.
    assert len(result.associations) == 1
