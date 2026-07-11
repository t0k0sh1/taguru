"""Unit-test helpers: a routed in-memory Taguru server via httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from taguru import AsyncTaguru, Taguru

AOMINE_ASSOCIATION = {
    "subject": "青嶺酒造",
    "label": "杜氏",
    "object": "高瀬",
    "weight": 1.0,
    "count": 1,
    "attributions": [
        {"source": "docs/aomine.md", "weight": 1.0, "count": 1, "paragraph": 1, "section": None}
    ],
}

FACT_ONLY_ASSOCIATION = {
    "subject": "青嶺酒造",
    "label": "創業年",
    "object": "1907年",
    "weight": 1.0,
    "count": 1,
    "attributions": [
        {"source": "口伝", "weight": 1.0, "count": 1, "paragraph": None, "section": None}
    ],
}


def ok(result: object) -> httpx.Response:
    return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.001})


class FakeServer:
    """Answers the retriever/ingester call surface; records every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.imported: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body: Any = None
        if request.content:
            try:
                body = json.loads(request.content)
            except json.JSONDecodeError:
                body = request.content.decode("utf-8")
        self.calls.append((path, body))
        if path.endswith("/resolve"):
            hits = (
                [{"name": "青嶺酒造", "score": 1.0, "tier": "lexical", "kind": "exact"}]
                if "青嶺" in body["cue"]
                else []
            )
            return ok(hits)
        if path.endswith("/activate"):
            return ok(
                {
                    "total": 2,
                    "matches": [
                        {"strength": 0.9, "path": ["青嶺酒造"], "association": AOMINE_ASSOCIATION},
                        {
                            "strength": 0.5,
                            "path": ["青嶺酒造"],
                            "association": FACT_ONLY_ASSOCIATION,
                        },
                    ],
                }
            )
        if path.endswith("/citations"):
            return ok({"text": "杜氏は高瀬である。", "source": body["source"], "section": "人物"})
        if path.endswith("/sources/search"):
            return ok(
                [
                    {
                        "source": "docs/aomine.md",
                        "paragraph": 1,
                        "score": 3.0,
                        "text": "杜氏は高瀬である。",
                        "lanes": {"bm25": {"rank": 0, "score": 3.0}},
                    },
                    {
                        "source": "docs/other.md",
                        "paragraph": 0,
                        "score": 1.0,
                        "text": "別の文書の段落。",
                        "lanes": {"bm25": {"rank": 1, "score": 1.0}},
                    },
                ]
            )
        if path.endswith("/labels"):
            return ok({"total": 2, "labels": ["代表銘柄", "杜氏"]})
        if path == "/import":
            self.imported.append(body if isinstance(body, str) else "")
            return ok(
                {
                    "batches": [
                        {
                            "context": "sake",
                            "source": "docs/aomine.md",
                            "created": False,
                            "retracted": 0,
                            "associations": 2,
                            "aliases": 1,
                            "passage_stored": True,
                            "passage_dropped": False,
                            "questions_stored": 1,
                            "questions_dropped": 0,
                            "sections_stored": 0,
                            "sections_dropped": 0,
                            "association_paragraphs_dropped": 0,
                        }
                    ]
                }
            )
        if path.endswith("/embeddings/refresh"):
            return httpx.Response(
                501, json={"status": "error", "error": "no provider", "time": 0.001}
            )
        raise AssertionError(f"unrouted path: {path}")


@pytest.fixture
def fake_server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def sync_client(fake_server: FakeServer) -> Taguru:
    transport = httpx.MockTransport(fake_server.handler)
    return Taguru("http://test", api_key="", http_client=httpx.Client(transport=transport))


@pytest.fixture
def async_client(fake_server: FakeServer) -> AsyncTaguru:
    transport = httpx.MockTransport(fake_server.handler)
    return AsyncTaguru(
        "http://test", api_key="", http_client=httpx.AsyncClient(transport=transport)
    )
