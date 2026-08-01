"""Unit-test helpers: a routed in-memory Taguru server via httpx.MockTransport."""

from __future__ import annotations

import json
import re
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

GROUP_ROWS = {
    "brewery": {
        "name": "brewery",
        "description": "蔵元一式",
        "contexts": ["sake", "tea"],
        "groups": [],
    },
    "parent": {"name": "parent", "description": "", "contexts": ["sake"], "groups": ["childg"]},
    "childg": {"name": "childg", "description": "", "contexts": ["tea"], "groups": []},
}


def ok(result: object) -> httpx.Response:
    return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.001})


class FakeServer:
    """Answers the retriever/ingester call surface; records every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.imported: list[str] = []
        # Context names whose every request should fail with a 500, to
        # exercise cross-context partial-failure handling.
        self.fail_contexts: set[str] = set()
        # When set, the cross-context /sources/search call (the text
        # lane) fails with a 500, to exercise text/graph lane isolation.
        self.fail_text_search = False
        # 200 = refreshed; 501 = no embedding provider configured (default);
        # 502 = the provider itself failed. The 501/502 pair both map to
        # EmbeddingUnavailableError, distinguished by its `reason`.
        self.embeddings_refresh_status = 501
        self.embeddings_refresh_result: dict[str, Any] = {"embedded": 0, "total": 0}
        # When set, /import fails with a 500 — exercises the "checkpoints
        # survive an import failure" case (issue #211).
        self.fail_import = False
        # Overrides the one canned ImportOutcome /import always answers
        # with below — set per test to assert a specific server-reported
        # field (e.g. sections_stored/locators_dropped, issue #347) copies
        # through TaguruIngester._record into IngestOutcome correctly.
        self.import_result_override: dict[str, Any] | None = None
        # The context's current source list, and every source
        # `Context.retract_source` has been asked to withdraw (in call
        # order) — backs `sync_object_storage`'s own `retract`/`mirror`
        # deletion-policy tests (issue #351): seed `sources` to simulate
        # what the server already holds, then assert `retracted`.
        self.sources: list[str] = []
        self.retracted: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/version":
            # The client's one-time compatibility preflight (ADR 0005
            # §3.8, §6) — routed ABOVE `self.calls.append` below and
            # not itself recorded, so it doesn't pollute the exact
            # call-list assertions the rest of this suite makes, and
            # doesn't need every existing test to route a path it
            # never used to send.
            return httpx.Response(
                200,
                json={"server": "0.6.0", "http_contract": {"current": 1, "supported": [1]}},
            )
        body: Any = None
        if request.content:
            try:
                body = json.loads(request.content)
            except json.JSONDecodeError:
                body = request.content.decode("utf-8")
        self.calls.append((path, body))
        context_match = re.match(r"^/contexts/([^/]+)/", path)
        if context_match and context_match.group(1) in self.fail_contexts:
            return httpx.Response(
                500,
                json={
                    "status": "error",
                    "code": "internal",
                    "error": "simulated failure",
                    "time": 0.001,
                },
            )
        if path.startswith("/groups/"):
            row = GROUP_ROWS.get(path.removeprefix("/groups/"))
            if row is None:
                return httpx.Response(
                    404,
                    json={
                        "status": "error",
                        "code": "no_group",
                        "error": "group not found",
                        "time": 0.001,
                    },
                )
            return ok(row)
        if path == "/sources/search":
            if self.fail_text_search:
                return httpx.Response(
                    500,
                    json={
                        "status": "error",
                        "code": "internal",
                        "error": "simulated failure",
                        "time": 0.001,
                    },
                )
            # The cross-context search: one tagged hit per named context,
            # already rank-interleaved the way the server merges, the
            # plan listing every target (#151).
            return ok(
                {
                    "plan": {
                        "contexts": [
                            {
                                "context": name,
                                "lanes": {
                                    "bm25": {"ran": True},
                                    "vector": {
                                        "ran": False,
                                        "reason": "no embedding provider is configured",
                                    },
                                },
                            }
                            for name in body["contexts"]
                        ]
                    },
                    "hits": [
                        {
                            "context": name,
                            "source": f"docs/{name}.md",
                            "paragraph": 0,
                            "score": 2.0,
                            "text": f"{name} の段落。",
                            "lanes": {"bm25": {"rank": 0, "score": 2.0}},
                        }
                        for name in body["contexts"]
                    ],
                }
            )
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
                {
                    "plan": {
                        "contexts": [
                            {
                                "context": "sake",
                                "lanes": {
                                    "bm25": {"ran": True},
                                    "vector": {
                                        "ran": False,
                                        "reason": "no embedding provider is configured",
                                    },
                                },
                            }
                        ]
                    },
                    "hits": [
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
                    ],
                }
            )
        if path.endswith("/labels"):
            return ok({"total": 2, "labels": ["代表銘柄", "杜氏"]})
        if path.endswith("/sources/retract"):
            source = body["source"]
            if source in self.sources:
                self.sources.remove(source)
            self.retracted.append(source)
            return ok({"associations_touched": 0, "passage_removed": True})
        if path.endswith("/sources") and request.method == "GET":
            params = request.url.params
            prefix = params.get("prefix")
            after = params.get("after")
            limit_raw = params.get("limit")
            limit = int(limit_raw) if limit_raw is not None else None
            candidates = sorted(s for s in self.sources if prefix is None or s.startswith(prefix))
            # `total` is the prefix-filtered count, BEFORE `after`/`limit`
            # slice it down for this one page — matching the real server's
            # own SourcePage contract (src/api/sources.rs).
            total = len(candidates)
            if after is not None:
                candidates = [s for s in candidates if s > after]
            if limit is not None:
                candidates = candidates[:limit]
            return ok(
                {
                    "total": total,
                    "sources": candidates,
                    "entries": [{"name": source} for source in candidates],
                }
            )
        if path == "/import":
            if self.fail_import:
                return httpx.Response(
                    500,
                    json={
                        "status": "error",
                        "code": "internal",
                        "error": "simulated import failure",
                        "time": 0.001,
                    },
                )
            self.imported.append(body if isinstance(body, str) else "")
            batch_result = self.import_result_override or {
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
                "locators_stored": 0,
                "locators_dropped": 0,
                "association_paragraphs_dropped": 0,
            }
            return ok({"batches": [batch_result]})
        if path.endswith("/embeddings/refresh"):
            if self.embeddings_refresh_status == 200:
                return ok(self.embeddings_refresh_result)
            message = "no provider" if self.embeddings_refresh_status == 501 else "provider failed"
            return httpx.Response(
                self.embeddings_refresh_status,
                json={"status": "error", "error": message, "time": 0.001},
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
