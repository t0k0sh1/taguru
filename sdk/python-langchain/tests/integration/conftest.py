"""Integration harness: real server via taguru.testing, seeded once."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from taguru import Taguru
from taguru.testing import SpawnedServer, default_binary

REPO_ROOT = Path(__file__).resolve().parents[4]

TOKEN = "lc-test-token"
SEEDED_CONTEXT = "sake"

AOMINE_DOC = """青嶺酒造は1907年創業の架空の酒蔵である。代表銘柄は「青嶺」。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。"""


@pytest.fixture(scope="session")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpawnedServer]:
    spawned = SpawnedServer(
        default_binary(REPO_ROOT),
        tmp_path_factory.mktemp("taguru-lc-data"),
        {"TAGURU_API_TOKEN": TOKEN},
    )
    # The standard suite constructs retrievers with no explicit connection —
    # they pick these up, the same way applications default.
    os.environ["TAGURU_URL"] = spawned.base_url
    os.environ["TAGURU_API_TOKEN"] = TOKEN
    yield spawned
    os.environ.pop("TAGURU_URL", None)
    os.environ.pop("TAGURU_API_TOKEN", None)
    spawned.stop()


@pytest.fixture(scope="session")
def client(server: SpawnedServer) -> Iterator[Taguru]:
    with Taguru(server.base_url, TOKEN) as instance:
        instance.wait_until_ready(timeout=30)
        yield instance


@pytest.fixture(scope="session")
def seeded(client: Taguru) -> str:
    """The 青嶺酒造 fixture context, created once per session."""
    client.contexts.create(SEEDED_CONTEXT, description="青嶺酒造という架空の酒蔵の知識")
    ctx = client.context(SEEDED_CONTEXT)
    ctx.add_associations(
        [
            {
                "subject": "青嶺酒造",
                "label": "創業年",
                "object": "1907年",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 0,
            },
            {
                "subject": "青嶺酒造",
                "label": "代表銘柄",
                "object": "青嶺",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 0,
            },
            {
                "subject": "青嶺酒造",
                "label": "杜氏",
                "object": "高瀬",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 1,
            },
            {
                "subject": "高瀬",
                "label": "重視する",
                "object": "寒仕込み",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 1,
            },
            {
                "subject": "青嶺酒造",
                "label": "行う",
                "object": "大量生産",
                "weight": -1.0,
                "source": "docs/aomine.md",
                "paragraph": 2,
            },
        ]
    )
    ctx.store_passages({"docs/aomine.md": AOMINE_DOC})
    return SEEDED_CONTEXT
