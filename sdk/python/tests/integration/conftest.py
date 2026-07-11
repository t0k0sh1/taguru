"""Integration harness: spawn the real server binary, mirror tests/http_api.rs.

The spawn/binary helpers live in taguru.testing so the langchain-taguru test
suite (and applications) can reuse them.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from taguru import Taguru
from taguru.testing import SpawnedServer, default_binary

REPO_ROOT = Path(__file__).resolve().parents[4]

ADMIN_TOKEN = "test-admin-token"
READER_TOKEN = "test-reader-token"


@pytest.fixture(scope="session")
def server_binary() -> Path:
    return default_binary(REPO_ROOT)


@pytest.fixture(scope="session")
def server(
    server_binary: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[SpawnedServer]:
    """The main server: named keys with an admin and a read-scoped one."""
    spawned = SpawnedServer(
        server_binary,
        tmp_path_factory.mktemp("taguru-data"),
        {
            "TAGURU_API_TOKENS": f"admin:{ADMIN_TOKEN},reader:{READER_TOKEN}",
            "TAGURU_KEY_SCOPES": '{"reader": "read"}',
        },
    )
    yield spawned
    spawned.stop()


@pytest.fixture(scope="session")
def client(server: SpawnedServer) -> Iterator[Taguru]:
    with Taguru(server.base_url, ADMIN_TOKEN) as instance:
        instance.wait_until_ready(timeout=30)
        yield instance


@pytest.fixture(scope="session")
def reader_client(server: SpawnedServer) -> Iterator[Taguru]:
    with Taguru(server.base_url, READER_TOKEN) as instance:
        yield instance


@pytest.fixture
def fresh_name() -> str:
    """A collision-free context name per test."""
    return f"t-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def spawn_server(server_binary: Path, tmp_path: Path) -> Iterator[type[SpawnedServer] | object]:
    """Factory for tests needing dedicated server settings (limits etc.)."""
    spawned: list[SpawnedServer] = []

    def factory(extra_env: dict[str, str]) -> SpawnedServer:
        instance = SpawnedServer(server_binary, tmp_path / f"data-{len(spawned)}", extra_env)
        spawned.append(instance)
        return instance

    yield factory
    for instance in spawned:
        instance.stop()
