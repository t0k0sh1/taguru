"""Unit-test helpers: clients wired to an in-memory httpx.MockTransport."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from taguru import AsyncTaguru, Taguru

Handler = Callable[[httpx.Request], httpx.Response]


def ok_response(result: object) -> httpx.Response:
    return httpx.Response(200, json={"result": result, "status": "ok", "time": 0.001})


def err_response(
    status: int,
    message: str,
    headers: dict[str, str] | None = None,
    code: str | None = None,
) -> httpx.Response:
    payload: dict[str, object] = {"status": "error", "error": message, "time": 0.001}
    if code is not None:
        payload["code"] = code
    return httpx.Response(status, json=payload, headers=headers or {})


def sync_client(handler: Handler, **kwargs: object) -> Taguru:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return Taguru("http://test", http_client=http, **kwargs)  # type: ignore[arg-type]


def async_client(handler: Handler, **kwargs: object) -> AsyncTaguru:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncTaguru("http://test", http_client=http, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out computed backoff so retry tests run instantly.

    Retry-After waits still happen (tests send "0")."""
    monkeypatch.setattr("taguru._sync.client.backoff_delay", lambda attempt: 0.0)
    monkeypatch.setattr("taguru._async.client.backoff_delay", lambda attempt: 0.0)
