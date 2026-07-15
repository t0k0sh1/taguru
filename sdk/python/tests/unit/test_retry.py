"""Retry policy: idempotency-first classification."""

from __future__ import annotations

import httpx
import pytest

from taguru import RateLimitError, ServerError, TransportError
from taguru._retry import parse_retry_after

from .conftest import err_response, ok_response, sync_client

EMPTY_MATCHES = {"total": 0, "matches": []}


class FlakyHandler:
    """Fails ``failures`` times, then succeeds with ``success`` as the result."""

    def __init__(self, failures: int, make_failure: object, success: object = 0) -> None:
        self.failures = failures
        self.make_failure = make_failure
        self.success = success
        self.calls = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.failures:
            result = self.make_failure  # type: ignore[assignment]
            if isinstance(result, Exception):
                raise result
            assert callable(result)
            failure = result()
            if isinstance(failure, Exception):
                raise failure
            return failure  # type: ignore[return-value]
        return ok_response(self.success)


def test_429_retries_even_on_unsafe_write_route() -> None:
    """Rate limiting rejects before the handler runs — nothing was applied."""
    handler = FlakyHandler(1, lambda: err_response(429, "budget", {"retry-after": "0"}))
    client = sync_client(handler)
    applied = client.context("sake").add_associations(
        [{"subject": "s", "label": "l", "object": "o", "weight": 1.0}]
    )
    assert applied == 0
    assert handler.calls == 2


def test_503_retries_and_honors_retry_after() -> None:
    handler = FlakyHandler(
        1, lambda: err_response(503, "shed", {"retry-after": "0"}), success=EMPTY_MATCHES
    )
    client = sync_client(handler)
    client.context("sake").recall("cue")
    assert handler.calls == 2


def test_500_is_not_retried() -> None:
    handler = FlakyHandler(5, lambda: err_response(500, "io"))
    client = sync_client(handler)
    with pytest.raises(ServerError):
        client.context("sake").recall("cue")
    assert handler.calls == 1


def test_502_retries_on_safe_read() -> None:
    handler = FlakyHandler(1, lambda: err_response(502, "provider"), success=EMPTY_MATCHES)
    client = sync_client(handler)
    client.context("sake").recall("cue")
    assert handler.calls == 2


def test_connect_failure_retries_even_on_unsafe_route() -> None:
    """A pre-connect failure never reached the server."""
    handler = FlakyHandler(1, lambda: httpx.ConnectError("refused"))
    client = sync_client(handler)
    client.context("sake").add_associations(
        [{"subject": "s", "label": "l", "object": "o", "weight": 1.0}]
    )
    assert handler.calls == 2


def test_ambiguous_failure_retries_safe_route() -> None:
    handler = FlakyHandler(1, lambda: httpx.ReadTimeout("mid-flight"), success=EMPTY_MATCHES)
    client = sync_client(handler)
    client.context("sake").recall("cue")
    assert handler.calls == 2


def test_ambiguous_failure_never_retries_add_associations() -> None:
    """The one route where a phantom retry silently doubles weight."""
    handler = FlakyHandler(1, lambda: httpx.ReadTimeout("mid-flight"))
    client = sync_client(handler)
    with pytest.raises(TransportError):
        client.context("sake").add_associations(
            [{"subject": "s", "label": "l", "object": "o", "weight": 1.0}]
        )
    assert handler.calls == 1


def test_ambiguous_failure_never_retries_rename() -> None:
    """A phantom retry could rename an already-renamed context again."""
    handler = FlakyHandler(1, lambda: httpx.ReadTimeout("mid-flight"))
    client = sync_client(handler)
    with pytest.raises(TransportError):
        client.contexts.rename("sake", "shochu")
    assert handler.calls == 1

    handler = FlakyHandler(1, lambda: httpx.ReadTimeout("mid-flight"))
    client = sync_client(handler)
    with pytest.raises(TransportError):
        client.groups.rename("kura", "gura")
    assert handler.calls == 1


def test_retries_zero_disables_retry() -> None:
    handler = FlakyHandler(1, lambda: err_response(429, "budget", {"retry-after": "0"}))
    client = sync_client(handler, retries=0)
    with pytest.raises(RateLimitError):
        client.context("sake").recall("cue")
    assert handler.calls == 1


def test_retry_budget_exhausts_and_raises_last_error() -> None:
    handler = FlakyHandler(99, lambda: err_response(429, "budget", {"retry-after": "0"}))
    client = sync_client(handler, retries=2)
    with pytest.raises(RateLimitError):
        client.context("sake").recall("cue")
    assert handler.calls == 3  # initial + 2 retries


def test_parse_retry_after_takes_a_bare_delay_and_refuses_the_rest() -> None:
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("  0.5  ") == 0.5
    assert parse_retry_after("1e3") == 1000.0
    assert parse_retry_after("0") == 0.0
    # A trailing tail, a non-finite spelling, an overflow, or a negative value
    # is malformed — return None so the caller falls back to computed backoff
    # instead of sleeping on garbage (or forever).
    assert parse_retry_after("5 seconds") is None
    assert parse_retry_after("0x10") is None
    assert parse_retry_after("Infinity") is None
    assert parse_retry_after("1e400") is None
    assert parse_retry_after("-1") is None
    assert parse_retry_after("") is None
    assert parse_retry_after(None) is None
