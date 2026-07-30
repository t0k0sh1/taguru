"""The one-time `GET /version` compatibility preflight (ADR 0005 §3.8, §6).

Every case here opts in with `check_contract=True` — the default client
fixtures pre-seed the check as already done (see `conftest.py`) so the rest
of this suite's exact call-count and call-path assertions aren't disturbed by
a surprise probe.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

import taguru
from taguru import IncompatibleServerError, TransportError

from .conftest import async_client, ok_response, sync_client

EMPTY_MATCHES = {"total": 0, "matches": []}


def version_response(payload: dict[str, object], status: int = 200) -> httpx.Response:
    """A bare `/version` body — not the `{result, status: "ok"}` envelope."""
    return httpx.Response(status, json=payload)


def compatible_version(**extra: object) -> dict[str, object]:
    return {"server": "0.6.0", "http_contract": {"current": 1, "supported": [1]}, **extra}


def incompatible_newer() -> dict[str, object]:
    return {"server": "0.7.0", "http_contract": {"current": 2, "supported": [2]}}


def test_preflight_runs_once_per_client() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == "/version":
            return version_response(compatible_version())
        return ok_response(EMPTY_MATCHES)

    client = sync_client(handler, check_contract=True)
    client.context("sake").recall("cue")
    client.context("sake").recall("cue")

    assert paths.count("/version") == 1
    assert paths == ["/version", "/contexts/sake/recall", "/contexts/sake/recall"]


async def test_preflight_runs_once_per_client_async() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == "/version":
            return version_response(compatible_version())
        return ok_response(EMPTY_MATCHES)

    client = async_client(handler, check_contract=True)
    await client.context("sake").recall("cue")
    await client.context("sake").recall("cue")

    assert paths.count("/version") == 1


async def test_concurrent_first_calls_share_one_probe_and_all_see_incompatibility() -> None:
    """A `gather()` of several calls on a fresh client must not let every
    caller but the first race past an unfinished compatibility check — a
    merely synchronous "checked" flag set before the probe's own `await`
    would let a second concurrent call see the flag already (falsely)
    settled and reach the real server before the first call's probe (still
    in flight) discovers the incompatibility."""
    paths: list[str] = []
    release_probe = asyncio.Event()

    async def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == "/version":
            # Delay the probe's own response so both `recall()` calls are
            # definitely both in flight, racing against it, before it
            # resolves — a same-tick race wouldn't prove much on its own.
            await release_probe.wait()
            return version_response(
                {"server": "0.7.0", "http_contract": {"current": 2, "supported": [2]}}
            )
        raise AssertionError(f"the real request must not run: {req.url.path}")

    client = async_client(handler, check_contract=True)  # type: ignore[arg-type]

    async def release_soon() -> None:
        await asyncio.sleep(0)
        release_probe.set()

    results = await asyncio.gather(
        client.context("sake").recall("cue"),
        client.context("tea").recall("cue"),
        release_soon(),
        return_exceptions=True,
    )

    # Both real calls see the incompatibility — neither raced ahead of
    # the shared, still-in-flight probe.
    assert isinstance(results[0], IncompatibleServerError)
    assert isinstance(results[1], IncompatibleServerError)
    assert paths == ["/version"], "only one probe, and no real request ever ran"


def test_incompatible_newer_server_raises_with_upgrade_sdk_remedy() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"the real request must not run: {req.url.path}")

    client = sync_client(handler, check_contract=True)
    with pytest.raises(IncompatibleServerError) as excinfo:
        client.context("sake").recall("cue")

    error = excinfo.value
    assert error.status is None
    assert error.sdk_version == taguru.__version__
    assert error.server_version == "0.7.0"
    assert error.supported_contracts == (1,)
    assert error.server_contracts == (2,)
    assert "0.7.0" in str(error)
    assert "Upgrade this SDK" in str(error)


async def test_incompatible_newer_server_raises_async() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"the real request must not run: {req.url.path}")

    client = async_client(handler, check_contract=True)
    with pytest.raises(IncompatibleServerError):
        await client.context("sake").recall("cue")


def test_incompatible_older_server_raises_with_upgrade_server_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("taguru._contract.SUPPORTED_HTTP_CONTRACTS", (2,))

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(compatible_version())  # server still speaks contract 1
        raise AssertionError(f"the real request must not run: {req.url.path}")

    client = sync_client(handler, check_contract=True)
    with pytest.raises(IncompatibleServerError) as excinfo:
        client.context("sake").recall("cue")

    error = excinfo.value
    assert error.supported_contracts == (2,)
    assert error.server_contracts == (1,)
    assert "Upgrade the server" in str(error)


def test_disjoint_interleaved_ranges_raise_the_generic_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither side is simply "newer" — SDK speaks {1, 3}, server speaks
    {2} — so the remedy names no direction, just that one side must
    move (ADR 0005 §6's dual-serving window is not decided, but the
    message must still make sense if it ever exists)."""
    monkeypatch.setattr("taguru._contract.SUPPORTED_HTTP_CONTRACTS", (1, 3))

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(
                {"server": "0.7.0", "http_contract": {"current": 2, "supported": [2]}}
            )
        raise AssertionError(f"the real request must not run: {req.url.path}")

    client = sync_client(handler, check_contract=True)
    with pytest.raises(IncompatibleServerError) as excinfo:
        client.context("sake").recall("cue")
    assert "SUPPORTED_HTTP_CONTRACTS" in str(excinfo.value)


@pytest.mark.parametrize(
    "version_handler",
    [
        pytest.param(lambda _req: httpx.Response(404), id="404-pre-0.6-server"),
        pytest.param(lambda _req: httpx.Response(200, text="ok"), id="bare-ok-text"),
        pytest.param(lambda _req: httpx.Response(200, json=[1, 2, 3]), id="json-array-body"),
        pytest.param(
            lambda _req: version_response({"server": "0.6.0"}), id="missing-http_contract"
        ),
        pytest.param(
            lambda _req: version_response({"http_contract": {"current": 1, "supported": []}}),
            id="empty-supported",
        ),
        pytest.param(
            lambda _req: version_response({"http_contract": {"current": 1, "supported": ["one"]}}),
            id="non-integer-supported",
        ),
        pytest.param(
            lambda _req: httpx.Response(401, json={"status": "error", "error": "nope"}),
            id="401-behind-a-proxy",
        ),
    ],
)
def test_uninformative_version_responses_fail_open(version_handler: object) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_handler(req)  # type: ignore[operator]
        return ok_response(EMPTY_MATCHES)

    client = sync_client(handler, check_contract=True)
    client.context("sake").recall("cue")  # must not raise


def test_transport_failure_leaves_the_contract_unchecked_for_a_retry() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        raise httpx.ConnectError("connection refused")

    client = sync_client(handler, check_contract=True, retries=0)

    # The server is entirely unreachable, so both the probe and the real
    # request fail — the caller sees the real request's own error, not
    # a /version-specific one.
    with pytest.raises(TransportError):
        client.context("sake").recall("cue")
    assert paths == ["/version", "/contexts/sake/recall"]

    # `_contract_checked` stayed False (only a successful probe or a
    # positive-but-uninformative answer marks it checked), so the next
    # call retries the preflight rather than skipping it forever.
    with pytest.raises(TransportError):
        client.context("sake").recall("cue")
    assert paths == [
        "/version",
        "/contexts/sake/recall",
        "/version",
        "/contexts/sake/recall",
    ]


def test_export_stream_raises_on_incompatibility() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"the export must not run: {req.url.path}")

    client = sync_client(handler, check_contract=True)
    with pytest.raises(IncompatibleServerError):
        list(client.context("sake").export_stream())


async def test_export_stream_raises_on_incompatibility_async() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"the export must not run: {req.url.path}")

    client = async_client(handler, check_contract=True)
    with pytest.raises(IncompatibleServerError):
        async for _chunk in client.context("sake").export_stream():
            pass


def test_export_to_file_raises_on_incompatibility(tmp_path: object) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"the export must not run: {req.url.path}")

    client = sync_client(handler, check_contract=True)
    target = tmp_path / "backup.jsonl"  # type: ignore[operator]
    with pytest.raises(IncompatibleServerError):
        client.context("sake").export_to_file(target)
    assert not target.exists()


def test_wait_until_ready_raises_immediately_instead_of_stalling() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"live/health must not run: {req.url.path}")

    client = sync_client(handler, check_contract=True)
    started = time.monotonic()
    with pytest.raises(IncompatibleServerError):
        client.wait_until_ready(timeout=5.0, interval=0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, "must not wait out the full timeout"
    assert paths == ["/version"], "must not retry a confirmed incompatibility"


async def test_wait_until_ready_raises_immediately_instead_of_stalling_async() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/version":
            return version_response(incompatible_newer())
        raise AssertionError(f"live/health must not run: {req.url.path}")

    client = async_client(handler, check_contract=True)
    started = time.monotonic()
    with pytest.raises(IncompatibleServerError):
        await client.wait_until_ready(timeout=5.0, interval=0.5)
    assert time.monotonic() - started < 1.0
