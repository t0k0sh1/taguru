"""The status → error-class table, endpoint-independent."""

from __future__ import annotations

import httpx
import pytest

from taguru import (
    AuthenticationError,
    ConflictError,
    EmbeddingUnavailableError,
    NotFoundError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    ServiceUnavailableError,
    StorageFullError,
    TaguruError,
    UnexpectedStatusError,
    ValidationError,
    error_for_status,
)

from .conftest import err_response, sync_client


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, ValidationError),
        (401, AuthenticationError),
        (403, PermissionDeniedError),
        (404, NotFoundError),
        (405, UnexpectedStatusError),
        (408, RequestTimeoutError),
        (409, ConflictError),
        (413, PayloadTooLargeError),
        (415, ValidationError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
        (501, EmbeddingUnavailableError),
        (502, EmbeddingUnavailableError),
        (503, ServiceUnavailableError),
        (507, StorageFullError),
        (599, ServerError),
    ],
)
def test_status_maps_to_error_class(status: int, expected: type[TaguruError]) -> None:
    client = sync_client(lambda _req: err_response(status, "boom"), retries=0)
    with pytest.raises(expected) as excinfo:
        client.context("sake").recall("cue")
    assert excinfo.value.status == status
    assert excinfo.value.message == "boom"
    assert excinfo.value.time == 0.001


def test_machine_readable_code_is_surfaced() -> None:
    """The server's stable `code` rides every error; old servers yield None."""
    client = sync_client(
        lambda _req: err_response(404, "context 'x' not found", code="no_context"), retries=0
    )
    with pytest.raises(NotFoundError) as excinfo:
        client.context("sake").recall("cue")
    assert excinfo.value.code == "no_context"

    coded = sync_client(
        lambda _req: err_response(429, "budget", {"retry-after": "7"}, code="rate_limited"),
        retries=0,
    )
    with pytest.raises(RateLimitError) as excinfo2:
        coded.context("sake").recall("cue")
    assert excinfo2.value.code == "rate_limited"
    assert excinfo2.value.retry_after == 7.0

    # A body without the field (a server predating it) decodes to None.
    legacy = sync_client(lambda _req: err_response(404, "gone"), retries=0)
    with pytest.raises(NotFoundError) as excinfo3:
        legacy.context("sake").recall("cue")
    assert excinfo3.value.code is None


def test_plain_text_413_still_maps() -> None:
    """Axum's body-limit rejection answers plain text, not the JSON shape."""
    client = sync_client(lambda _req: httpx.Response(413, text="length limit exceeded"), retries=0)
    with pytest.raises(PayloadTooLargeError) as excinfo:
        client.context("sake").recall("cue")
    assert excinfo.value.message == "length limit exceeded"
    assert excinfo.value.body == "length limit exceeded"


def test_rate_limit_carries_retry_after() -> None:
    client = sync_client(
        lambda _req: err_response(429, "over budget", {"retry-after": "7"}), retries=0
    )
    with pytest.raises(RateLimitError) as excinfo:
        client.context("sake").recall("cue")
    assert excinfo.value.retry_after == 7.0


def test_service_unavailable_is_a_server_error_and_carries_retry_after() -> None:
    client = sync_client(lambda _req: err_response(503, "shed", {"retry-after": "2"}), retries=0)
    with pytest.raises(ServiceUnavailableError) as excinfo:
        client.context("sake").recall("cue")
    assert isinstance(excinfo.value, ServerError)
    assert excinfo.value.retry_after == 2.0


def test_embedding_error_reason_distinguishes_501_from_502() -> None:
    client = sync_client(lambda _req: err_response(501, "no provider"), retries=0)
    with pytest.raises(EmbeddingUnavailableError) as excinfo:
        client.context("sake").refresh_embeddings()
    assert excinfo.value.reason == "not_configured"

    client = sync_client(lambda _req: err_response(502, "provider died"), retries=0)
    with pytest.raises(EmbeddingUnavailableError) as excinfo:
        client.context("sake").refresh_embeddings()
    assert excinfo.value.reason == "provider_error"


@pytest.mark.parametrize("status", [None, 500, 503])
def test_embedding_error_rejects_any_status_other_than_501_or_502(
    status: int | None,
) -> None:
    with pytest.raises(ValueError):
        EmbeddingUnavailableError("boom", status=status)


def test_error_for_status_is_reexported_and_builds_by_status() -> None:
    """``error_for_status`` builds the same table the client applies
    internally — it must be importable from the top-level package too, not
    just from the private ``_errors`` module the client itself uses."""
    error = error_for_status(404, "not found", code="no_context")
    assert isinstance(error, NotFoundError)
    assert error.code == "no_context"


_ATTRIBUTE_CARRIER_CASES = [
    (401, AuthenticationError),
    (403, PermissionDeniedError),
    (404, NotFoundError),
    (409, ConflictError),
    (400, ValidationError),
    (415, ValidationError),
    (422, ValidationError),
    (413, PayloadTooLargeError),
    (408, RequestTimeoutError),
    (429, RateLimitError),
    (503, ServiceUnavailableError),
    (507, StorageFullError),
    (501, EmbeddingUnavailableError),
    (502, EmbeddingUnavailableError),
    (500, ServerError),
    (599, ServerError),
    (600, UnexpectedStatusError),
    (418, UnexpectedStatusError),
]


@pytest.mark.parametrize(
    "status,expected",
    _ATTRIBUTE_CARRIER_CASES,
    ids=[f"{status}-{cls.__name__}" for status, cls in _ATTRIBUTE_CARRIER_CASES],
)
def test_error_for_status_threads_every_attribute_through(
    status: int, expected: type[TaguruError]
) -> None:
    """Each mapped class must carry ALL the caller's fields, not just be the
    right type — a dropped ``code``/``time``/``body`` keyword would lose
    exactly the diagnostics the error exists to surface. 600 rides along to
    pin the 5xx range's exclusive upper bound."""
    body = {"detail": "sentinel"}
    error = error_for_status(
        status, "boom", code="some_code", time=1.25, body=body, retry_after=7.5
    )
    assert type(error) is expected, f"{status} must map to exactly {expected.__name__}"
    assert error.message == "boom"
    assert error.status == status
    assert error.code == "some_code"
    assert error.time == 1.25
    assert error.body is body
    if isinstance(error, (RateLimitError, ServiceUnavailableError)):
        assert error.retry_after == 7.5


def test_direct_constructors_default_their_status_and_keep_every_field() -> None:
    body = {"detail": "sentinel"}

    rate = RateLimitError("boom", code="over_limit", time=0.5, body=body, retry_after=2.0)
    assert (rate.status, rate.code, rate.time, rate.body, rate.retry_after) == (
        429,
        "over_limit",
        0.5,
        body,
        2.0,
    )
    assert RateLimitError("boom").status == 429

    unavailable = ServiceUnavailableError(
        "boom", code="shedding", time=0.5, body=body, retry_after=2.0
    )
    assert (
        unavailable.status,
        unavailable.code,
        unavailable.time,
        unavailable.body,
        unavailable.retry_after,
    ) == (503, "shedding", 0.5, body, 2.0)
    assert ServiceUnavailableError("boom").status == 503

    embedding = EmbeddingUnavailableError(
        "boom", status=501, code="not_configured", time=0.5, body=body
    )
    assert (embedding.status, embedding.code, embedding.time, embedding.body) == (
        501,
        "not_configured",
        0.5,
        body,
    )


def test_embedding_error_rejection_names_the_offending_status() -> None:
    with pytest.raises(ValueError) as excinfo:
        EmbeddingUnavailableError("boom", status=500)
    assert str(excinfo.value) == "EmbeddingUnavailableError requires status=501 or 502, got 500"
