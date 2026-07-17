"""Error hierarchy raised by the Taguru client.

Every non-2xx HTTP response maps to a :class:`TaguruError` subclass chosen by
HTTP status. The server's error body
(``{"status": "error", "code": "...", "error": "...", "time": ...}``) also
carries a stable machine-readable ``code`` (e.g. ``"no_context"``,
``"over_limit"`` — ``GET /protocol`` lists the vocabulary), surfaced here as
``.code`` for branching finer than the class hierarchy; servers predating the
field yield ``None``. :class:`TransportError` covers failures where no HTTP
response was obtained at all.
"""

from __future__ import annotations

__all__ = [
    "TaguruError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "PayloadTooLargeError",
    "RequestTimeoutError",
    "RateLimitError",
    "ServerError",
    "ServiceUnavailableError",
    "StorageFullError",
    "EmbeddingUnavailableError",
    "TransportError",
    "UnexpectedStatusError",
    "error_for_status",
]


class TaguruError(Exception):
    """Base class for every error this SDK raises.

    Attributes:
        message: The server's ``error`` text (or a synthesized description).
        status: HTTP status code, or ``None`` when no response was obtained.
        code: The server's stable machine-readable failure kind (e.g.
            ``"no_context"``), or ``None`` when the response carried none.
        time: The server-reported handling time in seconds, when available.
        body: The parsed error body (dict) or raw text, when available.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        time: float | None = None,
        body: object = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.time = time
        self.body = body


class AuthenticationError(TaguruError):
    """401 — missing or wrong bearer token."""


class PermissionDeniedError(TaguruError):
    """403 — the key's role or context scope does not cover this operation."""


class NotFoundError(TaguruError):
    """404 — unknown context, source, paragraph, or route."""


class ConflictError(TaguruError):
    """409 — duplicate create, alias conflict, or partial-write conflict."""


class ValidationError(TaguruError):
    """400/415/422 — malformed body, wrong content type, or mistyped JSON."""


class PayloadTooLargeError(TaguruError):
    """413 — request body over the server's cap.

    Current servers answer it in the JSON error shape; servers predating
    that change answered axum's plain text — both parse here.
    """


class RequestTimeoutError(TaguruError):
    """408 — the server's own per-request budget expired."""


class RateLimitError(TaguruError):
    """429 — this key is over its request budget; wait ``retry_after`` seconds."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = 429,
        code: str | None = None,
        time: float | None = None,
        body: object = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status=status, code=code, time=time, body=body)
        self.retry_after = retry_after


class ServerError(TaguruError):
    """500 and any unmapped 5xx."""


class ServiceUnavailableError(ServerError):
    """503 — load shedding or a degraded write path; honors ``retry_after``."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = 503,
        code: str | None = None,
        time: float | None = None,
        body: object = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status=status, code=code, time=time, body=body)
        self.retry_after = retry_after


class StorageFullError(ServerError):
    """507 — the write hit a storage cap and was NOT applied."""


class EmbeddingUnavailableError(ServerError):
    """501/502 — no embedding provider configured, or the provider failed.

    ``reason`` is ``"not_configured"`` for 501 and ``"provider_error"`` for 502.
    Unlike the single-status sibling errors, this class covers two statuses,
    so neither has a safe implicit default — ``status`` must be passed as
    501 or 502 explicitly.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        code: str | None = None,
        time: float | None = None,
        body: object = None,
    ) -> None:
        if status not in (501, 502):
            raise ValueError(
                f"EmbeddingUnavailableError requires status=501 or 502, got {status!r}"
            )
        super().__init__(message, status=status, code=code, time=time, body=body)
        self.reason = "not_configured" if status == 501 else "provider_error"


class TransportError(TaguruError):
    """No HTTP response was obtained (DNS, refused connection, timeout, ...)."""


class UnexpectedStatusError(TaguruError):
    """Any status with no specific mapping (e.g. 405) — exhaustiveness fallback."""


def error_for_status(
    status: int,
    message: str,
    *,
    code: str | None = None,
    time: float | None = None,
    body: object = None,
    retry_after: float | None = None,
) -> TaguruError:
    """Build the error for an HTTP status, endpoint-independently."""
    if status == 401:
        return AuthenticationError(message, status=status, code=code, time=time, body=body)
    if status == 403:
        return PermissionDeniedError(message, status=status, code=code, time=time, body=body)
    if status == 404:
        return NotFoundError(message, status=status, code=code, time=time, body=body)
    if status == 409:
        return ConflictError(message, status=status, code=code, time=time, body=body)
    if status in (400, 415, 422):
        return ValidationError(message, status=status, code=code, time=time, body=body)
    if status == 413:
        return PayloadTooLargeError(message, status=status, code=code, time=time, body=body)
    if status == 408:
        return RequestTimeoutError(message, status=status, code=code, time=time, body=body)
    if status == 429:
        return RateLimitError(
            message, status=status, code=code, time=time, body=body, retry_after=retry_after
        )
    if status == 503:
        return ServiceUnavailableError(
            message, status=status, code=code, time=time, body=body, retry_after=retry_after
        )
    if status == 507:
        return StorageFullError(message, status=status, code=code, time=time, body=body)
    if status in (501, 502):
        return EmbeddingUnavailableError(message, status=status, code=code, time=time, body=body)
    if 500 <= status < 600:
        return ServerError(message, status=status, code=code, time=time, body=body)
    return UnexpectedStatusError(message, status=status, code=code, time=time, body=body)
