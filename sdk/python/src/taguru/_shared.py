"""Transport-independent pieces shared by the sync and async clients."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, NoReturn

import httpx

from ._decode import decode
from ._errors import TaguruError, error_for_status
from ._models import GroupImportOutcome, ImportOutcome, ImportResult
from ._retry import parse_retry_after
from ._types import AssocOp

DEFAULT_BASE_URL = "http://127.0.0.1:8248"
ENV_URL = "TAGURU_URL"
ENV_TOKEN = "TAGURU_API_TOKEN"
# Matches the server's own TAGURU_REQUEST_TIMEOUT_SECS default; raise both
# together when the server has an embedding provider configured.
DEFAULT_TIMEOUT_SECS = 30.0
# Server-enforced caps mirrored client-side by add_associations_batched.
MAX_OPS_PER_REQUEST = 10_000
MAX_CHUNK_BYTES = 8 * 1024 * 1024


def encode_name(name: str) -> str:
    """Percent-encode one path segment (context names may be any UTF-8)."""
    from urllib.parse import quote

    return quote(name, safe="")


def citation_key(source: str, paragraph: int) -> tuple[str, int]:
    """The key ``RetrievalResult.citations`` is indexed by."""
    return (source, paragraph)


def drop_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Omit absent optional fields instead of sending nulls."""
    return {key: value for key, value in mapping.items() if value is not None}


def dumps_compact(obj: Any) -> bytes:
    """Serialize a JSON body compactly (UTF-8, no ASCII escaping).

    Used for every JSON request so that ``chunk_associations``'s byte
    accounting is exact for what actually goes on the wire.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def chunk_associations(
    ops: list[AssocOp], chunk_size: int, max_chunk_bytes: int
) -> Iterator[list[AssocOp]]:
    """Split a batch by both element count and serialized body size."""
    chunk: list[AssocOp] = []
    chunk_bytes = 2  # "[" + "]"
    for op in ops:
        op_bytes = len(dumps_compact(op))
        added = op_bytes + (1 if chunk else 0)  # separating comma
        if chunk and (len(chunk) >= chunk_size or chunk_bytes + added > max_chunk_bytes):
            yield chunk
            chunk = []
            chunk_bytes = 2
            added = op_bytes
        chunk.append(op)
        chunk_bytes += added
    if chunk:
        yield chunk


def raise_for_response(response: httpx.Response) -> NoReturn:
    """Raise the mapped error for a non-2xx response.

    The body is normally the JSON error shape (whose ``code`` rides onto the
    error); a plain-text body (e.g. a pre-JSON-413 server) still maps by
    status alone.
    """
    status = response.status_code
    retry_after = parse_retry_after(response.headers.get("retry-after"))
    time: float | None = None
    body: object
    try:
        data = response.json()
    except ValueError:
        text = response.text
        body = text
        message = text.strip() or f"HTTP {status}"
    else:
        body = data
        code = None
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            message = data["error"]
            reported = data.get("time")
            if isinstance(reported, (int, float)):
                time = float(reported)
            if isinstance(data.get("code"), str):
                code = data["code"]
        else:
            message = f"HTTP {status}"
        raise error_for_status(
            status, message, code=code, time=time, body=body, retry_after=retry_after
        )
    raise error_for_status(status, message, time=time, body=body, retry_after=retry_after)


def unwrap_envelope(response: httpx.Response) -> Any:
    """Extract ``result`` from the ``{"result", "status": "ok", "time"}`` envelope."""
    try:
        data = response.json()
    except ValueError as exc:
        raise TaguruError(
            "expected a JSON envelope, got a non-JSON body",
            status=response.status_code,
            body=response.text,
        ) from exc
    if isinstance(data, dict) and data.get("status") == "ok" and "result" in data:
        return data["result"]
    raise TaguruError(
        "response is not the taguru envelope shape",
        status=response.status_code,
        body=data,
    )


def normalize_import_outcomes(result: Any) -> ImportResult:
    """Normalize /import's response to ``ImportResult(batches, groups)``.

    Current servers always answer ``{batches: [...], groups: [...]}``
    (``groups`` omitted entirely when the stream carried none); servers
    predating that change answered a bare outcome for a single batch — both
    parse here, so callers never branch on response shape.
    """
    if isinstance(result, dict) and isinstance(result.get("batches"), list):
        batches = [decode(ImportOutcome, outcome) for outcome in result["batches"]]
        groups = [decode(GroupImportOutcome, group) for group in result.get("groups", [])]
        return ImportResult(batches=batches, groups=groups)
    return ImportResult(batches=[decode(ImportOutcome, result)], groups=[])
