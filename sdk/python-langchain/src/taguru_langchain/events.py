"""Progress/diagnostic events for :class:`~taguru_langchain.ingest.TaguruIngester`.

A discriminated union of frozen dataclasses, one per stage of an ingest run
(document -> chunk -> attempt -> import -> embedding refresh). Every variant
carries a ``kind`` literal so callers can ``match`` on it exhaustively.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True, kw_only=True)
class ProviderMetadata:
    """Provider-reported details for one LLM call, when the model exposes them.

    ``finish_reason`` is read from ``response_metadata`` under whichever of
    ``done_reason``/``finish_reason``/``stop_reason`` the provider used.
    Token counts come from LangChain's normalized ``usage_metadata``.
    """

    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(slots=True, frozen=True, kw_only=True)
class DocumentStarted:
    kind: Literal["document_started"] = "document_started"
    source: str
    text_bytes: int


@dataclass(slots=True, frozen=True, kw_only=True)
class ChunkStarted:
    kind: Literal["chunk_started"] = "chunk_started"
    source: str
    index: int
    total: int


@dataclass(slots=True, frozen=True, kw_only=True)
class AttemptStarted:
    kind: Literal["attempt_started"] = "attempt_started"
    source: str
    chunk_index: int
    attempt: int
    max_attempts: int
    stage: Literal["item", "cross_chunk"] = "item"
    """Which corrective loop this attempt belongs to: ``"item"`` is the
    per-chunk Stage 1 loop (syntax/validity), ``"cross_chunk"`` is the
    single targeted Stage 2 correction ``cross_output_issues`` can trigger
    once every chunk's output is known (issue #180's ADR 0001 §8 bucket 2,
    ported from the Rust twin's ``cross_output_issues``)."""


@dataclass(slots=True, frozen=True, kw_only=True)
class AttemptFailed:
    kind: Literal["attempt_failed"] = "attempt_failed"
    source: str
    chunk_index: int
    attempt: int
    max_attempts: int
    parse_error: str
    elapsed_seconds: float
    provider_metadata: ProviderMetadata | None
    length_limited: bool
    """Whether ``provider_metadata.finish_reason`` says this answer was cut
    off at the provider's output cap (see
    :func:`taguru_langchain._extract.indicates_length_limit`) — the next
    attempt's corrective turn asks for a shorter answer instead of
    repeating the same ask verbatim when this is true."""
    stage: Literal["item", "cross_chunk"] = "item"
    """See :attr:`AttemptStarted.stage`."""
    validation_issues: list[str] | None = None
    """The path-addressed issues (e.g. ``"associations[1].weight: expected
    finite non-zero number, got string \\"strong\\""``) that failed this
    attempt, when it was syntactically valid JSON but not a valid
    extraction (:class:`taguru_langchain._extract.InvalidFault`) or a
    Stage 2 cross-chunk alias problem. ``None`` for every other failure
    kind — ``parse_error`` already carries a human-readable diagnosis for
    all of them."""


@dataclass(slots=True, frozen=True, kw_only=True)
class ChunkCompleted:
    """Raw per-chunk proposal counts, from before cross-chunk merge/dedup.

    These intentionally do not match ``IngestOutcome``'s final tallies
    (``associations``, ``aliases``, ...): ``merge()`` dedupes across chunks,
    so a per-chunk "accepted" count doesn't exist until every chunk is in.
    """

    kind: Literal["chunk_completed"] = "chunk_completed"
    source: str
    index: int
    total: int
    associations_proposed: int
    aliases_proposed: int
    questions_proposed: int
    llm_calls: int
    elapsed_seconds: float


@dataclass(slots=True, frozen=True, kw_only=True)
class ImportStarted:
    kind: Literal["import_started"] = "import_started"
    source: str


@dataclass(slots=True, frozen=True, kw_only=True)
class ImportCompleted:
    kind: Literal["import_completed"] = "import_completed"
    source: str
    elapsed_seconds: float


@dataclass(slots=True, frozen=True, kw_only=True)
class EmbeddingRefreshStarted:
    kind: Literal["embedding_refresh_started"] = "embedding_refresh_started"
    source: str


@dataclass(slots=True, frozen=True, kw_only=True)
class EmbeddingRefreshCompleted:
    """Terminal, non-error outcome of a refresh attempt.

    ``configured=False`` covers the 501 "no provider configured" case,
    which is expected steady state for most deployments, not a failure.
    """

    kind: Literal["embedding_refresh_completed"] = "embedding_refresh_completed"
    source: str
    configured: bool
    embedded: int
    total: int


@dataclass(slots=True, frozen=True, kw_only=True)
class EmbeddingRefreshWarning:
    kind: Literal["embedding_refresh_warning"] = "embedding_refresh_warning"
    source: str
    message: str


IngestEvent = (
    DocumentStarted
    | ChunkStarted
    | AttemptStarted
    | AttemptFailed
    | ChunkCompleted
    | ImportStarted
    | ImportCompleted
    | EmbeddingRefreshStarted
    | EmbeddingRefreshCompleted
    | EmbeddingRefreshWarning
)

IngestEventCallback = Callable[[IngestEvent], None]
"""Must be synchronous and non-blocking — ``aingest_text`` calls it directly,
without a thread hop or an await, so a slow callback stalls the event loop.
Exceptions raised here are caught and reported via ``warnings.warn``; they
never interrupt the ingest they were reporting on."""
