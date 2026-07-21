"""TaguruIngester: LLM-driven document decomposition into a Taguru context.

The LangChain twin of ``taguru extract``: a chat model decomposes each
document into associations/aliases (and optional doc2query questions) under
the protocol's ingest discipline, and the result is applied through
``POST /import`` — one batch per source, retract-then-apply, so re-ingesting
a document replaces its contribution instead of double-counting weights.

Two improvements over the offline extractor, possible only against a live
server: the context's existing relation vocabulary seeds the prompt (reuse
over synonym-coining), and embeddings refresh best-effort after each run.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from taguru import (
    AsyncTaguru,
    EmbeddingUnavailableError,
    ImportOutcome,
    NotFoundError,
    Taguru,
)

from ._extract import (
    CHUNK_BYTES,
    MAX_PASSAGE_BYTES,
    MAX_QUESTIONS_PER_PARAGRAPH,
    VOCABULARY_CAP,
    ModelOutput,
    chunk,
    corrective_assistant_turn_content,
    corrective_message,
    indicates_length_limit,
    labeled_document,
    merge,
    parse_model_output,
    render_batch,
    reparse_batch,
    split_paragraphs,
    system_prompt,
    user_message,
)
from .events import (
    AttemptFailed,
    AttemptStarted,
    ChunkCompleted,
    ChunkStarted,
    DocumentStarted,
    EmbeddingRefreshCompleted,
    EmbeddingRefreshStarted,
    EmbeddingRefreshWarning,
    ImportCompleted,
    ImportStarted,
    IngestEvent,
    IngestEventCallback,
    ProviderMetadata,
)

DEFAULT_MAX_ATTEMPTS = 2

# Hard ceiling on max_attempts: a misconfigured value must not be able to
# turn one stubborn chunk into an unbounded number of model calls. Kept in
# sync with src/extract.rs's MAX_EXTRACT_ATTEMPTS.
MAX_ATTEMPTS_CEILING = 10


@dataclass
class IngestOutcome:
    """What one document's ingest amounted to.

    ``ndjson`` always carries the rendered batch for inspection (dry-run or
    not). The ``created``/``retracted``/... fields pass through the server's
    ImportOutcome when the batch was applied.
    """

    source: str
    ok: bool
    ndjson: str | None = None
    created: bool = False
    retracted: int = 0
    associations: int = 0
    aliases: int = 0
    passage_stored: bool = False
    questions_stored: int = 0
    duplicates_dropped: int = 0
    invalid_dropped: int = 0
    llm_calls: int = 0
    chunks: int = 0
    error: str | None = None
    embeddings_refresh_warning: str | None = None


class TaguruIngester:
    """Decompose LangChain Documents into one Taguru context via a chat model.

    Args:
        context: Target context name.
        llm: Any LangChain chat model; asked for a single JSON object, with
            corrective turns on a malformed answer (see ``max_attempts``),
            mirroring taguru extract.
        client / async_client: Core-SDK clients; built from
            ``base_url``/``api_key`` (or env) when neither is given.
        create_context: Stamp a create block on each batch so the context is
            created on first ingest (requires ``context_description``).
        source_key: Document metadata key holding the source id — REQUIRED on
            every document, no hashing fallback: the source id is the
            retract-then-apply idempotency unit, and a content hash would mint
            a new source on every edit, orphaning the old one forever.
        questions: doc2query questions per paragraph (None/0 = don't ask;
            capped at 8, the server's own per-paragraph cap).
        fact_budget: Soft cap on associations per chunk, folded into the
            prompt as an instruction (None = off, the default — unbounded;
            0 is rejected, use None instead). Never enforced post-hoc: a
            model that ignores it just returns everything it produced.
        max_attempts: Total attempts (1 initial + corrections) at getting the
            model to answer with the JSON object asked for, from 1 to
            ``MAX_ATTEMPTS_CEILING`` (default ``DEFAULT_MAX_ATTEMPTS`` = 2,
            today's fixed behavior). Each retry rebuilds the conversation
            from the system/user base and appends only the most recent bad
            turn, so ``corrective_context_bytes`` bounds every retry alike,
            not just the first.
        corrective_context_bytes: How much of the model's own prior bad
            answer gets replayed back to it in the next attempt's corrective
            turn: ``None`` replays it in full (the default, today's
            behavior), ``0`` omits it behind a placeholder, and any other
            value truncates it to that many bytes.
        include_passage: Store the verbatim document as the batch's passage
            (paragraph locators are stripped when off, matching extract).
        chunk_bytes: Prompt-input chunk cap; the stored passage is never
            chunked.
        vocabulary_cap: How many existing labels to seed the prompt with.
        refresh_embeddings: Best-effort ``embeddings/refresh`` after each
            call (501 = not configured is ignored; 502 lands as a warning).
        raise_on_error: ``ingest_documents`` raises on the first failed
            document instead of reporting it in its outcome.
        on_event: Optional callback invoked synchronously for each stage of
            an ingest run — document/chunk/attempt/import/embedding-refresh
            progress (see :mod:`taguru_langchain.events`). Must not block:
            ``aingest_text`` calls it directly too, with no thread hop or
            await. Exceptions it raises are caught and reported via
            ``warnings.warn`` rather than failing the ingest.
    """

    def __init__(
        self,
        *,
        context: str,
        llm: BaseChatModel,
        client: Taguru | None = None,
        async_client: AsyncTaguru | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        create_context: bool = False,
        context_description: str | None = None,
        source_key: str = "source",
        questions: int | None = None,
        fact_budget: int | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        corrective_context_bytes: int | None = None,
        include_passage: bool = True,
        chunk_bytes: int = CHUNK_BYTES,
        vocabulary_cap: int = VOCABULARY_CAP,
        refresh_embeddings: bool = True,
        raise_on_error: bool = False,
        on_event: IngestEventCallback | None = None,
    ) -> None:
        if create_context and context_description is None:
            raise ValueError("create_context=True requires context_description")
        if questions is not None and not 0 <= questions <= MAX_QUESTIONS_PER_PARAGRAPH:
            raise ValueError(f"questions must be between 0 and {MAX_QUESTIONS_PER_PARAGRAPH}")
        if fact_budget is not None and fact_budget < 1:
            raise ValueError(
                "fact_budget must be a positive integer, or None to leave it unbounded"
            )
        if not 1 <= max_attempts <= MAX_ATTEMPTS_CEILING:
            raise ValueError(f"max_attempts must be between 1 and {MAX_ATTEMPTS_CEILING}")
        if corrective_context_bytes is not None and corrective_context_bytes < 0:
            raise ValueError(
                "corrective_context_bytes must be a non-negative integer, or None to replay "
                "the prior bad answer in full"
            )
        self.context = context
        self.llm = llm
        self._owns_clients = client is None and async_client is None
        if self._owns_clients:
            client = Taguru(base_url, api_key)
            async_client = AsyncTaguru(base_url, api_key)
        self.client = client
        self.async_client = async_client
        self.create_context = create_context
        self.context_description = context_description
        self.source_key = source_key
        self.questions = questions or 0
        self.fact_budget = fact_budget or 0
        self.max_attempts = max_attempts
        self.corrective_context_bytes = corrective_context_bytes
        self.include_passage = include_passage
        self.chunk_bytes = chunk_bytes
        self.vocabulary_cap = vocabulary_cap
        self.refresh_embeddings = refresh_embeddings
        self.raise_on_error = raise_on_error
        self.on_event = on_event

    # -- shared, transport-free pieces ------------------------------------

    def _source_of(self, document: Document) -> str:
        source = document.metadata.get(self.source_key)
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"document metadata lacks a {self.source_key!r} key — the source id is "
                "the retract-then-apply idempotency unit and cannot be invented"
            )
        return source

    def _corrective_turn(
        self,
        base_messages: list[BaseMessage],
        content: str,
        error: ValueError,
        length_limited: bool,
    ) -> list[BaseMessage]:
        """Rebuilds from the system/user base plus only the most recent bad
        turn — never the whole history — so ``corrective_context_bytes``
        bounds every retry alike, not just the first. ``length_limited``
        (the provider's own ``finish_reason`` saying the prior answer was
        cut off at its output cap) swaps the ask from "try again" to "try
        again shorter" (see ``corrective_message``) — repeating the
        same-length ask just reproduces the same cutoff, the stall Issue
        #178 reported. At the all-defaults policy, with a normal
        (non-length-limited) answer, this reproduces the previous fixed
        implementation's request bodies exactly."""
        return [
            *base_messages,
            AIMessage(
                content=corrective_assistant_turn_content(content, self.corrective_context_bytes)
            ),
            HumanMessage(content=corrective_message(str(error), length_limited, self.fact_budget)),
        ]

    def _build_batch(
        self,
        source: str,
        text: str,
        outputs: list[ModelOutput],
        outcome: IngestOutcome,
    ) -> str:
        paragraph_count = len(split_paragraphs(text))
        extraction = merge(outputs, self.questions, paragraph_count)
        outcome.duplicates_dropped = extraction.duplicates
        outcome.invalid_dropped = extraction.dropped
        description = self.context_description if self.create_context else None
        ndjson = render_batch(
            self.context,
            source,
            description,
            extraction,
            text if self.include_passage else None,
        )
        reparse_batch(ndjson)
        return ndjson

    def _record(self, outcome: IngestOutcome, applied: ImportOutcome) -> None:
        outcome.created = applied.created
        outcome.retracted = applied.retracted
        outcome.associations = applied.associations
        outcome.aliases = applied.aliases
        outcome.passage_stored = applied.passage_stored
        outcome.questions_stored = applied.questions_stored

    @staticmethod
    def _content_text(message: object) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return str(content)

    def _emit(self, event: IngestEvent) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception as error:
            warnings.warn(
                f"TaguruIngester on_event callback raised {error!r}; ingest continues",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _provider_metadata(response: AIMessage) -> ProviderMetadata | None:
        response_metadata = response.response_metadata or {}
        finish_reason: str | None = None
        for key in ("done_reason", "finish_reason", "stop_reason"):
            value = response_metadata.get(key)
            if value is not None:
                finish_reason = str(value)
                break
        usage = response.usage_metadata
        if finish_reason is None and usage is None:
            return None
        return ProviderMetadata(
            finish_reason=finish_reason,
            input_tokens=usage["input_tokens"] if usage else None,
            output_tokens=usage["output_tokens"] if usage else None,
            total_tokens=usage["total_tokens"] if usage else None,
        )

    # -- sync ----------------------------------------------------------------

    def ingest_text(self, text: str, *, source: str, dry_run: bool = False) -> IngestOutcome:
        """Ingest one text under one source id. Raises on failure (unlike
        ``ingest_documents``, there is no "continue with the rest" here)."""
        if self.client is None:
            raise ValueError("this ingester was built with only an async client")
        outcome = IngestOutcome(source=source, ok=False)
        if self.include_passage and len(text.encode("utf-8")) > MAX_PASSAGE_BYTES:
            raise ValueError(f"document exceeds the {MAX_PASSAGE_BYTES}-byte passage cap")
        self._emit(DocumentStarted(source=source, text_bytes=len(text.encode("utf-8"))))

        vocabulary = self._fetch_vocabulary()
        system = system_prompt(vocabulary, self.questions, self.fact_budget)
        chunks = chunk(labeled_document(text, self.chunk_bytes), self.chunk_bytes)
        outcome.chunks = len(chunks)

        outputs: list[ModelOutput] = []
        for index, piece in enumerate(chunks):
            chunk_started_at = time.monotonic()
            self._emit(ChunkStarted(source=source, index=index, total=len(chunks)))
            user = user_message(source, index, len(chunks), piece)
            base_messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            output = None
            prior_bad_turn: tuple[str, ValueError, bool] | None = None
            chunk_llm_calls = 0
            for attempt in range(1, self.max_attempts + 1):
                messages = (
                    base_messages
                    if prior_bad_turn is None
                    else self._corrective_turn(base_messages, *prior_bad_turn)
                )
                self._emit(
                    AttemptStarted(
                        source=source,
                        chunk_index=index,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                    )
                )
                attempt_started_at = time.monotonic()
                response = self.llm.invoke(messages)
                outcome.llm_calls += 1
                chunk_llm_calls += 1
                content = self._content_text(response)
                try:
                    output = parse_model_output(content)
                    break
                except ValueError as error:
                    metadata = self._provider_metadata(response)
                    length_limited = indicates_length_limit(
                        metadata.finish_reason if metadata else None
                    )
                    self._emit(
                        AttemptFailed(
                            source=source,
                            chunk_index=index,
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            parse_error=str(error),
                            elapsed_seconds=time.monotonic() - attempt_started_at,
                            provider_metadata=metadata,
                            length_limited=length_limited,
                        )
                    )
                    prior_bad_turn = (content, error, length_limited)
            if output is None:
                assert prior_bad_turn is not None
                raise ValueError(
                    f"the model would not produce the JSON object: {prior_bad_turn[1]}"
                )
            outputs.append(output)
            self._emit(
                ChunkCompleted(
                    source=source,
                    index=index,
                    total=len(chunks),
                    associations_proposed=len(output.associations),
                    aliases_proposed=len(output.aliases),
                    questions_proposed=len(output.questions),
                    llm_calls=chunk_llm_calls,
                    elapsed_seconds=time.monotonic() - chunk_started_at,
                )
            )

        ndjson = self._build_batch(source, text, outputs, outcome)
        outcome.ndjson = ndjson
        if dry_run:
            outcome.ok = True
            return outcome

        self._emit(ImportStarted(source=source))
        import_started_at = time.monotonic()
        applied = self.client.import_batches(ndjson)
        self._emit(
            ImportCompleted(source=source, elapsed_seconds=time.monotonic() - import_started_at)
        )
        self._record(outcome, applied.batches[0])
        outcome.ok = True

        if self.refresh_embeddings:
            self._emit(EmbeddingRefreshStarted(source=source))
            try:
                result = self.client.context(self.context).refresh_embeddings()
                self._emit(
                    EmbeddingRefreshCompleted(
                        source=source,
                        configured=True,
                        embedded=result.embedded,
                        total=result.total,
                    )
                )
            except EmbeddingUnavailableError as error:
                if error.reason == "provider_error":
                    outcome.embeddings_refresh_warning = error.message
                    self._emit(EmbeddingRefreshWarning(source=source, message=error.message))
                else:
                    self._emit(
                        EmbeddingRefreshCompleted(
                            source=source, configured=False, embedded=0, total=0
                        )
                    )
        return outcome

    def ingest_documents(
        self, documents: Sequence[Document], *, dry_run: bool = False
    ) -> list[IngestOutcome]:
        """Ingest each document independently; one failure never stops the
        rest (set ``raise_on_error=True`` to fail fast)."""
        outcomes: list[IngestOutcome] = []
        for document in documents:
            try:
                source = self._source_of(document)
            except ValueError as error:
                if self.raise_on_error:
                    raise
                outcomes.append(IngestOutcome(source="", ok=False, error=str(error)))
                continue
            try:
                outcomes.append(
                    self.ingest_text(document.page_content, source=source, dry_run=dry_run)
                )
            except Exception as error:
                if self.raise_on_error:
                    raise
                outcomes.append(IngestOutcome(source=source, ok=False, error=str(error)))
        return outcomes

    def _fetch_vocabulary(self) -> list[str]:
        """The context's live relation vocabulary — an advantage the offline
        extractor structurally lacks. Best-effort: an absent context is fine."""
        assert self.client is not None
        try:
            page = self.client.context(self.context).list_labels(limit=self.vocabulary_cap)
        except NotFoundError:
            return []
        return page.labels

    # -- async ------------------------------------------------------------------

    async def aingest_text(self, text: str, *, source: str, dry_run: bool = False) -> IngestOutcome:
        """Async ``ingest_text``."""
        if self.async_client is None:
            raise ValueError("this ingester was built with only a sync client")
        outcome = IngestOutcome(source=source, ok=False)
        if self.include_passage and len(text.encode("utf-8")) > MAX_PASSAGE_BYTES:
            raise ValueError(f"document exceeds the {MAX_PASSAGE_BYTES}-byte passage cap")
        self._emit(DocumentStarted(source=source, text_bytes=len(text.encode("utf-8"))))

        vocabulary = await self._afetch_vocabulary()
        system = system_prompt(vocabulary, self.questions, self.fact_budget)
        chunks = chunk(labeled_document(text, self.chunk_bytes), self.chunk_bytes)
        outcome.chunks = len(chunks)

        outputs: list[ModelOutput] = []
        for index, piece in enumerate(chunks):
            chunk_started_at = time.monotonic()
            self._emit(ChunkStarted(source=source, index=index, total=len(chunks)))
            user = user_message(source, index, len(chunks), piece)
            base_messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            output = None
            prior_bad_turn: tuple[str, ValueError, bool] | None = None
            chunk_llm_calls = 0
            for attempt in range(1, self.max_attempts + 1):
                messages = (
                    base_messages
                    if prior_bad_turn is None
                    else self._corrective_turn(base_messages, *prior_bad_turn)
                )
                self._emit(
                    AttemptStarted(
                        source=source,
                        chunk_index=index,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                    )
                )
                attempt_started_at = time.monotonic()
                response = await self.llm.ainvoke(messages)
                outcome.llm_calls += 1
                chunk_llm_calls += 1
                content = self._content_text(response)
                try:
                    output = parse_model_output(content)
                    break
                except ValueError as error:
                    metadata = self._provider_metadata(response)
                    length_limited = indicates_length_limit(
                        metadata.finish_reason if metadata else None
                    )
                    self._emit(
                        AttemptFailed(
                            source=source,
                            chunk_index=index,
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            parse_error=str(error),
                            elapsed_seconds=time.monotonic() - attempt_started_at,
                            provider_metadata=metadata,
                            length_limited=length_limited,
                        )
                    )
                    prior_bad_turn = (content, error, length_limited)
            if output is None:
                assert prior_bad_turn is not None
                raise ValueError(
                    f"the model would not produce the JSON object: {prior_bad_turn[1]}"
                )
            outputs.append(output)
            self._emit(
                ChunkCompleted(
                    source=source,
                    index=index,
                    total=len(chunks),
                    associations_proposed=len(output.associations),
                    aliases_proposed=len(output.aliases),
                    questions_proposed=len(output.questions),
                    llm_calls=chunk_llm_calls,
                    elapsed_seconds=time.monotonic() - chunk_started_at,
                )
            )

        ndjson = self._build_batch(source, text, outputs, outcome)
        outcome.ndjson = ndjson
        if dry_run:
            outcome.ok = True
            return outcome

        self._emit(ImportStarted(source=source))
        import_started_at = time.monotonic()
        applied = await self.async_client.import_batches(ndjson)
        self._emit(
            ImportCompleted(source=source, elapsed_seconds=time.monotonic() - import_started_at)
        )
        self._record(outcome, applied.batches[0])
        outcome.ok = True

        if self.refresh_embeddings:
            self._emit(EmbeddingRefreshStarted(source=source))
            try:
                result = await self.async_client.context(self.context).refresh_embeddings()
                self._emit(
                    EmbeddingRefreshCompleted(
                        source=source,
                        configured=True,
                        embedded=result.embedded,
                        total=result.total,
                    )
                )
            except EmbeddingUnavailableError as error:
                if error.reason == "provider_error":
                    outcome.embeddings_refresh_warning = error.message
                    self._emit(EmbeddingRefreshWarning(source=source, message=error.message))
                else:
                    self._emit(
                        EmbeddingRefreshCompleted(
                            source=source, configured=False, embedded=0, total=0
                        )
                    )
        return outcome

    async def aingest_documents(
        self, documents: Sequence[Document], *, dry_run: bool = False
    ) -> list[IngestOutcome]:
        """Async ``ingest_documents``."""
        outcomes: list[IngestOutcome] = []
        for document in documents:
            try:
                source = self._source_of(document)
            except ValueError as error:
                if self.raise_on_error:
                    raise
                outcomes.append(IngestOutcome(source="", ok=False, error=str(error)))
                continue
            try:
                outcomes.append(
                    await self.aingest_text(document.page_content, source=source, dry_run=dry_run)
                )
            except Exception as error:
                if self.raise_on_error:
                    raise
                outcomes.append(IngestOutcome(source=source, ok=False, error=str(error)))
        return outcomes

    async def _afetch_vocabulary(self) -> list[str]:
        assert self.async_client is not None
        try:
            page = await self.async_client.context(self.context).list_labels(
                limit=self.vocabulary_cap
            )
        except NotFoundError:
            return []
        return page.labels

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP client(s) this ingester built itself, best-effort.

        A client passed in via ``client``/``async_client`` stays the
        caller's to close. The sync client always closes cleanly here;
        closing the async client needs a running event loop, so one is
        spun up just for that when none is already running. Called from
        inside a running loop, the async client is left for :meth:`aclose`
        instead.
        """
        if not self._owns_clients:
            return
        if self.client is not None:
            self.client.close()
        if self.async_client is not None:
            try:
                asyncio.run(self.async_client.close())
            except RuntimeError:
                pass

    async def aclose(self) -> None:
        """Close both the sync and async HTTP clients this ingester owns."""
        if self._owns_clients:
            if self.client is not None:
                self.client.close()
            if self.async_client is not None:
                await self.async_client.close()

    def __enter__(self) -> TaguruIngester:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def __aenter__(self) -> TaguruIngester:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
