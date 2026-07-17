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
    labeled_document,
    merge,
    parse_model_output,
    render_batch,
    reparse_batch,
    split_paragraphs,
    system_prompt,
    user_message,
)


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
        llm: Any LangChain chat model; asked for a single JSON object (one
            corrective turn on a malformed answer, mirroring taguru extract).
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
        include_passage: Store the verbatim document as the batch's passage
            (paragraph locators are stripped when off, matching extract).
        chunk_bytes: Prompt-input chunk cap; the stored passage is never
            chunked.
        vocabulary_cap: How many existing labels to seed the prompt with.
        refresh_embeddings: Best-effort ``embeddings/refresh`` after each
            call (501 = not configured is ignored; 502 lands as a warning).
        raise_on_error: ``ingest_documents`` raises on the first failed
            document instead of reporting it in its outcome.
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
        include_passage: bool = True,
        chunk_bytes: int = CHUNK_BYTES,
        vocabulary_cap: int = VOCABULARY_CAP,
        refresh_embeddings: bool = True,
        raise_on_error: bool = False,
    ) -> None:
        if create_context and context_description is None:
            raise ValueError("create_context=True requires context_description")
        if questions is not None and not 0 <= questions <= MAX_QUESTIONS_PER_PARAGRAPH:
            raise ValueError(f"questions must be between 0 and {MAX_QUESTIONS_PER_PARAGRAPH}")
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
        self.include_passage = include_passage
        self.chunk_bytes = chunk_bytes
        self.vocabulary_cap = vocabulary_cap
        self.refresh_embeddings = refresh_embeddings
        self.raise_on_error = raise_on_error

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
        self, messages: list[BaseMessage], content: str, error: ValueError
    ) -> list[BaseMessage]:
        return [
            *messages,
            AIMessage(content=content),
            HumanMessage(
                content=(
                    f"That was not the single JSON object asked for ({error}). "
                    "Answer again with only the JSON object."
                )
            ),
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

    # -- sync ----------------------------------------------------------------

    def ingest_text(self, text: str, *, source: str, dry_run: bool = False) -> IngestOutcome:
        """Ingest one text under one source id. Raises on failure (unlike
        ``ingest_documents``, there is no "continue with the rest" here)."""
        if self.client is None:
            raise ValueError("this ingester was built with only an async client")
        outcome = IngestOutcome(source=source, ok=False)
        if self.include_passage and len(text.encode("utf-8")) > MAX_PASSAGE_BYTES:
            raise ValueError(f"document exceeds the {MAX_PASSAGE_BYTES}-byte passage cap")

        vocabulary = self._fetch_vocabulary()
        system = system_prompt(vocabulary, self.questions)
        chunks = chunk(labeled_document(text, self.chunk_bytes), self.chunk_bytes)
        outcome.chunks = len(chunks)

        outputs: list[ModelOutput] = []
        for index, piece in enumerate(chunks):
            user = user_message(source, index, len(chunks), piece)
            messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            output = None
            last_error: ValueError | None = None
            for _attempt in range(2):
                response = self.llm.invoke(messages)
                outcome.llm_calls += 1
                content = self._content_text(response)
                try:
                    output = parse_model_output(content)
                    break
                except ValueError as error:
                    last_error = error
                    messages = self._corrective_turn(messages, content, error)
            if output is None:
                raise ValueError(f"the model would not produce the JSON object: {last_error}")
            outputs.append(output)

        ndjson = self._build_batch(source, text, outputs, outcome)
        outcome.ndjson = ndjson
        if dry_run:
            outcome.ok = True
            return outcome

        applied = self.client.import_batches(ndjson)
        self._record(outcome, applied.batches[0])
        outcome.ok = True

        if self.refresh_embeddings:
            try:
                self.client.context(self.context).refresh_embeddings()
            except EmbeddingUnavailableError as error:
                if error.reason == "provider_error":
                    outcome.embeddings_refresh_warning = error.message
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

        vocabulary = await self._afetch_vocabulary()
        system = system_prompt(vocabulary, self.questions)
        chunks = chunk(labeled_document(text, self.chunk_bytes), self.chunk_bytes)
        outcome.chunks = len(chunks)

        outputs: list[ModelOutput] = []
        for index, piece in enumerate(chunks):
            user = user_message(source, index, len(chunks), piece)
            messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            output = None
            last_error: ValueError | None = None
            for _attempt in range(2):
                response = await self.llm.ainvoke(messages)
                outcome.llm_calls += 1
                content = self._content_text(response)
                try:
                    output = parse_model_output(content)
                    break
                except ValueError as error:
                    last_error = error
                    messages = self._corrective_turn(messages, content, error)
            if output is None:
                raise ValueError(f"the model would not produce the JSON object: {last_error}")
            outputs.append(output)

        ndjson = self._build_batch(source, text, outputs, outcome)
        outcome.ndjson = ndjson
        if dry_run:
            outcome.ok = True
            return outcome

        applied = await self.async_client.import_batches(ndjson)
        self._record(outcome, applied.batches[0])
        outcome.ok = True

        if self.refresh_embeddings:
            try:
                await self.async_client.context(self.context).refresh_embeddings()
            except EmbeddingUnavailableError as error:
                if error.reason == "provider_error":
                    outcome.embeddings_refresh_warning = error.message
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
