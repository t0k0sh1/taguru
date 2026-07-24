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
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel
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
    MODEL_OUTPUT_JSON_SCHEMA,
    VOCABULARY_CAP,
    InvalidFault,
    ItemRules,
    ModelOutput,
    SyntaxFault,
    chunk,
    corrective_assistant_turn_content,
    corrective_message,
    corrective_validation_message,
    cross_output_issues,
    effective_item_rules,
    empty_answer_diagnosis,
    evaluate_answer,
    indicates_length_limit,
    indicates_refusal,
    interpret_model_output,
    is_empty_answer,
    labeled_document,
    merge,
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
    """Under the strict default this counts only merge()'s policy trims
    (per-paragraph question-cap overflow, a volunteered question when
    none was requested) — a business-rule-invalid item is corrected or
    fails the source before merge() ever runs (issue #180). Under
    ``lossy=True`` it is the old drop-and-proceed tally: every item
    merge() silently discarded."""
    llm_calls: int = 0
    chunks: int = 0
    correction_attempts: int = 0
    """How many corrective turns (Stage 1 syntax/validity retries plus any
    Stage 2 cross-chunk alias correction) this ingest needed — 0 means
    every chunk's first answer was accepted as-is."""
    lossless_repairs: list[str] = field(default_factory=list)
    """Labels of automatic, information-preserving JSON repairs applied
    across every accepted answer (e.g. ``"trailing_comma"``, ``"bom"``,
    ``"code_fence"``, ``"braces_slice"`` — see ``candidate_json``). Empty
    when no repair was ever needed."""
    error: str | None = None
    embeddings_refresh_warning: str | None = None


# -- one attempt's §7-style classification (issue #180, mirrors extract.rs's ------
# -- classify_attempt / evaluate_answer) ------------------------------------------

_AttemptKind = Literal["valid", "length_limited", "refusal", "empty", "syntax", "invalid"]


@dataclass
class _Attempt:
    """One model call's outcome, classified from provider metadata BEFORE
    any parse-level interpretation — a `length`-terminated answer is
    length-limited even when its prefix happens to parse, since a valid
    prefix of a cut-off extraction is exactly the "deleted-subset called
    complete" ADR 0001 forbids."""

    kind: _AttemptKind
    content: str
    metadata: ProviderMetadata | None
    output: ModelOutput | None = None
    repairs: list[str] = field(default_factory=list)
    error: str | None = None
    issues: list[str] | None = None


def _classify_text(
    content: str, metadata: ProviderMetadata | None, rules: ItemRules | None
) -> _Attempt:
    finish_reason = metadata.finish_reason if metadata else None
    if indicates_length_limit(finish_reason):
        return _Attempt(
            kind="length_limited",
            content=content,
            metadata=metadata,
            error="the answer hit the provider's output limit",
        )
    if indicates_refusal(finish_reason):
        return _Attempt(kind="refusal", content=content, metadata=metadata, error=finish_reason)
    if is_empty_answer(content):
        return _Attempt(
            kind="empty", content=content, metadata=metadata, error=empty_answer_diagnosis()
        )
    try:
        output, repairs = evaluate_answer(content, rules)
    except InvalidFault as fault:
        return _Attempt(kind="invalid", content=content, metadata=metadata, issues=fault.issues)
    except SyntaxFault as fault:
        return _Attempt(kind="syntax", content=content, metadata=metadata, error=str(fault))
    return _Attempt(
        kind="valid", content=content, metadata=metadata, output=output, repairs=repairs
    )


def _classify_structured(
    content: str,
    metadata: ProviderMetadata | None,
    parsed: Any,
    parsing_error: Any,
    rules: ItemRules | None,
) -> _Attempt:
    """The structured-output twin of ``_classify_text``: the provider has
    already parsed (or failed to parse) its own tool-call args against
    ``MODEL_OUTPUT_JSON_SCHEMA``, so there is no raw JSON text here to
    lossless-repair, and ``content`` is conventionally empty for a
    tool-calling response — ``is_empty_answer`` would misfire on it, so
    emptiness is judged from ``parsed`` instead."""
    finish_reason = metadata.finish_reason if metadata else None
    if indicates_length_limit(finish_reason):
        return _Attempt(
            kind="length_limited",
            content=content,
            metadata=metadata,
            error="the answer hit the provider's output limit",
        )
    if indicates_refusal(finish_reason):
        return _Attempt(kind="refusal", content=content, metadata=metadata, error=finish_reason)
    if parsed is None:
        error = (
            parsing_error
            if isinstance(parsing_error, ValueError)
            else ValueError(str(parsing_error))
        )
        return _Attempt(kind="syntax", content=content, metadata=metadata, error=str(error))
    value = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
    output, issues = interpret_model_output(value, effective_item_rules(rules))
    if rules is not None and issues:
        return _Attempt(kind="invalid", content=content, metadata=metadata, issues=issues)
    return _Attempt(kind="valid", content=content, metadata=metadata, output=output)


def _diagnosis(result: _Attempt) -> str:
    """The raw, unwrapped human-readable reason one attempt failed — used
    for ``AttemptFailed.parse_error``, the corrective ask, and (for
    ``"invalid"``/``"empty"``) the final failure message verbatim."""
    if result.kind == "invalid":
        assert result.issues is not None
        return (
            f"the answer left {len(result.issues)} invalid item(s) uncorrected: "
            f"{'; '.join(result.issues)}"
        )
    return result.error or ""


def _final_message(result: _Attempt) -> str:
    """The message raised when a chunk's attempts are exhausted without a
    valid answer. ``"invalid"``/``"empty"`` stay unwrapped (extract.rs's
    ``AttemptOutcome::Invalid``/``Empty`` never get the generic wrapper
    either); ``"syntax"``/``"length_limited"`` get the same
    "the model would not produce the JSON object" wrapper today's
    (pre-#180) behavior always used."""
    diagnosis = _diagnosis(result)
    if result.kind in ("invalid", "empty"):
        return diagnosis
    return f"the model would not produce the JSON object: {diagnosis}"


def _corrective_ask(result: _Attempt, fact_budget: int) -> str:
    """The next attempt's user-facing ask, addressed to whichever kind of
    failure ``result`` was. ``"empty"`` reuses ``corrective_message``'s
    ordinary (non-length-limited) text with the empty diagnosis as its
    ``parse_error`` — the same ask extract.rs's ladder gives an empty
    answer's one bounded correction."""
    if result.kind == "invalid":
        assert result.issues is not None
        return corrective_validation_message(result.issues)
    length_limited = result.kind == "length_limited"
    return corrective_message(result.error or "", length_limited, fact_budget)


def _cross_chunk_failure_message(label: str, result: _Attempt) -> str:
    """The Stage 2 (cross-chunk alias correction) terminal message for one
    offending chunk's non-valid reply — mirrors extract.rs's
    ``correct_cross_output_issues`` per-kind texts verbatim."""
    if result.kind == "length_limited":
        return (
            f"{label}: the cross-chunk alias correction was cut off at the output limit — "
            "failing the source rather than importing a truncated correction"
        )
    if result.kind == "refusal":
        return (
            f"{label}: the provider refused the cross-chunk alias correction "
            f"(finish_reason {result.error})"
        )
    if result.kind == "empty":
        return f"{label}: {result.error}"
    if result.kind == "invalid":
        assert result.issues is not None
        return (
            f"{label}: the cross-chunk alias correction still left {len(result.issues)} "
            f"invalid item(s) uncorrected: {'; '.join(result.issues)}"
        )
    return (
        f"{label}: the cross-chunk alias correction was not the JSON object asked for "
        f"({result.error})"
    )


@dataclass
class _ChunkRecord:
    """One chunk's accepted output, plus everything Stage 2's single
    targeted corrective turn needs to rebuild THAT chunk's own
    conversation (never the whole document's) if `cross_output_issues`
    flags it: extract.rs's ``ChunkOutput``."""

    output: ModelOutput
    chunk_index: int
    user: str
    answer: str


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
        structured_output: Ask ``llm`` for JSON-schema-constrained generation
            — ``llm.with_structured_output(MODEL_OUTPUT_JSON_SCHEMA,
            include_raw=True)`` — instead of a free-text answer parsed
            afterward (default ``False``; strictly opt-in, matching the
            Rust and TypeScript producers — see issue #185). Provider/model
            dependent: a chat model that cannot bind tools raises out of
            this constructor immediately, before any document is ingested,
            rather than surfacing later as a per-attempt failure. Either
            way the result still goes through the same lenient validation
            walk and merge()'s business-rule checks (byte caps, weight
            bounds, an in-range paragraph index, ...) — a schema only
            narrows what shape a well-behaved provider can return, it does
            not replace validation.
        include_passage: Store the verbatim document as the batch's passage
            (paragraph locators are stripped when off, matching extract).
        chunk_bytes: Prompt-input chunk cap; the stored passage is never
            chunked.
        vocabulary_cap: How many existing labels to seed the prompt with.
        refresh_embeddings: Best-effort ``embeddings/refresh`` after each
            call (501 = not configured is ignored; 502 lands as a warning).
        raise_on_error: ``ingest_documents`` raises on the first failed
            document instead of reporting it in its outcome.
        lossy: Restore the pre-issue-#180 drop-and-proceed behavior: a
            business-rule-invalid item (bad weight, dangling alias,
            out-of-range question, ...) is silently dropped and the source
            still reports success, exactly like ``merge()`` always did.
            Default ``False`` (ADR 0001 §8's never-silent-drop default):
            an invalid item instead earns one targeted, path-addressed
            corrective turn, and the source fails outright (no ``/import``
            call) if it is still invalid afterward. This is the one
            deliberate, opt-out-only behavior change issue #180 makes —
            see ``IngestOutcome.invalid_dropped``.
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
        structured_output: bool = False,
        include_passage: bool = True,
        chunk_bytes: int = CHUNK_BYTES,
        vocabulary_cap: int = VOCABULARY_CAP,
        refresh_embeddings: bool = True,
        raise_on_error: bool = False,
        lossy: bool = False,
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
        self.structured_output = structured_output
        self._structured_llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] | None = (
            llm.with_structured_output(MODEL_OUTPUT_JSON_SCHEMA, include_raw=True)
            if structured_output
            else None
        )
        self.include_passage = include_passage
        self.chunk_bytes = chunk_bytes
        self.vocabulary_cap = vocabulary_cap
        self.refresh_embeddings = refresh_embeddings
        self.raise_on_error = raise_on_error
        self.lossy = lossy
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
        self, base_messages: list[BaseMessage], prior_answer: str, ask: str
    ) -> list[BaseMessage]:
        """Rebuilds from the system/user base plus only the most recent bad
        turn — never the whole history — so ``corrective_context_bytes``
        bounds every retry alike, not just the first. At the all-defaults
        policy, with a normal (non-length-limited, syntactically bad)
        answer, this reproduces the previous fixed implementation's
        request bodies exactly. Shared by the Stage 1 per-chunk loop and
        Stage 2's cross-chunk alias correction."""
        return [
            *base_messages,
            AIMessage(
                content=corrective_assistant_turn_content(
                    prior_answer, self.corrective_context_bytes
                )
            ),
            HumanMessage(content=ask),
        ]

    def _item_rules(self, paragraph_count: int) -> ItemRules | None:
        """``None`` under ``lossy=True`` — ``evaluate_answer``/
        ``interpret_model_output`` then parse leniently and discard
        whatever they'd have flagged, reproducing pre-issue-#180 behavior
        byte for byte (``merge()`` alone decides what survives)."""
        if self.lossy:
            return None
        return ItemRules(paragraph_count=paragraph_count, questions_requested=self.questions > 0)

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
        paragraph_count = len(split_paragraphs(text))
        rules = self._item_rules(paragraph_count)
        chunks = chunk(labeled_document(text, self.chunk_bytes), self.chunk_bytes)
        outcome.chunks = len(chunks)

        records: list[_ChunkRecord] = []
        for index, piece in enumerate(chunks):
            chunk_started_at = time.monotonic()
            self._emit(ChunkStarted(source=source, index=index, total=len(chunks)))
            user = user_message(source, index, len(chunks), piece)
            base_messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            record: _ChunkRecord | None = None
            pending_ask: str | None = None
            prior_bad_answer: str | None = None
            last_diagnosis = ""
            empty_corrected = False
            chunk_llm_calls = 0
            for attempt in range(1, self.max_attempts + 1):
                messages = (
                    base_messages
                    if prior_bad_answer is None
                    else self._corrective_turn(base_messages, prior_bad_answer, pending_ask or "")
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
                result = self._attempt_once(messages, rules)
                outcome.llm_calls += 1
                chunk_llm_calls += 1

                if result.kind == "valid":
                    assert result.output is not None
                    record = _ChunkRecord(
                        output=result.output, chunk_index=index, user=user, answer=result.content
                    )
                    outcome.lossless_repairs.extend(result.repairs)
                    break

                if result.kind == "refusal":
                    message = (
                        f"the provider refused this content (finish_reason {result.error}) — "
                        "a policy refusal is terminal; no corrective turn can change it"
                    )
                    self._emit(
                        AttemptFailed(
                            source=source,
                            chunk_index=index,
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            parse_error=message,
                            elapsed_seconds=time.monotonic() - attempt_started_at,
                            provider_metadata=result.metadata,
                            length_limited=False,
                        )
                    )
                    raise ValueError(message)

                if result.kind == "empty" and empty_corrected:
                    diagnosis = _diagnosis(result)
                    self._emit(
                        AttemptFailed(
                            source=source,
                            chunk_index=index,
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            parse_error=diagnosis,
                            elapsed_seconds=time.monotonic() - attempt_started_at,
                            provider_metadata=result.metadata,
                            length_limited=False,
                        )
                    )
                    raise ValueError(diagnosis)
                if result.kind == "empty":
                    empty_corrected = True

                last_diagnosis = _final_message(result)
                self._emit(
                    AttemptFailed(
                        source=source,
                        chunk_index=index,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                        parse_error=_diagnosis(result),
                        elapsed_seconds=time.monotonic() - attempt_started_at,
                        provider_metadata=result.metadata,
                        length_limited=(result.kind == "length_limited"),
                        validation_issues=result.issues,
                    )
                )
                outcome.correction_attempts += 1
                pending_ask = _corrective_ask(result, self.fact_budget)
                prior_bad_answer = result.content

            if record is None:
                raise ValueError(last_diagnosis)
            records.append(record)
            self._emit(
                ChunkCompleted(
                    source=source,
                    index=index,
                    total=len(chunks),
                    associations_proposed=len(record.output.associations),
                    aliases_proposed=len(record.output.aliases),
                    questions_proposed=len(record.output.questions),
                    llm_calls=chunk_llm_calls,
                    elapsed_seconds=time.monotonic() - chunk_started_at,
                )
            )

        if not self.lossy:
            self._correct_cross_chunk_issues(source, system, records, rules, len(chunks), outcome)

        ndjson = self._build_batch(source, text, [record.output for record in records], outcome)
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
                refresh_result = self.client.context(self.context).refresh_embeddings()
                self._emit(
                    EmbeddingRefreshCompleted(
                        source=source,
                        configured=True,
                        embedded=refresh_result.embedded,
                        total=refresh_result.total,
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

    def _attempt_once(self, messages: list[BaseMessage], rules: ItemRules | None) -> _Attempt:
        """Runs one model call for one attempt and classifies it (issue
        #180's §7-style state machine: length -> refusal -> empty ->
        syntax/invalid/valid). Goes through the ``with_structured_output()``
        pipeline built in ``__init__`` when ``structured_output`` is on,
        else today's plain ``invoke()`` + the free-text validation walk."""
        if self._structured_llm is not None:
            result = self._structured_llm.invoke(messages)
            assert isinstance(result, dict)
            raw = result["raw"]
            content = self._content_text(raw)
            metadata = self._provider_metadata(raw)
            return _classify_structured(
                content, metadata, result["parsed"], result["parsing_error"], rules
            )
        response = self.llm.invoke(messages)
        content = self._content_text(response)
        metadata = self._provider_metadata(response)
        return _classify_text(content, metadata, rules)

    def _correct_cross_chunk_issues(
        self,
        source: str,
        system: str,
        records: list[_ChunkRecord],
        rules: ItemRules | None,
        chunk_total: int,
        outcome: IngestOutcome,
    ) -> None:
        """Issue #180 Stage 2: one targeted corrective turn per output
        ``cross_output_issues`` flags, rebuilding THAT chunk's own
        conversation base (never the whole document's) and replaying its
        own final answer as the prior bad turn. Bounded to exactly one
        extra call per offending chunk regardless of ``max_attempts``: a
        still-invalid, still-cross-conflicting, length-limited, refused,
        or empty reply fails the source outright — Stage 2 never loops a
        second round. Mirrors extract.rs's ``correct_cross_output_issues``."""
        for record_index, issues in cross_output_issues([r.output for r in records]):
            record = records[record_index]
            label = f"chunk {record.chunk_index + 1}/{chunk_total}"
            messages = self._corrective_turn(
                [SystemMessage(content=system), HumanMessage(content=record.user)],
                record.answer,
                corrective_validation_message(issues),
            )
            self._emit(
                AttemptStarted(
                    source=source,
                    chunk_index=record.chunk_index,
                    attempt=1,
                    max_attempts=1,
                    stage="cross_chunk",
                )
            )
            attempt_started_at = time.monotonic()
            result = self._attempt_once(messages, rules)
            outcome.llm_calls += 1
            outcome.correction_attempts += 1
            if result.kind == "valid":
                assert result.output is not None
                records[record_index] = _ChunkRecord(
                    output=result.output,
                    chunk_index=record.chunk_index,
                    user=record.user,
                    answer=result.content,
                )
                outcome.lossless_repairs.extend(result.repairs)
                continue
            message = _cross_chunk_failure_message(label, result)
            self._emit(
                AttemptFailed(
                    source=source,
                    chunk_index=record.chunk_index,
                    attempt=1,
                    max_attempts=1,
                    parse_error=message,
                    elapsed_seconds=time.monotonic() - attempt_started_at,
                    provider_metadata=result.metadata,
                    length_limited=(result.kind == "length_limited"),
                    stage="cross_chunk",
                    validation_issues=result.issues,
                )
            )
            raise ValueError(message)

        # Re-check rather than trust the single corrective turn blindly: a
        # correction can rename an association another chunk's alias
        # depended on, introducing a FRESH cross-chunk issue. This is the
        # bounded re-check, not a second round — any issue here fails the
        # source.
        recheck = cross_output_issues([r.output for r in records])
        if recheck:
            record_index, issues = recheck[0]
            chunk_index = records[record_index].chunk_index
            raise ValueError(
                f"chunk {chunk_index + 1}/{chunk_total}: still has {len(issues)} cross-chunk "
                f"alias issue(s) after correction: {'; '.join(issues)}"
            )

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
        paragraph_count = len(split_paragraphs(text))
        rules = self._item_rules(paragraph_count)
        chunks = chunk(labeled_document(text, self.chunk_bytes), self.chunk_bytes)
        outcome.chunks = len(chunks)

        records: list[_ChunkRecord] = []
        for index, piece in enumerate(chunks):
            chunk_started_at = time.monotonic()
            self._emit(ChunkStarted(source=source, index=index, total=len(chunks)))
            user = user_message(source, index, len(chunks), piece)
            base_messages: list[BaseMessage] = [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
            record: _ChunkRecord | None = None
            pending_ask: str | None = None
            prior_bad_answer: str | None = None
            last_diagnosis = ""
            empty_corrected = False
            chunk_llm_calls = 0
            for attempt in range(1, self.max_attempts + 1):
                messages = (
                    base_messages
                    if prior_bad_answer is None
                    else self._corrective_turn(base_messages, prior_bad_answer, pending_ask or "")
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
                result = await self._aattempt_once(messages, rules)
                outcome.llm_calls += 1
                chunk_llm_calls += 1

                if result.kind == "valid":
                    assert result.output is not None
                    record = _ChunkRecord(
                        output=result.output, chunk_index=index, user=user, answer=result.content
                    )
                    outcome.lossless_repairs.extend(result.repairs)
                    break

                if result.kind == "refusal":
                    message = (
                        f"the provider refused this content (finish_reason {result.error}) — "
                        "a policy refusal is terminal; no corrective turn can change it"
                    )
                    self._emit(
                        AttemptFailed(
                            source=source,
                            chunk_index=index,
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            parse_error=message,
                            elapsed_seconds=time.monotonic() - attempt_started_at,
                            provider_metadata=result.metadata,
                            length_limited=False,
                        )
                    )
                    raise ValueError(message)

                if result.kind == "empty" and empty_corrected:
                    diagnosis = _diagnosis(result)
                    self._emit(
                        AttemptFailed(
                            source=source,
                            chunk_index=index,
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            parse_error=diagnosis,
                            elapsed_seconds=time.monotonic() - attempt_started_at,
                            provider_metadata=result.metadata,
                            length_limited=False,
                        )
                    )
                    raise ValueError(diagnosis)
                if result.kind == "empty":
                    empty_corrected = True

                last_diagnosis = _final_message(result)
                self._emit(
                    AttemptFailed(
                        source=source,
                        chunk_index=index,
                        attempt=attempt,
                        max_attempts=self.max_attempts,
                        parse_error=_diagnosis(result),
                        elapsed_seconds=time.monotonic() - attempt_started_at,
                        provider_metadata=result.metadata,
                        length_limited=(result.kind == "length_limited"),
                        validation_issues=result.issues,
                    )
                )
                outcome.correction_attempts += 1
                pending_ask = _corrective_ask(result, self.fact_budget)
                prior_bad_answer = result.content

            if record is None:
                raise ValueError(last_diagnosis)
            records.append(record)
            self._emit(
                ChunkCompleted(
                    source=source,
                    index=index,
                    total=len(chunks),
                    associations_proposed=len(record.output.associations),
                    aliases_proposed=len(record.output.aliases),
                    questions_proposed=len(record.output.questions),
                    llm_calls=chunk_llm_calls,
                    elapsed_seconds=time.monotonic() - chunk_started_at,
                )
            )

        if not self.lossy:
            await self._acorrect_cross_chunk_issues(
                source, system, records, rules, len(chunks), outcome
            )

        ndjson = self._build_batch(source, text, [record.output for record in records], outcome)
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
                refresh_result = await self.async_client.context(self.context).refresh_embeddings()
                self._emit(
                    EmbeddingRefreshCompleted(
                        source=source,
                        configured=True,
                        embedded=refresh_result.embedded,
                        total=refresh_result.total,
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

    async def _aattempt_once(
        self, messages: list[BaseMessage], rules: ItemRules | None
    ) -> _Attempt:
        """Async twin of ``_attempt_once``."""
        if self._structured_llm is not None:
            result = await self._structured_llm.ainvoke(messages)
            assert isinstance(result, dict)
            raw = result["raw"]
            content = self._content_text(raw)
            metadata = self._provider_metadata(raw)
            return _classify_structured(
                content, metadata, result["parsed"], result["parsing_error"], rules
            )
        response = await self.llm.ainvoke(messages)
        content = self._content_text(response)
        metadata = self._provider_metadata(response)
        return _classify_text(content, metadata, rules)

    async def _acorrect_cross_chunk_issues(
        self,
        source: str,
        system: str,
        records: list[_ChunkRecord],
        rules: ItemRules | None,
        chunk_total: int,
        outcome: IngestOutcome,
    ) -> None:
        """Async twin of ``_correct_cross_chunk_issues``."""
        for record_index, issues in cross_output_issues([r.output for r in records]):
            record = records[record_index]
            label = f"chunk {record.chunk_index + 1}/{chunk_total}"
            messages = self._corrective_turn(
                [SystemMessage(content=system), HumanMessage(content=record.user)],
                record.answer,
                corrective_validation_message(issues),
            )
            self._emit(
                AttemptStarted(
                    source=source,
                    chunk_index=record.chunk_index,
                    attempt=1,
                    max_attempts=1,
                    stage="cross_chunk",
                )
            )
            attempt_started_at = time.monotonic()
            result = await self._aattempt_once(messages, rules)
            outcome.llm_calls += 1
            outcome.correction_attempts += 1
            if result.kind == "valid":
                assert result.output is not None
                records[record_index] = _ChunkRecord(
                    output=result.output,
                    chunk_index=record.chunk_index,
                    user=record.user,
                    answer=result.content,
                )
                outcome.lossless_repairs.extend(result.repairs)
                continue
            message = _cross_chunk_failure_message(label, result)
            self._emit(
                AttemptFailed(
                    source=source,
                    chunk_index=record.chunk_index,
                    attempt=1,
                    max_attempts=1,
                    parse_error=message,
                    elapsed_seconds=time.monotonic() - attempt_started_at,
                    provider_metadata=result.metadata,
                    length_limited=(result.kind == "length_limited"),
                    stage="cross_chunk",
                    validation_issues=result.issues,
                )
            )
            raise ValueError(message)

        recheck = cross_output_issues([r.output for r in records])
        if recheck:
            record_index, issues = recheck[0]
            chunk_index = records[record_index].chunk_index
            raise ValueError(
                f"chunk {chunk_index + 1}/{chunk_total}: still has {len(issues)} cross-chunk "
                f"alias issue(s) after correction: {'; '.join(issues)}"
            )

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
