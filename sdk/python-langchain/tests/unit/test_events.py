"""TaguruIngester.on_event: per-attempt progress/diagnostic events."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from taguru import AsyncTaguru, Taguru

from taguru_langchain.events import (
    AttemptFailed,
    AttemptStarted,
    ChunkCompleted,
    DocumentStarted,
    EmbeddingRefreshCompleted,
    EmbeddingRefreshWarning,
    IngestEvent,
)

from .conftest import FakeServer
from .test_ingester import DOC_TEXT, MODEL_ANSWER, make_ingester_with_messages

SUCCESSFUL_KINDS = [
    "document_started",
    "chunk_started",
    "attempt_started",
    "chunk_completed",
    "import_started",
    "import_completed",
    "embedding_refresh_started",
    "embedding_refresh_completed",
]


def test_successful_first_attempt_emits_no_attempt_failed(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=events.append
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")

    assert outcome.ok
    assert [event.kind for event in events] == SUCCESSFUL_KINDS

    document_started = events[0]
    assert isinstance(document_started, DocumentStarted)
    assert document_started.source == "docs/aomine.md"
    assert document_started.text_bytes == len(DOC_TEXT.encode("utf-8"))

    chunk_completed = next(e for e in events if isinstance(e, ChunkCompleted))
    assert chunk_completed.llm_calls == 1
    # Raw, pre-merge/dedup proposal counts (MODEL_ANSWER has 4 associations,
    # one duplicate and one invalid, that only get dropped during merge()).
    assert chunk_completed.associations_proposed == 4
    assert chunk_completed.aliases_proposed == 1
    assert chunk_completed.questions_proposed == 1

    refresh_completed = next(e for e in events if isinstance(e, EmbeddingRefreshCompleted))
    assert refresh_completed.configured is False  # FakeServer defaults to 501
    assert refresh_completed.source == "docs/aomine.md"


async def test_successful_first_attempt_async_matches_sync(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=events.append
    )
    outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")

    assert outcome.ok
    assert [event.kind for event in events] == SUCCESSFUL_KINDS


def test_corrective_success_emits_attempt_failed_with_provider_metadata(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    events: list[IngestEvent] = []
    malformed = AIMessage(
        content="not json",
        response_metadata={"done_reason": "length"},
        usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )
    ingester, _llm = make_ingester_with_messages(
        sync_client,
        async_client,
        [malformed, AIMessage(content=MODEL_ANSWER)],
        on_event=events.append,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok

    failed = [event for event in events if isinstance(event, AttemptFailed)]
    assert len(failed) == 1
    assert failed[0].attempt == 1
    assert failed[0].max_attempts == 2
    assert failed[0].parse_error != ""
    metadata = failed[0].provider_metadata
    assert metadata is not None
    assert metadata.finish_reason == "length"
    assert metadata.input_tokens == 10
    assert metadata.output_tokens == 20
    assert metadata.total_tokens == 30

    started = [event for event in events if isinstance(event, AttemptStarted)]
    assert [event.attempt for event in started] == [1, 2]


async def test_acorrective_success_emits_attempt_failed(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client,
        async_client,
        [AIMessage(content="not json"), AIMessage(content=MODEL_ANSWER)],
        on_event=events.append,
    )
    outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert sum(1 for event in events if isinstance(event, AttemptFailed)) == 1


def test_terminal_parse_failure_emits_two_attempt_failed(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client,
        async_client,
        [AIMessage(content="nope"), AIMessage(content="still nope")],
        on_event=events.append,
    )
    with pytest.raises(ValueError, match="would not produce the JSON object"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")

    failed = [event for event in events if isinstance(event, AttemptFailed)]
    assert [event.attempt for event in failed] == [1, 2]
    # The chunk never completed, so no import/refresh events followed.
    assert not any(event.kind == "chunk_completed" for event in events)
    assert not any(event.kind == "import_started" for event in events)


def test_embedding_refresh_provider_error_emits_warning(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.embeddings_refresh_status = 502
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=events.append
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.embeddings_refresh_warning is not None

    warned = [event for event in events if isinstance(event, EmbeddingRefreshWarning)]
    assert len(warned) == 1
    assert warned[0].message == outcome.embeddings_refresh_warning
    assert warned[0].source == "docs/aomine.md"
    assert not any(event.kind == "embedding_refresh_completed" for event in events)


async def test_aembedding_refresh_provider_error_emits_warning(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.embeddings_refresh_status = 502
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=events.append
    )
    outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.embeddings_refresh_warning is not None
    assert any(isinstance(event, EmbeddingRefreshWarning) for event in events)


def test_embedding_refresh_success_reports_counts(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.embeddings_refresh_status = 200
    fake_server.embeddings_refresh_result = {"embedded": 3, "total": 5}
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=events.append
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok

    completed = [event for event in events if isinstance(event, EmbeddingRefreshCompleted)]
    assert len(completed) == 1
    assert completed[0].configured is True
    assert completed[0].embedded == 3
    assert completed[0].total == 5
    assert completed[0].source == "docs/aomine.md"


def test_ingest_documents_refresh_events_carry_the_right_source(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The regression this field exists for: ingest_documents() ingests
    multiple documents through the same ingester, so without a source on
    the refresh events a caller can't tell which document just finished."""
    events: list[IngestEvent] = []
    ingester, _llm = make_ingester_with_messages(
        sync_client,
        async_client,
        [AIMessage(content=MODEL_ANSWER), AIMessage(content=MODEL_ANSWER)],
        on_event=events.append,
    )
    outcomes = ingester.ingest_documents(
        [
            Document(page_content=DOC_TEXT, metadata={"source": "docs/aomine.md"}),
            Document(page_content=DOC_TEXT, metadata={"source": "docs/other.md"}),
        ]
    )
    assert [outcome.ok for outcome in outcomes] == [True, True]

    completed = [event for event in events if isinstance(event, EmbeddingRefreshCompleted)]
    assert [event.source for event in completed] == ["docs/aomine.md", "docs/other.md"]


def test_on_event_exception_warns_but_does_not_break_ingest(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    def bad_callback(event: IngestEvent) -> None:
        raise RuntimeError("callback bug")

    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=bad_callback
    )
    with pytest.warns(RuntimeWarning, match="callback bug"):
        outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert len(fake_server.imported) == 1


async def test_aon_event_exception_warns_but_does_not_break_ingest(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    def bad_callback(event: IngestEvent) -> None:
        raise RuntimeError("callback bug")

    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)], on_event=bad_callback
    )
    with pytest.warns(RuntimeWarning, match="callback bug"):
        outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert len(fake_server.imported) == 1


def test_no_callback_means_no_overhead_surprises(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The default (on_event=None) path — existing callers see no behavior
    change, confirming _emit's no-op short-circuit."""
    ingester, _llm = make_ingester_with_messages(
        sync_client, async_client, [AIMessage(content=MODEL_ANSWER)]
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
