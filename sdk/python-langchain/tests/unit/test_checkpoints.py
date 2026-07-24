"""TaguruIngester checkpoint/resume (issue #211, the Python twin of #210):
durable per-chunk checkpoints, cooperative stop, and resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from taguru import AsyncTaguru, ServerError, Taguru

from taguru_langchain import TaguruIngester
from taguru_langchain.checkpoints import CheckpointStore, FilesystemCheckpointStore

from .conftest import FakeServer
from .test_ingester import (
    CHUNK1_ANSWER,
    CHUNK2_CORRECTED_ANSWER,
    CHUNK2_SHADOWING_ANSWER,
    CROSS_CHUNK_BYTES,
    DOC_TEXT,
    MODEL_ANSWER,
    RecordingFakeChatModel,
    ToolCallingFakeChatModel,
    make_ingester,
    make_ingester_with_messages,
)

# Paragraph 2 gets one extra ASCII character relative to DOC_TEXT — still
# well inside CROSS_CHUNK_BYTES's 40-byte cap for either paragraph, so it
# stays a clean 2-chunk document, but its content hash (and therefore its
# checkpoint fingerprint) differs.
CHANGED_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬xである。"


class RecordingCheckpointStore:
    """In-memory CheckpointStore recording every call in order, seedable
    with a prior run's bytes — the double every checkpoint/resume test
    below is built on."""

    def __init__(self, seed: dict[str, bytes] | None = None) -> None:
        self.state: dict[str, bytes] = dict(seed or {})
        self.log: list[tuple[str, str]] = []

    def load(self, source: str) -> bytes | None:
        self.log.append(("load", source))
        return self.state.get(source)

    def save(self, source: str, data: bytes) -> None:
        self.log.append(("save", source))
        self.state[source] = data

    def delete(self, source: str) -> None:
        self.log.append(("delete", source))
        self.state.pop(source, None)


class FailingLoadStore(RecordingCheckpointStore):
    def load(self, source: str) -> bytes | None:
        self.log.append(("load", source))
        raise OSError("simulated load failure")


class FailingSaveStore(RecordingCheckpointStore):
    def save(self, source: str, data: bytes) -> None:
        self.log.append(("save", source))
        raise OSError("simulated save failure")


class FailingDeleteStore(RecordingCheckpointStore):
    def delete(self, source: str) -> None:
        self.log.append(("delete", source))
        raise OSError("simulated delete failure")


class KillPartwayThroughChatModel(RecordingFakeChatModel):
    """Answers normally up to (but not including) call number
    ``kill_at_call``, then raises — simulating a process kill/crash
    mid-document (an arbitrary uncaught exception escaping ingest_text/
    aingest_text with no cleanup, unlike the ordinary attempt-exhaustion
    failure path)."""

    kill_at_call: int = 1
    calls: int = 0

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls > self.kill_at_call:
            raise RuntimeError("simulated process kill")
        return super()._generate(messages, *args, **kwargs)


def _build_ingester(
    sync_client: Taguru,
    async_client: AsyncTaguru,
    responses: list[str],
    *,
    checkpoint_store: CheckpointStore | None,
    checkpoint_model_id: str | None = "fake-model",
    **overrides: Any,
) -> tuple[TaguruIngester, RecordingFakeChatModel]:
    """Full-control ingester factory for checkpoint tests: unlike
    ``make_ingester`` (which hardcodes ``context``/``questions``), every
    field can be overridden — needed to exercise fingerprint mismatches on
    fields ``make_ingester`` fixes."""
    llm = RecordingFakeChatModel(responses=responses, seen_prompts=[])
    kwargs: dict[str, Any] = dict(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        checkpoint_store=checkpoint_store,
        checkpoint_model_id=checkpoint_model_id,
    )
    kwargs.update(overrides)
    return TaguruIngester(**kwargs), llm


def _unit_count(store: RecordingCheckpointStore, source: str) -> int:
    data = store.state.get(source)
    if data is None:
        return 0
    return len(json.loads(data)["units"])


def _seed_failed_run(
    sync_client: Taguru,
    async_client: AsyncTaguru,
    store: CheckpointStore,
    *,
    source: str = "docs/aomine.md",
    **overrides: Any,
) -> str:
    """Runs a 2-chunk document where chunk 0 succeeds and chunk 1 never
    produces valid JSON, leaving exactly one checkpointed unit behind —
    the Python twin of extract.rs's test helper
    ``setup_one_checkpointed_chunk_and_one_failure``. Returns the source id."""
    ingester, _llm = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "not json", "not json"],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
        **overrides,
    )
    with pytest.raises(ValueError):
        ingester.ingest_text(DOC_TEXT, source=source)
    return source


# -- reuse across a failed-then-resumed document ------------------------------


def test_checkpoint_reuses_a_completed_chunk_after_a_failed_document_without_recalling_the_model(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = _seed_failed_run(sync_client, async_client, store)
    assert _unit_count(store, source) == 1
    assert fake_server.imported == []

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = ingester2.ingest_text(DOC_TEXT, source=source)

    assert outcome.ok
    assert outcome.chunks_reused == 1
    assert len(llm2.seen_prompts) == 1
    # The one real call was for chunk index 1 (the second, never-cached
    # chunk) — chunk 0 was never re-asked.
    human_contents = [
        m.content for m in llm2.seen_prompts[0] if getattr(m, "type", None) == "human"
    ]
    assert any("part 2 of 2" in content for content in human_contents)
    assert len(fake_server.imported) == 1
    assert _unit_count(store, source) == 0  # cleared once the batch landed


async def test_acheckpoint_reuses_a_completed_chunk_without_recalling_the_model(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    ingester, _llm = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "not json", "not json"],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError):
        await ingester.aingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 1

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = await ingester2.aingest_text(DOC_TEXT, source=source)
    assert outcome.ok
    assert outcome.chunks_reused == 1
    assert len(llm2.seen_prompts) == 1
    assert len(fake_server.imported) == 1


def test_no_checkpoint_store_means_no_reuse_and_unchanged_behavior(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "not json", "not json"],
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    assert ingester.checkpoint_store is None
    with pytest.raises(ValueError):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert len(llm.seen_prompts) == 3

    ingester2, llm2 = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "not json", "not json"],
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError):
        ingester2.ingest_text(DOC_TEXT, source="docs/aomine.md")
    # No reduction whatsoever — nothing was ever cached without a store.
    assert len(llm2.seen_prompts) == 3


def test_a_changed_fact_budget_invalidates_checkpoints_even_though_content_is_unchanged(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = RecordingCheckpointStore()
    source = _seed_failed_run(sync_client, async_client, store)
    assert _unit_count(store, source) == 1

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
        fact_budget=3,
    )
    outcome = ingester2.ingest_text(DOC_TEXT, source=source)
    assert outcome.ok
    assert outcome.chunks_reused == 0
    assert len(llm2.seen_prompts) == 2  # both chunks re-extracted


def test_checkpoint_resumes_a_killed_multi_chunk_document_without_recalling_completed_chunks(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    killer = KillPartwayThroughChatModel(responses=[CHUNK1_ANSWER], seen_prompts=[], kill_at_call=1)
    ingester = TaguruIngester(
        context="sake",
        llm=killer,
        client=sync_client,
        async_client=async_client,
        questions=2,
        chunk_bytes=CROSS_CHUNK_BYTES,
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.raises(RuntimeError, match="simulated process kill"):
        ingester.ingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 1
    assert fake_server.imported == []

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = ingester2.ingest_text(DOC_TEXT, source=source)
    assert outcome.ok
    assert outcome.chunks_reused == 1
    assert len(llm2.seen_prompts) == 1
    assert len(fake_server.imported) == 1


# -- cooperative stop ----------------------------------------------------------


def test_cooperative_stop_between_chunks_saves_checkpoints_and_skips_import(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # let chunk 0 run, stop before chunk 1

    ingester, llm = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source=source, should_stop=should_stop)

    assert outcome.interrupted
    assert not outcome.ok
    assert outcome.ndjson is None
    assert len(llm.seen_prompts) == 1
    assert fake_server.imported == []
    assert _unit_count(store, source) == 1

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome2 = ingester2.ingest_text(DOC_TEXT, source=source)
    assert outcome2.ok
    assert outcome2.chunks_reused == 1
    assert len(llm2.seen_prompts) == 1
    assert len(fake_server.imported) == 1
    assert _unit_count(store, source) == 0


async def test_acooperative_stop_between_chunks_saves_checkpoints_and_skips_import(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    ingester, llm = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = await ingester.aingest_text(DOC_TEXT, source=source, should_stop=should_stop)

    assert outcome.interrupted
    assert not outcome.ok
    assert len(llm.seen_prompts) == 1
    assert fake_server.imported == []
    assert _unit_count(store, source) == 1


def test_stop_before_the_first_chunk_makes_no_model_calls(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    events: list[Any] = []
    ingester, llm = make_ingester(sync_client, async_client, [MODEL_ANSWER], on_event=events.append)
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md", should_stop=lambda: True)
    assert outcome.interrupted
    assert not outcome.ok
    assert llm.seen_prompts == []
    assert [event.kind for event in events] == ["document_started"]


def test_ingest_documents_stops_between_documents_and_reports_only_attempted_ones(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, [MODEL_ANSWER, MODEL_ANSWER])
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2  # doc1's own document+chunk checks, then stop before doc2

    outcomes = ingester.ingest_documents(
        [
            Document(page_content=DOC_TEXT, metadata={"source": "docs/a.md"}),
            Document(page_content=DOC_TEXT, metadata={"source": "docs/b.md"}),
        ],
        should_stop=should_stop,
    )
    assert len(outcomes) == 1
    assert outcomes[0].ok
    assert outcomes[0].source == "docs/a.md"
    assert len(llm.seen_prompts) == 1


async def test_aingest_documents_stops_between_documents_and_reports_only_attempted_ones(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, [MODEL_ANSWER, MODEL_ANSWER])
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    outcomes = await ingester.aingest_documents(
        [
            Document(page_content=DOC_TEXT, metadata={"source": "docs/a.md"}),
            Document(page_content=DOC_TEXT, metadata={"source": "docs/b.md"}),
        ],
        should_stop=should_stop,
    )
    assert len(outcomes) == 1
    assert outcomes[0].ok


def test_ingest_documents_interruption_bypasses_raise_on_error(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 5

    ingester, llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER, CHUNK1_ANSWER],
        chunk_bytes=CROSS_CHUNK_BYTES,
        raise_on_error=True,
    )
    outcomes = ingester.ingest_documents(
        [
            Document(page_content=DOC_TEXT, metadata={"source": "docs/a.md"}),
            Document(page_content=DOC_TEXT, metadata={"source": "docs/b.md"}),
        ],
        should_stop=should_stop,
    )
    # No exception, despite raise_on_error=True: an interruption is not a
    # failure, so it never enters the except-Exception/raise_on_error path.
    assert len(outcomes) == 2
    assert outcomes[0].ok
    assert outcomes[1].interrupted
    assert not outcomes[1].ok


# -- dry-run interaction --------------------------------------------------------


def test_dry_run_records_checkpoints_but_never_deletes_them(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    ingester, _llm = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source=source, dry_run=True)
    assert outcome.ok
    assert fake_server.imported == []
    assert _unit_count(store, source) == 2

    # A second dry run reuses both chunks — real model work already
    # happened once, and dry-run recording is what makes that not wasted.
    ingester2, llm2 = _build_ingester(
        sync_client, async_client, [], checkpoint_store=store, chunk_bytes=CROSS_CHUNK_BYTES
    )
    outcome2 = ingester2.ingest_text(DOC_TEXT, source=source, dry_run=True)
    assert outcome2.ok
    assert outcome2.chunks_reused == 2
    assert llm2.seen_prompts == []
    assert fake_server.imported == []
    assert _unit_count(store, source) == 2  # still not deleted

    # A real (non-dry) run also reuses both chunks, then finally clears
    # the checkpoint once the batch actually lands.
    ingester3, llm3 = _build_ingester(
        sync_client, async_client, [], checkpoint_store=store, chunk_bytes=CROSS_CHUNK_BYTES
    )
    outcome3 = ingester3.ingest_text(DOC_TEXT, source=source)
    assert outcome3.ok
    assert outcome3.chunks_reused == 2
    assert llm3.seen_prompts == []
    assert len(fake_server.imported) == 1
    assert _unit_count(store, source) == 0


# -- fingerprint gate and on-disk degradation -----------------------------------


FINGERPRINT_MISMATCH_CASES = [
    pytest.param({}, CHANGED_TEXT, id="content_changed"),
    pytest.param({"questions": 1}, None, id="questions_changed"),
    pytest.param({"include_passage": False}, None, id="include_passage_changed"),
    pytest.param({"lossy": True}, None, id="lossy_changed"),
    pytest.param({"context": "tea"}, None, id="context_changed"),
    pytest.param(
        {"create_context": True, "context_description": "蔵元一式"}, None, id="description_changed"
    ),
    pytest.param({"checkpoint_model_id": "a-different-model"}, None, id="model_changed"),
]


@pytest.mark.parametrize("overrides,text_override", FINGERPRINT_MISMATCH_CASES)
def test_fingerprint_mismatch_on_each_field_invalidates(
    sync_client: Taguru,
    async_client: AsyncTaguru,
    fake_server: FakeServer,
    overrides: dict[str, Any],
    text_override: str | None,
) -> None:
    store = RecordingCheckpointStore()
    source = _seed_failed_run(sync_client, async_client, store)
    assert _unit_count(store, source) == 1

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
        **overrides,
    )
    text = text_override if text_override is not None else DOC_TEXT
    outcome = ingester2.ingest_text(text, source=source)

    assert outcome.ok
    assert outcome.chunks_reused == 0
    assert len(llm2.seen_prompts) == 2  # nothing reused — a settings mismatch invalidates all


def test_a_changed_structured_output_mode_invalidates_checkpoints(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The one fingerprint field ``FINGERPRINT_MISMATCH_CASES`` can't cover
    generically: turning on ``structured_output`` needs a tool-calling fake
    model (``RecordingFakeChatModel``'s inherited ``bind_tools()`` raises
    ``NotImplementedError``), not the plain chat fake every other case in
    that matrix shares."""
    store = RecordingCheckpointStore()
    source = _seed_failed_run(sync_client, async_client, store)
    assert _unit_count(store, source) == 1

    llm = ToolCallingFakeChatModel(
        tool_call_args=[json.loads(CHUNK1_ANSWER), json.loads(CHUNK2_CORRECTED_ANSWER)]
    )
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        chunk_bytes=CROSS_CHUNK_BYTES,
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
        structured_output=True,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source=source)

    assert outcome.ok
    assert outcome.chunks_reused == 0  # the structured_output mismatch invalidated everything
    assert outcome.llm_calls == 2


def test_a_corrupt_checkpoint_file_on_disk_silently_degrades_to_empty(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer, tmp_path: Path
) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    source = "docs/aomine.md"
    store.path_for(source).parent.mkdir(parents=True, exist_ok=True)
    store.path_for(source).write_bytes(b"{not-valid-json-at-all")

    ingester, llm = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "not json", "not json"],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError):
        ingester.ingest_text(DOC_TEXT, source=source)

    # Both chunks were attempted in full — nothing was reused from the
    # unreadable file.
    assert len(llm.seen_prompts) == 3

    # The file now holds valid, parseable state with the one unit this
    # run actually completed.
    parsed = json.loads(store.path_for(source).read_bytes())
    assert len(parsed["units"]) == 1


def test_a_stale_filesystem_store_is_reused_across_store_instances(
    sync_client: Taguru, async_client: AsyncTaguru, tmp_path: Path
) -> None:
    """A FilesystemCheckpointStore is a thin view over a directory, not a
    process-lifetime cache — a fresh instance pointed at the same
    directory (a new process after a kill/restart) resumes identically."""
    source = "docs/aomine.md"
    store1 = FilesystemCheckpointStore(tmp_path)
    ingester1, _llm1 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "not json", "not json"],
        checkpoint_store=store1,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError):
        ingester1.ingest_text(DOC_TEXT, source=source)

    store2 = FilesystemCheckpointStore(tmp_path)  # a brand new instance, same directory
    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store2,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = ingester2.ingest_text(DOC_TEXT, source=source)
    assert outcome.ok
    assert outcome.chunks_reused == 1
    assert len(llm2.seen_prompts) == 1


# -- semantics interlocks -------------------------------------------------------


def test_a_reused_chunk_still_participates_in_cross_chunk_correction(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """The money test: a checkpointed unit's STORED user/answer, not a
    recomputed one, is what Stage 2 replays — proving `_ChunkRecord`
    reconstruction from a checkpoint is faithful enough for cross-chunk
    correction to work exactly like it does on a freshly-extracted chunk."""
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    # Both chunks pass Stage 1, but chunk 1's alias shadows chunk 0's
    # object — Stage 2 flags chunk 1, and even its ONE corrective turn
    # still shadows, so the bounded re-check fails the whole document.
    # Both Stage-1-valid units are still checkpointed: checkpointing
    # happens in the Stage 1 loop, never touched by Stage 2's outcome.
    ingester1, _llm1 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_SHADOWING_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError, match="cross-chunk"):
        ingester1.ingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 2

    # Rerun: BOTH chunks are satisfied from checkpoint (zero Stage 1
    # calls), and Stage 2 still correctly rebuilds chunk 1's conversation
    # from its stored user/answer to fix the (still-cached) shadowing.
    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = ingester2.ingest_text(DOC_TEXT, source=source)

    assert outcome.ok
    assert outcome.chunks_reused == 2
    assert outcome.correction_attempts == 1
    assert len(llm2.seen_prompts) == 1  # only the Stage 2 corrective call

    stage2_messages = llm2.seen_prompts[0]
    human_contents = [m.content for m in stage2_messages if getattr(m, "type", None) == "human"]
    ai_contents = [m.content for m in stage2_messages if getattr(m, "type", None) == "ai"]
    assert any("part 2 of 2" in content for content in human_contents)
    # The replayed prior-answer turn is the STORED (checkpointed) answer,
    # verbatim — proof the reused record's own conversation was rebuilt
    # from checkpoint data, not re-derived some other way.
    assert CHUNK2_SHADOWING_ANSWER in ai_contents

    assert len(fake_server.imported) == 1
    assert _unit_count(store, source) == 0


async def test_a_reused_chunk_still_participates_in_cross_chunk_correction_async(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    ingester1, _llm1 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_SHADOWING_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError, match="cross-chunk"):
        await ingester1.aingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 2

    ingester2, llm2 = _build_ingester(
        sync_client,
        async_client,
        [CHUNK2_CORRECTED_ANSWER],
        checkpoint_store=store,
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = await ingester2.aingest_text(DOC_TEXT, source=source)
    assert outcome.ok
    assert outcome.chunks_reused == 2
    assert len(llm2.seen_prompts) == 1
    assert len(fake_server.imported) == 1


def test_checkpoints_survive_a_refusal(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    from langchain_core.messages import AIMessage

    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    refused = AIMessage(content="", response_metadata={"done_reason": "content_filter"})
    ingester, _llm = make_ingester_with_messages(
        sync_client,
        async_client,
        [AIMessage(content=CHUNK1_ANSWER), refused],
        chunk_bytes=CROSS_CHUNK_BYTES,
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.raises(ValueError, match="policy refusal is terminal"):
        ingester.ingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 1


def test_checkpoints_survive_an_empty_terminal_failure(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, "", ""],
        chunk_bytes=CROSS_CHUNK_BYTES,
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.raises(ValueError):
        ingester.ingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 1


def test_checkpoints_survive_a_stage_2_residual_issue(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_SHADOWING_ANSWER],
        chunk_bytes=CROSS_CHUNK_BYTES,
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.raises(ValueError, match="cross-chunk"):
        ingester.ingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 2


def test_checkpoints_survive_an_import_failure(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.fail_import = True
    store = RecordingCheckpointStore()
    source = "docs/aomine.md"
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_CORRECTED_ANSWER],
        chunk_bytes=CROSS_CHUNK_BYTES,
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.raises(ServerError):
        ingester.ingest_text(DOC_TEXT, source=source)
    assert _unit_count(store, source) == 2
    assert fake_server.imported == []
    assert ("delete", source) not in store.log


def test_checkpoint_deletion_happens_before_embeddings_refresh_and_a_warning_does_not_resurrect_it(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    fake_server.embeddings_refresh_status = 502
    store = RecordingCheckpointStore()
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
        on_event=lambda event: store.log.append(("event", event.kind)),
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.embeddings_refresh_warning is not None

    delete_index = store.log.index(("delete", "docs/aomine.md"))
    warning_index = store.log.index(("event", "embedding_refresh_warning"))
    assert delete_index < warning_index
    assert store.state == {}


def test_a_checkpoint_save_failure_warns_but_the_unit_still_counts_this_run(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FailingSaveStore()
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.warns(RuntimeWarning, match="failed to save a checkpoint"):
        outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert len(fake_server.imported) == 1


def test_a_checkpoint_delete_failure_is_silently_ignored(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer, recwarn: Any
) -> None:
    store = FailingDeleteStore()
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert len(fake_server.imported) == 1
    assert not recwarn.list


def test_a_checkpoint_load_failure_warns_and_starts_without_any_cached_chunks(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FailingLoadStore()
    ingester, llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        checkpoint_store=store,
        checkpoint_model_id="fake-model",
    )
    with pytest.warns(RuntimeWarning, match="failed to load"):
        outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert len(llm.seen_prompts) == 1


# -- model-identity fail-fast ----------------------------------------------------


def test_checkpoint_store_without_a_derivable_model_identity_requires_checkpoint_model_id(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    llm = RecordingFakeChatModel(responses=[], seen_prompts=[])
    with pytest.raises(ValueError, match="checkpoint_model_id"):
        TaguruIngester(
            context="sake",
            llm=llm,
            client=sync_client,
            async_client=async_client,
            checkpoint_store=RecordingCheckpointStore(),
        )


def test_checkpoint_store_with_an_explicit_model_id_does_not_require_derivation(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    llm = RecordingFakeChatModel(responses=[], seen_prompts=[])
    TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        checkpoint_store=RecordingCheckpointStore(),
        checkpoint_model_id="fake-model",
    )  # must not raise


class _NamedFakeChatModel(RecordingFakeChatModel):
    model: str = "demo-v1"


def test_model_identity_is_derived_from_a_model_attribute_when_present(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    llm = _NamedFakeChatModel(responses=[], seen_prompts=[])
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        checkpoint_store=RecordingCheckpointStore(),
    )  # must not raise — model identity derived from llm.model
    fingerprint = ingester._checkpoint_fingerprint("some text")  # noqa: SLF001
    assert fingerprint.model == "_NamedFakeChatModel:demo-v1"
