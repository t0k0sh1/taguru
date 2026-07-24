"""TaguruIngester with a deterministic fake chat model (no network LLM)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruIngester
from taguru_langchain._extract import MAX_PASSAGE_BYTES
from taguru_langchain.events import AttemptFailed, AttemptStarted

from .conftest import FakeServer

MODEL_ANSWER = json.dumps(
    {
        "associations": [
            {
                "subject": "青嶺酒造",
                "label": "杜氏",
                "object": "高瀬",
                "weight": 1.0,
                "paragraph": 1,
            },
            {
                "subject": "青嶺酒造",
                "label": "創業年",
                "object": "1907年",
                "weight": 1.0,
                "paragraph": 0,
            },
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0},  # duplicate
        ],
        "aliases": [{"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}],
        "questions": [{"paragraph": 1, "question": "杜氏は誰?"}],
    },
    ensure_ascii=False,
)

# The business-rule-invalid item issue #180's strict default now rejects on
# sight (empty subject) rather than letting merge() silently drop it —
# split out of MODEL_ANSWER so most fixtures stay a clean one-call success;
# reused by the lossy-mode and corrective-turn tests below.
INVALID_ASSOCIATION = {"subject": "", "label": "壊", "object": "れ", "weight": 1.0}

DOC_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬である。"

# chunk_bytes=40 splits DOC_TEXT into exactly its two paragraphs (one chunk
# each) — the minimum needed to exercise cross-chunk alias validation
# (issue #180 Stage 2), which only ever fires with more than one chunk.
CROSS_CHUNK_BYTES = 40

# Chunk 1 (paragraph 0) introduces the concept name "1907年" as an
# association object.
CHUNK1_ANSWER = json.dumps(
    {
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0}
        ],
        "aliases": [],
    },
    ensure_ascii=False,
)
# Chunk 2 (paragraph 1) passes Stage 1 alone (every alias field present, not
# a self-alias) — the shadowing is only visible once chunk 1's "1907年" is
# in the merged name set, so only Stage 2's cross_output_issues catches it.
CHUNK2_SHADOWING_ANSWER = json.dumps(
    {
        "associations": [{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0}],
        "aliases": [{"alias": "1907年", "canonical": "青嶺酒造", "kind": "concept"}],
    },
    ensure_ascii=False,
)
CHUNK2_CORRECTED_ANSWER = json.dumps(
    {
        "associations": [{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0}],
        "aliases": [{"alias": "Sake", "canonical": "青嶺酒造", "kind": "concept"}],
    },
    ensure_ascii=False,
)


class RecordingFakeChatModel(FakeListChatModel):
    """FakeListChatModel that also records every prompt it was given."""

    seen_prompts: list[list[Any]] = []

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        self.seen_prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


class RecordingFakeMessagesListChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel that also records every prompt it was given.

    Unlike RecordingFakeChatModel (string-only responses), responses here
    are full AIMessage objects, so response_metadata/usage_metadata can be
    injected — needed to assert on TaguruIngester's provider-metadata
    normalization.
    """

    seen_prompts: list[list[Any]] = []

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        self.seen_prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


class ThrowingChatModel(FakeListChatModel):
    """Raises a genuine (non-TaguruError, non-ValueError) provider error —
    e.g. what an ``openai.RateLimitError`` looks like from the caller's side.
    ``_agenerate`` is inherited from ``SimpleChatModel``, which runs this
    same ``_generate`` in an executor, so one override covers both paths."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("rate limited")


class ToolCallingFakeChatModel(BaseChatModel):
    """Minimal fake that implements ``bind_tools()`` — unlike the
    ``FakeListChatModel`` family above, whose inherited ``bind_tools()``
    raises ``NotImplementedError`` — needed to exercise
    ``TaguruIngester(structured_output=True)`` end to end.
    ``_agenerate`` is inherited from ``BaseChatModel``, which (like
    ``SimpleChatModel``) runs ``_generate`` in an executor, so one override
    covers both paths.
    """

    tool_call_args: list[dict[str, Any]]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> ChatResult:
        args_for_call = self.tool_call_args[self.calls]
        self.calls += 1
        # The name a schema with `"title": "ModelOutput"` resolves to via
        # with_structured_output()'s convert_to_openai_tool() — a real
        # provider fills this in from the bound tool spec; this fake just
        # hardcodes the one name that schema will ever produce.
        message = AIMessage(
            content="",
            tool_calls=[{"name": "ModelOutput", "args": args_for_call, "id": f"call_{self.calls}"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_ingester(
    sync_client: Taguru, async_client: AsyncTaguru, responses: list[str], **kwargs: Any
) -> tuple[TaguruIngester, RecordingFakeChatModel]:
    llm = RecordingFakeChatModel(responses=responses, seen_prompts=[])
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        **kwargs,
    )
    return ingester, llm


def make_ingester_with_messages(
    sync_client: Taguru, async_client: AsyncTaguru, responses: list[BaseMessage], **kwargs: Any
) -> tuple[TaguruIngester, RecordingFakeMessagesListChatModel]:
    llm = RecordingFakeMessagesListChatModel(responses=responses, seen_prompts=[])
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        **kwargs,
    )
    return ingester, llm


def test_ingest_text_builds_the_batch_and_imports(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, [MODEL_ANSWER])
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")

    assert outcome.ok
    assert outcome.llm_calls == 1
    assert outcome.chunks == 1
    assert outcome.duplicates_dropped == 1
    assert outcome.invalid_dropped == 0
    # The server's ImportOutcome passed through.
    assert outcome.associations == 2
    assert outcome.aliases == 1
    assert outcome.passage_stored

    # The wire batch: header, verbatim passage, question, 2 facts, alias.
    assert len(fake_server.imported) == 1
    lines = [json.loads(line) for line in fake_server.imported[0].strip().split("\n")]
    assert lines[0]["taguru_batch"] == 1
    assert lines[0]["context"] == "sake"
    assert lines[0]["source"] == "docs/aomine.md"
    assert "create" not in lines[0]
    assert lines[1] == {"passage": DOC_TEXT}  # verbatim, unchunked, unlabeled
    assert lines[2] == {"paragraph": 1, "question": "杜氏は誰?"}
    assert {line["label"] for line in lines[3:5]} == {"杜氏", "創業年"}
    assert lines[3]["paragraph"] == 1
    assert lines[5] == {"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}

    # The live vocabulary seeded the prompt (an edge extract.rs cannot have).
    system = llm.seen_prompts[0][0].content
    assert "代表銘柄" in system
    assert "[0] 青嶺酒造は1907年創業。" in llm.seen_prompts[0][1].content


def test_structured_output_true_uses_with_structured_output(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """The opt-in path: the tool call's args arrive already
    MODEL_OUTPUT_JSON_SCHEMA-shaped — no free-text JSON to parse — yet the
    batch comes out the same as the free-text path fed the same content."""
    llm = ToolCallingFakeChatModel(tool_call_args=[json.loads(MODEL_ANSWER)])
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        structured_output=True,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")

    assert outcome.ok
    assert outcome.llm_calls == 1
    assert outcome.duplicates_dropped == 1
    assert outcome.invalid_dropped == 0
    assert outcome.associations == 2
    assert outcome.aliases == 1


async def test_astructured_output_true_uses_with_structured_output(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """Async twin: the structured path goes through ainvoke(), not invoke()."""
    llm = ToolCallingFakeChatModel(tool_call_args=[json.loads(MODEL_ANSWER)])
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        structured_output=True,
    )
    outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")

    assert outcome.ok
    assert outcome.llm_calls == 1
    assert outcome.associations == 2
    assert outcome.aliases == 1


def test_dry_run_renders_but_never_sends(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, _llm = make_ingester(sync_client, async_client, [MODEL_ANSWER])
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md", dry_run=True)
    assert outcome.ok
    assert outcome.ndjson is not None and '"taguru_batch"' in outcome.ndjson
    assert fake_server.imported == []
    assert outcome.associations == 0  # nothing applied


def test_malformed_answer_gets_one_corrective_turn(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(
        sync_client, async_client, ["I cannot answer in JSON, sorry!", MODEL_ANSWER]
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 2
    corrective = llm.seen_prompts[1]
    assert "not the single JSON object" in corrective[-1].content

    ingester, _llm = make_ingester(sync_client, async_client, ["nope", "still nope"])
    with pytest.raises(ValueError, match="would not produce the JSON object"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")


def test_create_context_stamps_the_header(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        create_context=True,
        context_description="酒蔵の知識",
    )
    ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    header = json.loads(fake_server.imported[0].split("\n", 1)[0])
    assert header["create"] == {"description": "酒蔵の知識"}


def test_documents_require_a_source_id(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    ingester, _llm = make_ingester(sync_client, async_client, [MODEL_ANSWER, MODEL_ANSWER])
    outcomes = ingester.ingest_documents(
        [
            Document(page_content=DOC_TEXT),  # no source metadata
            Document(page_content=DOC_TEXT, metadata={"source": "docs/aomine.md"}),
        ]
    )
    assert not outcomes[0].ok
    assert outcomes[0].error is not None and "source" in outcomes[0].error
    assert outcomes[1].ok

    strict, _llm = make_ingester(sync_client, async_client, [MODEL_ANSWER], raise_on_error=True)
    with pytest.raises(ValueError, match="source"):
        strict.ingest_documents([Document(page_content=DOC_TEXT)])


def test_ingest_documents_records_a_real_llm_error_as_a_failed_outcome(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """A genuine provider error (neither TaguruError nor ValueError) must
    still land in the outcome — not blow past ingest_documents entirely."""
    ingester = TaguruIngester(
        context="sake",
        llm=ThrowingChatModel(responses=[]),
        client=sync_client,
        async_client=async_client,
        questions=2,
    )
    outcomes = ingester.ingest_documents(
        [Document(page_content=DOC_TEXT, metadata={"source": "docs/aomine.md"})]
    )
    assert not outcomes[0].ok
    assert outcomes[0].error is not None and "rate limited" in outcomes[0].error

    strict = TaguruIngester(
        context="sake",
        llm=ThrowingChatModel(responses=[]),
        client=sync_client,
        async_client=async_client,
        questions=2,
        raise_on_error=True,
    )
    with pytest.raises(RuntimeError, match="rate limited"):
        strict.ingest_documents(
            [Document(page_content=DOC_TEXT, metadata={"source": "docs/aomine.md"})]
        )


async def test_aingest_documents_records_a_real_llm_error_as_a_failed_outcome(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester = TaguruIngester(
        context="sake",
        llm=ThrowingChatModel(responses=[]),
        client=sync_client,
        async_client=async_client,
        questions=2,
    )
    outcomes = await ingester.aingest_documents(
        [Document(page_content=DOC_TEXT, metadata={"source": "docs/aomine.md"})]
    )
    assert not outcomes[0].ok
    assert outcomes[0].error is not None and "rate limited" in outcomes[0].error


def test_include_passage_false_skips_the_size_cap(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The passage cap only matters when the passage is actually going to
    be sent — include_passage=False never sends it (the server itself
    only checks passage size when a passage key is present), so an
    oversized text must not be rejected for a limit that no longer
    applies."""
    big_text = DOC_TEXT + "\n\n" + "a" * MAX_PASSAGE_BYTES
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        include_passage=False,
        chunk_bytes=MAX_PASSAGE_BYTES + 1024,
    )
    outcome = ingester.ingest_text(big_text, source="docs/aomine.md")
    assert outcome.ok

    # With include_passage=True (the default), the same oversized text is
    # still rejected — the cap is real, just conditional on actually
    # needing it.
    strict, _llm = make_ingester(
        sync_client, async_client, [MODEL_ANSWER], chunk_bytes=MAX_PASSAGE_BYTES + 1024
    )
    with pytest.raises(ValueError, match="passage cap"):
        strict.ingest_text(big_text, source="docs/aomine.md")


async def test_aingest_text_include_passage_false_skips_the_size_cap(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    big_text = DOC_TEXT + "\n\n" + "a" * MAX_PASSAGE_BYTES
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [MODEL_ANSWER],
        include_passage=False,
        chunk_bytes=MAX_PASSAGE_BYTES + 1024,
    )
    outcome = await ingester.aingest_text(big_text, source="docs/aomine.md")
    assert outcome.ok


def test_embeddings_501_is_silently_ignored(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    """The fake server answers 501 (no provider) — the outcome stays clean."""
    ingester, _llm = make_ingester(sync_client, async_client, [MODEL_ANSWER])
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.embeddings_refresh_warning is None


async def test_async_ingest_matches_sync(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, _llm = make_ingester(sync_client, async_client, [MODEL_ANSWER, MODEL_ANSWER])
    sync_outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    async_outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert async_outcome.ndjson == sync_outcome.ndjson
    assert len(fake_server.imported) == 2


def test_validation_rejects_bad_construction(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    llm = FakeListChatModel(responses=["{}"])
    with pytest.raises(ValueError, match="context_description"):
        TaguruIngester(context="c", llm=llm, client=sync_client, create_context=True)
    with pytest.raises(ValueError, match="questions"):
        TaguruIngester(context="c", llm=llm, client=sync_client, questions=99)


def test_validation_rejects_bad_fact_budget_max_attempts_and_corrective_context_bytes(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    llm = FakeListChatModel(responses=["{}"])
    with pytest.raises(ValueError, match="fact_budget"):
        TaguruIngester(context="c", llm=llm, client=sync_client, fact_budget=0)
    with pytest.raises(ValueError, match="max_attempts must be between 1 and 10"):
        TaguruIngester(context="c", llm=llm, client=sync_client, max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts must be between 1 and 10"):
        TaguruIngester(context="c", llm=llm, client=sync_client, max_attempts=11)
    with pytest.raises(ValueError, match="corrective_context_bytes"):
        TaguruIngester(context="c", llm=llm, client=sync_client, corrective_context_bytes=-1)


def test_structured_output_is_off_by_default(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """Building against a model that cannot bind tools succeeds as long as
    structured_output stays at its default (False) — with_structured_output()
    is only ever called once the caller opts in."""
    llm = FakeListChatModel(responses=["{}"])
    ingester = TaguruIngester(context="c", llm=llm, client=sync_client)
    assert ingester.structured_output is False
    assert ingester._structured_llm is None


def test_structured_output_true_raises_when_the_model_cannot_bind_tools(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """Provider/model dependent, and fails fast: opting in against a chat
    model that doesn't implement bind_tools() — most of the Fake* models
    used elsewhere in this file, matching plenty of real integrations —
    raises out of the constructor, before any document is ingested."""
    llm = FakeListChatModel(responses=["{}"])
    with pytest.raises(NotImplementedError):
        TaguruIngester(context="c", llm=llm, client=sync_client, structured_output=True)


def test_fact_budget_is_folded_into_the_system_prompt(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, [MODEL_ANSWER], fact_budget=3)
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    system = llm.seen_prompts[0][0].content
    assert "Keep this answer to at most 3 association(s) total" in system


def test_max_attempts_env_extends_corrective_retries_past_the_default(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """Two bad answers followed by a good one survive under max_attempts=3 —
    past the default policy (2 total attempts), which would never reach the
    third call."""
    ingester, llm = make_ingester(
        sync_client,
        async_client,
        ["still not json", "nope, still not", MODEL_ANSWER],
        max_attempts=3,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 3
    assert len(llm.seen_prompts) == 3
    # Every corrective turn rebuilds from the base — never accumulates the
    # first bad answer once the second corrective turn is built.
    assert len(llm.seen_prompts[2]) == 4


def test_max_attempts_of_one_skips_the_corrective_turn(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, ["not json at all"], max_attempts=1)
    with pytest.raises(ValueError, match="would not produce the JSON object"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert len(llm.seen_prompts) == 1


def test_corrective_context_bytes_caps_the_replayed_bad_answer(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    bad_answer = "not json at all, definitely not a JSON object"
    ingester, llm = make_ingester(
        sync_client, async_client, [bad_answer, MODEL_ANSWER], corrective_context_bytes=10
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    corrective = llm.seen_prompts[1]
    replayed = corrective[-2].content
    assert "[truncated to 10 bytes]" in replayed
    assert bad_answer not in replayed


def test_corrective_context_bytes_of_zero_omits_the_bad_answer(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    bad_answer = "not json at all, definitely not a JSON object"
    ingester, llm = make_ingester(
        sync_client, async_client, [bad_answer, MODEL_ANSWER], corrective_context_bytes=0
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    corrective = llm.seen_prompts[1]
    replayed = corrective[-2].content
    assert replayed == "[omitted: not the requested JSON object]"
    assert bad_answer not in replayed


def test_a_length_limited_bad_answer_asks_for_shorter_and_names_the_fact_budget(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The fix for Issue #178's stall: a `finish_reason` saying the prior
    answer was cut off at the output cap swaps the corrective ask from "try
    again" to "try again shorter" and names the run's fact_budget."""
    malformed = AIMessage(content="not json, and huge", response_metadata={"done_reason": "length"})
    ingester, llm = make_ingester_with_messages(
        sync_client,
        async_client,
        [malformed, AIMessage(content=MODEL_ANSWER)],
        fact_budget=4,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok

    corrective = llm.seen_prompts[1][-1].content
    assert "SHORTER" in corrective
    assert "cut off at the output limit" in corrective
    assert "Keep it to at most 4 association(s) total." in corrective
    assert "Answer again with only the JSON object." not in corrective


def test_close_leaves_a_caller_supplied_client_open(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, _llm = make_ingester(sync_client, async_client, ["{}"])
    ingester.close()
    assert not sync_client._http.is_closed


async def test_aclose_leaves_caller_supplied_clients_open(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, _llm = make_ingester(sync_client, async_client, ["{}"])
    await ingester.aclose()
    assert not sync_client._http.is_closed
    assert not async_client._http.is_closed


def test_close_closes_a_self_built_client() -> None:
    llm = FakeListChatModel(responses=["{}"])
    ingester = TaguruIngester(context="sake", llm=llm, base_url="http://test")
    client = ingester.client
    assert client is not None
    ingester.close()
    assert client._http.is_closed


def test_close_also_closes_the_self_built_async_client() -> None:
    """close() has no event loop to await aclose() with, but must not leak
    the async client's connections when aingest_* was ever used before it."""
    llm = FakeListChatModel(responses=["{}"])
    ingester = TaguruIngester(context="sake", llm=llm, base_url="http://test")
    async_client_ = ingester.async_client
    assert async_client_ is not None
    ingester.close()
    assert async_client_._http.is_closed


async def test_aclose_closes_both_self_built_clients() -> None:
    llm = FakeListChatModel(responses=["{}"])
    ingester = TaguruIngester(context="sake", llm=llm, base_url="http://test")
    client, async_client_ = ingester.client, ingester.async_client
    assert client is not None
    assert async_client_ is not None
    await ingester.aclose()
    assert client._http.is_closed
    assert async_client_._http.is_closed


def test_sync_context_manager_closes_the_self_built_client_on_exit() -> None:
    llm = FakeListChatModel(responses=["{}"])
    with TaguruIngester(context="sake", llm=llm, base_url="http://test") as ingester:
        client, async_client_ = ingester.client, ingester.async_client
        assert client is not None
        assert async_client_ is not None
        assert not client._http.is_closed
    assert client._http.is_closed
    assert async_client_._http.is_closed


async def test_async_context_manager_closes_self_built_clients_on_exit() -> None:
    llm = FakeListChatModel(responses=["{}"])
    async with TaguruIngester(context="sake", llm=llm, base_url="http://test") as ingester:
        client, async_client_ = ingester.client, ingester.async_client
        assert client is not None
        assert async_client_ is not None
    assert client._http.is_closed
    assert async_client_._http.is_closed


# -- issue #180: lossless JSON repair and path-specific corrective retry ---------


def test_a_trailing_comma_answer_repairs_without_an_llm_call_or_item_change(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    answer = (
        '{"associations": [{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", '
        '"weight": 1.0},], "aliases": [], "questions": []}'
    )
    ingester, _llm = make_ingester(sync_client, async_client, [answer])
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 1
    assert outcome.correction_attempts == 0
    assert outcome.lossless_repairs == ["trailing_comma"]
    # The trailing comma is gone and the one association survived intact —
    # fake_server's /import stub echoes fixed counts, so check the actual
    # rendered batch instead of outcome.associations.
    assert outcome.ndjson is not None
    lines = [json.loads(line) for line in outcome.ndjson.strip().split("\n")]
    facts = [line for line in lines if "subject" in line]
    assert facts == [{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0}]


def test_a_wrong_typed_field_gets_a_path_specific_corrective_turn(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    bad_weight_answer = json.dumps(
        {
            "associations": [
                {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": "strong"}
            ],
            "aliases": [],
        },
        ensure_ascii=False,
    )
    events: list[Any] = []
    ingester, llm = make_ingester(
        sync_client, async_client, [bad_weight_answer, MODEL_ANSWER], on_event=events.append
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 2
    assert outcome.correction_attempts == 1

    corrective = llm.seen_prompts[1][-1].content
    assert "That was valid JSON but not a valid extraction (1 issue(s)):" in corrective
    assert "associations[0].weight: expected finite non-zero number" in corrective
    assert "keep every item" in corrective

    failed = [event for event in events if isinstance(event, AttemptFailed)]
    assert len(failed) == 1
    assert failed[0].validation_issues == [
        'associations[0].weight: expected finite non-zero number, got string "strong"'
    ]


async def test_a_wrong_typed_field_gets_a_path_specific_corrective_turn_async(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    bad_weight_answer = json.dumps(
        {
            "associations": [
                {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": "strong"}
            ],
            "aliases": [],
        },
        ensure_ascii=False,
    )
    ingester, llm = make_ingester(sync_client, async_client, [bad_weight_answer, MODEL_ANSWER])
    outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 2
    assert outcome.correction_attempts == 1
    assert len(llm.seen_prompts) == 2


def test_a_second_invalid_answer_fails_the_source_without_import(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    bad_answer = json.dumps({"associations": [INVALID_ASSOCIATION], "aliases": []})
    ingester, _llm = make_ingester(sync_client, async_client, [bad_answer, bad_answer])
    with pytest.raises(ValueError, match="invalid item"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert fake_server.imported == []


def test_a_length_limited_answer_that_parses_is_never_imported(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """ADR 0001: a `length`-terminated answer is length-limited even when
    its own content happens to parse cleanly — a valid prefix of a cut-off
    extraction must never be imported as if complete."""
    truncated_but_parseable = AIMessage(
        content=MODEL_ANSWER, response_metadata={"done_reason": "length"}
    )
    ingester, llm = make_ingester_with_messages(
        sync_client, async_client, [truncated_but_parseable, AIMessage(content=MODEL_ANSWER)]
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 2
    corrective = llm.seen_prompts[1][-1].content
    assert "SHORTER" in corrective

    strict, _llm = make_ingester_with_messages(
        sync_client, async_client, [truncated_but_parseable], max_attempts=1
    )
    with pytest.raises(ValueError, match="would not produce the JSON object"):
        strict.ingest_text(DOC_TEXT, source="docs/aomine.md")
    # Only the first (successful) ingest above ever called /import.
    assert len(fake_server.imported) == 1


def test_an_empty_answer_gets_one_corrective_then_the_named_diagnosis(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, ["", ""], max_attempts=5)
    with pytest.raises(ValueError, match="the answer was empty"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    # Bounded to exactly one corrective regardless of how high max_attempts is.
    assert len(llm.seen_prompts) == 2

    ingester, _llm = make_ingester(sync_client, async_client, ["", MODEL_ANSWER], max_attempts=5)
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 2


async def test_an_empty_answer_gets_one_corrective_then_the_named_diagnosis_async(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, ["", ""], max_attempts=5)
    with pytest.raises(ValueError, match="the answer was empty"):
        await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert len(llm.seen_prompts) == 2


def test_a_refusal_finish_reason_is_terminal_without_a_corrective_turn(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    refused = AIMessage(content="", response_metadata={"done_reason": "content_filter"})
    ingester, llm = make_ingester_with_messages(
        sync_client, async_client, [refused, AIMessage(content=MODEL_ANSWER)]
    )
    with pytest.raises(ValueError, match="policy refusal is terminal"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert len(llm.seen_prompts) == 1


def test_cross_chunk_alias_issues_get_one_targeted_corrective_turn(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    events: list[Any] = []
    ingester, llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_CORRECTED_ANSWER],
        chunk_bytes=CROSS_CHUNK_BYTES,
        on_event=events.append,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.chunks == 2
    assert outcome.llm_calls == 3
    assert outcome.correction_attempts == 1

    cross_chunk_started = [
        event
        for event in events
        if isinstance(event, AttemptStarted) and event.stage == "cross_chunk"
    ]
    assert len(cross_chunk_started) == 1
    assert cross_chunk_started[0].chunk_index == 1  # the second chunk, 0-indexed

    corrective_prompt = llm.seen_prompts[-1]
    assert "杜氏は高瀬" in corrective_prompt[1].content  # chunk 2's OWN user turn, replayed
    assert "names something the associations already contain" in corrective_prompt[-1].content


async def test_cross_chunk_alias_issues_get_one_targeted_corrective_turn_async(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_CORRECTED_ANSWER],
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    outcome = await ingester.aingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 3
    assert len(llm.seen_prompts) == 3


def test_a_cross_chunk_correction_that_does_not_fix_the_issue_fails_without_import(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """The bounded re-check, not a second round: a "corrective" reply that
    still structurally passes Stage 1 (so the single Stage 2 call itself
    counts as "valid") but repeats the exact same shadowing alias must
    still fail the source — Stage 2 never loops for a second attempt."""
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, CHUNK2_SHADOWING_ANSWER],
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError, match="still has 1 cross-chunk alias issue.s. after correction"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert fake_server.imported == []


def test_a_structurally_invalid_cross_chunk_correction_fails_without_import(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """Unlike the re-check above, Stage 2's OWN corrective reply can itself
    be Stage-1-invalid (e.g. a wrong-typed field) — a distinct failure
    surface with its own message."""
    chunk2_invalid_correction = json.dumps(
        {
            "associations": [
                {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": "strong"}
            ],
            "aliases": [{"alias": "Sake", "canonical": "青嶺酒造", "kind": "concept"}],
        },
        ensure_ascii=False,
    )
    ingester, _llm = make_ingester(
        sync_client,
        async_client,
        [CHUNK1_ANSWER, CHUNK2_SHADOWING_ANSWER, chunk2_invalid_correction],
        chunk_bytes=CROSS_CHUNK_BYTES,
    )
    with pytest.raises(ValueError, match="cross-chunk alias correction still left"):
        ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert fake_server.imported == []


def test_lossy_true_restores_drop_and_proceed(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    answer = json.dumps({"associations": [INVALID_ASSOCIATION], "aliases": []})

    lossy, _llm = make_ingester(sync_client, async_client, [answer], lossy=True)
    outcome = lossy.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 1
    assert outcome.invalid_dropped == 1

    strict, _llm = make_ingester(sync_client, async_client, [answer, answer])
    with pytest.raises(ValueError, match="invalid item"):
        strict.ingest_text(DOC_TEXT, source="docs/aomine.md")


def test_a_failed_reingest_leaves_the_existing_source_untouched(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, _llm = make_ingester(sync_client, async_client, [MODEL_ANSWER])
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert len(fake_server.imported) == 1

    bad_answer = json.dumps({"associations": [INVALID_ASSOCIATION], "aliases": []})
    reingest, _llm2 = make_ingester(sync_client, async_client, [bad_answer, bad_answer])
    with pytest.raises(ValueError, match="invalid item"):
        reingest.ingest_text(DOC_TEXT, source="docs/aomine.md")
    # The failed re-ingest never called /import — the prior batch stands.
    assert len(fake_server.imported) == 1


def test_structured_output_invalid_args_get_a_validation_corrective_turn(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    bad_args = {
        "associations": [
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": "strong"}
        ],
        "aliases": [],
    }
    good_args = json.loads(MODEL_ANSWER)
    llm = ToolCallingFakeChatModel(tool_call_args=[bad_args, good_args])
    events: list[Any] = []
    ingester = TaguruIngester(
        context="sake",
        llm=llm,
        client=sync_client,
        async_client=async_client,
        questions=2,
        structured_output=True,
        on_event=events.append,
    )
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")
    assert outcome.ok
    assert outcome.llm_calls == 2
    assert outcome.correction_attempts == 1

    failed = [event for event in events if isinstance(event, AttemptFailed)]
    assert len(failed) == 1
    assert failed[0].validation_issues == [
        'associations[0].weight: expected finite non-zero number, got string "strong"'
    ]
