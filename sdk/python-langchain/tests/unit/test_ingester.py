"""TaguruIngester with a deterministic fake chat model (no network LLM)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruIngester

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
            {"subject": "", "label": "壊", "object": "れ", "weight": 1.0},  # invalid
        ],
        "aliases": [{"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}],
        "questions": [{"paragraph": 1, "question": "杜氏は誰?"}],
    },
    ensure_ascii=False,
)

DOC_TEXT = "青嶺酒造は1907年創業。\n\n杜氏は高瀬である。"


class RecordingFakeChatModel(FakeListChatModel):
    """FakeListChatModel that also records every prompt it was given."""

    seen_prompts: list[list[Any]] = []

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
        self.seen_prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


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


def test_ingest_text_builds_the_batch_and_imports(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    ingester, llm = make_ingester(sync_client, async_client, [MODEL_ANSWER])
    outcome = ingester.ingest_text(DOC_TEXT, source="docs/aomine.md")

    assert outcome.ok
    assert outcome.llm_calls == 1
    assert outcome.chunks == 1
    assert outcome.duplicates_dropped == 1
    assert outcome.invalid_dropped == 1
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
        client = ingester.client
        assert client is not None
        assert not client._http.is_closed
    assert client._http.is_closed


async def test_async_context_manager_closes_self_built_clients_on_exit() -> None:
    llm = FakeListChatModel(responses=["{}"])
    async with TaguruIngester(context="sake", llm=llm, base_url="http://test") as ingester:
        client, async_client_ = ingester.client, ingester.async_client
        assert client is not None
        assert async_client_ is not None
    assert client._http.is_closed
    assert async_client_._http.is_closed
