"""Taguru as an assistant's long-term memory: remember across sessions, correct on update.

Chat history is short-lived; what the user TOLD you should not be. After
each session the transcript is decomposed into the memory context by
TaguruIngester — one source id per session ("conversations/<date>"), so
re-memorizing a session replaces it instead of double-counting. In the
next session every user turn first pulls the relevant memories through
TaguruRetriever and hands them to the chat model.

Runs self-contained: with no TAGURU_URL set it spawns a real server binary
(builds it with cargo on first run), and with no OPENAI_API_KEY it drives
both LLM roles (memorizing and chatting) with deterministic fake models.

    cd examples/langchain && .venv/bin/python conversational_memory/python/main.py

What to look for in the output: session 2's first turn retrieves the そば
allergy memorized a week earlier; the correction turn writes 締切→9月15日
with weight +1 AND 締切→8月末 with weight -1 in one batch — the query
afterwards shows the outdated fact's weight cancelled to 0 while the new
one stands, so the re-asked question now gets the current answer. The
retriever matches lexically and structurally (そば pulls そば facts); leaps
like ガレット→そば粉 are the chat model's job — put a query-rewriting step
in front if you need them retrieved.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from taguru import Taguru
from taguru_langchain import TaguruIngester, TaguruRetriever

REPO_ROOT = Path(__file__).resolve().parents[4]

SESSION_1 = """ユーザー: そばアレルギーがあるので、そば粉を使った料理は食べられません。

ユーザー: いま「たぐる導入」というプロジェクトを進めていて、締切は8月末です。

ユーザー: 犬を飼っています。名前はハナです。"""

SESSION_2 = """ユーザー: 「たぐる導入」の締切は9月15日に延期になった。8月末ではなくなったよ。"""

# What a real chat model would extract from each transcript — canned so the
# demo runs without an API key. Session 2 both asserts the new deadline and
# NEGATES the outdated one: correction is a first-class write.
FAKE_MEMORY_SESSION_1 = {
    "associations": [
        {"subject": "ユーザー", "label": "アレルギー", "object": "そば", "weight": 1.0, "paragraph": 0},
        {"subject": "ユーザー", "label": "食べられる", "object": "そば粉を使った料理", "weight": -1.0, "paragraph": 0},
        {"subject": "ユーザー", "label": "進めている", "object": "たぐる導入", "weight": 1.0, "paragraph": 1},
        {"subject": "たぐる導入", "label": "締切", "object": "8月末", "weight": 1.0, "paragraph": 1},
        {"subject": "ユーザー", "label": "飼っている", "object": "犬", "weight": 1.0, "paragraph": 2},
        {"subject": "犬", "label": "名前", "object": "ハナ", "weight": 1.0, "paragraph": 2},
    ],
    "aliases": [],
    "questions": [],
}
FAKE_MEMORY_SESSION_2 = {
    "associations": [
        {"subject": "たぐる導入", "label": "締切", "object": "9月15日", "weight": 1.0, "paragraph": 0},
        {"subject": "たぐる導入", "label": "締切", "object": "8月末", "weight": -1.0, "paragraph": 0},
    ],
    "aliases": [],
    "questions": [],
}

TURNS = [
    "今夜の夕食、そば屋さんはどうかな?",
    "「たぐる導入」の締切っていつだったっけ?",
]
RE_ASKED = "もう一度確認だけど、「たぐる導入」の締切はいつ?"

# What a grounded model would answer from the retrieved memories — canned in
# call order: the two session-2 turns, then the re-asked question.
FAKE_REPLIES = [
    "そばアレルギーをお持ちなので、そば屋は避けたほうがよさそうです(2026-07-05の会話より)。"
    "別のお店を探しましょうか。",
    "「たぐる導入」の締切は8月末です(2026-07-05の会話より)。",
    "「たぐる導入」の締切は9月15日です。8月末から延期になりました(2026-07-12の会話より)。",
]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the user's personal assistant. Answer in Japanese. Ground yourself in "
            "the long-term memories below (each line notes which conversation it came from). "
            "A negative-weight fact is something that is NOT true.\n\n{memory}",
        ),
        ("human", "{message}"),
    ]
)


def make_llm(fake_responses: list[str]) -> BaseChatModel:
    """A real model when OPENAI_API_KEY is available, else the canned fake."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError:
            print("(OPENAI_API_KEY set but langchain-openai not installed — using the fake model)")
        else:
            return ChatOpenAI(model="gpt-4.1", temperature=0)
    else:
        print("(no OPENAI_API_KEY — using a canned fake model)")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=fake_responses)


def format_memory(documents: list[Document]) -> str:
    """Retrieved Documents → the memory block: provenance, plus each backing
    graph fact with its weight — negations (weight < 0) stay visible."""
    lines = []
    for document in documents:
        meta = document.metadata
        lines.append(f"- ({meta.get('source') or 'graph'}) {document.page_content}")
        for association in meta.get("associations") or []:
            lines.append(
                f"    fact: {association['subject']} –{association['label']}→ "
                f"{association['object']} (weight {association['weight']:+g})"
            )
    return "\n".join(lines) if lines else "- (no relevant memories)"


def main() -> int:
    spawned = None
    if not os.environ.get("TAGURU_URL"):
        from taguru.testing import SpawnedServer, default_binary

        print("(no TAGURU_URL — spawning a local server)")
        spawned = SpawnedServer(default_binary(REPO_ROOT), tempfile.mkdtemp(), {})
        os.environ["TAGURU_URL"] = spawned.base_url
        os.environ.pop("TAGURU_API_TOKEN", None)

    try:
        client = Taguru()
        client.wait_until_ready()

        ingester = TaguruIngester(
            context="assistant-memory",
            llm=make_llm(
                [
                    json.dumps(FAKE_MEMORY_SESSION_1, ensure_ascii=False),
                    json.dumps(FAKE_MEMORY_SESSION_2, ensure_ascii=False),
                ]
            ),
            client=client,
            create_context=True,
            context_description="アシスタントがユーザーについて記憶していること",
        )
        retriever = TaguruRetriever(context="assistant-memory", client=client, k=4)
        chat = make_llm(FAKE_REPLIES)
        respond = PROMPT | chat | StrOutputParser()

        print("== session 1 (2026-07-05): the user talks; the transcript is memorized after ==")
        print(SESSION_1)
        memorized = ingester.ingest_documents(
            [Document(page_content=SESSION_1, metadata={"source": "conversations/2026-07-05"})]
        )[0]
        if not memorized.ok:
            print(f"FAILED to ingest {memorized.source}: {memorized.error}", file=sys.stderr)
            return 1
        print(f"memorized {memorized.source}: {memorized.associations} facts")

        print("\n== session 2 (2026-07-12): every turn first recalls, then answers ==")
        for turn in TURNS:
            print(f"\nuser: {turn}")
            memories = retriever.invoke(turn)
            print(format_memory(memories))
            print(f"assistant: {respond.invoke({'memory': format_memory(memories), 'message': turn})}")

        print("\n== the user corrects a fact; the correction is memorized ==")
        print(SESSION_2)
        memorized = ingester.ingest_text(SESSION_2, source="conversations/2026-07-12")
        print(f"memorized {memorized.source}: {memorized.associations} facts")
        ctx = client.context("assistant-memory")
        for match in ctx.query(subject="たぐる導入", label="締切").matches:
            print(f"  たぐる導入 –締切→ {match.object}: weight {match.weight:+g}")

        print(f"\nuser: {RE_ASKED}")
        memories = retriever.invoke(RE_ASKED)
        print(format_memory(memories))
        print(f"assistant: {respond.invoke({'memory': format_memory(memories), 'message': RE_ASKED})}")
        return 0
    finally:
        if spawned is not None:
            spawned.stop()


if __name__ == "__main__":
    sys.exit(main())
