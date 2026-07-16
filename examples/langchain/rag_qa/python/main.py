"""RAG question answering over Taguru with LangChain: ingest, retrieve, answer with citations.

The write path decomposes three short documents about a fictional sake
brewery into the association graph (TaguruIngester); the read path is a
plain LCEL chain — TaguruRetriever feeding a prompt and a chat model — that
answers questions in Japanese with per-paragraph citations.

Runs self-contained: with no TAGURU_URL set it spawns a real server binary
(builds it with cargo on first run), and with no OPENAI_API_KEY it drives
both LLM roles (extraction and answering) with deterministic fake models.

    cd examples/langchain && .venv/bin/python rag_qa/python/main.py

What to look for in the output: each question first shows what the
retriever brought back — which lane it came from (graph / text /
graph+text) and where it lives (source ¶paragraph) — and then the model's
answer, whose bracketed citations point back at exactly those locators.
The last question lands on a NEGATIVELY weighted fact (青嶺酒造–行う–大量生産,
weight -1.0): the extract discipline stores what a document denies as
first-class evidence, and the retriever carries the weight through in the
Document metadata.
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
from langchain_core.runnables import RunnablePassthrough
from taguru import Taguru
from taguru_langchain import TaguruIngester, TaguruRetriever

REPO_ROOT = Path(__file__).resolve().parents[4]

DOCS = {
    "docs/aomine/brewery.md": """青嶺酒造は1907年創業の架空の酒蔵である。蔵は岩手県遠野市にある。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。""",
    "docs/aomine/lineup.md": """「青嶺 大吟醸」は精米歩合40%の山田錦で仕込まれる。

「青嶺 純米」は地元米の遠野錦を使う。冬季限定の「冬青嶺」は12月から2月にのみ出荷される。""",
    "docs/aomine/visits.md": """蔵見学は要予約で、所要時間はおよそ60分である。

蔵見学の定休日は水曜日である。試飲は見学コースの最後に含まれる。""",
}

# The decomposition a real chat model would produce under the extract
# discipline — canned per source so the demo runs without an API key.
FAKE_EXTRACTIONS = {
    "docs/aomine/brewery.md": {
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "所在地", "object": "岩手県遠野市", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1},
            {"subject": "高瀬", "label": "重視する", "object": "寒仕込み", "weight": 1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "paragraph": 2},
        ],
        "aliases": [{"alias": "Aomine Brewery", "canonical": "青嶺酒造", "kind": "concept"}],
        "questions": [{"paragraph": 1, "question": "青嶺酒造の杜氏は誰?"}],
    },
    "docs/aomine/lineup.md": {
        "associations": [
            {"subject": "青嶺 大吟醸", "label": "精米歩合", "object": "40%", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺 大吟醸", "label": "原料米", "object": "山田錦", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺 純米", "label": "原料米", "object": "遠野錦", "weight": 1.0, "paragraph": 1},
            {"subject": "冬青嶺", "label": "出荷時期", "object": "12月から2月", "weight": 1.0, "paragraph": 1},
        ],
        "aliases": [],
        "questions": [{"paragraph": 0, "question": "大吟醸の精米歩合はいくつ?"}],
    },
    "docs/aomine/visits.md": {
        "associations": [
            {"subject": "蔵見学", "label": "予約", "object": "必要", "weight": 1.0, "paragraph": 0},
            {"subject": "蔵見学", "label": "所要時間", "object": "60分", "weight": 1.0, "paragraph": 0},
            {"subject": "蔵見学", "label": "定休日", "object": "水曜日", "weight": 1.0, "paragraph": 1},
            {"subject": "試飲", "label": "含まれる", "object": "見学コース", "weight": 1.0, "paragraph": 1},
        ],
        "aliases": [{"alias": "見学", "canonical": "蔵見学", "kind": "concept"}],
        "questions": [{"paragraph": 1, "question": "見学が休みなのは何曜日?"}],
    },
}

QUESTIONS = [
    "青嶺酒造の杜氏は誰ですか?",
    "青嶺 大吟醸はどんな米で造られていますか?",
    "青嶺酒造はお酒を大量生産していますか?",
]

# What a grounded model would answer from the retrieved context — canned in
# the same order as QUESTIONS.
FAKE_ANSWERS = [
    "杜氏は高瀬です [docs/aomine/brewery.md ¶1]。高瀬は寒仕込みを重視しています [docs/aomine/brewery.md ¶1]。",
    "「青嶺 大吟醸」は精米歩合40%まで磨いた山田錦で仕込まれます [docs/aomine/lineup.md ¶0]。",
    "いいえ、青嶺酒造は大量生産を行っていません [docs/aomine/brewery.md ¶2]。",
]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about 青嶺酒造 in Japanese, using ONLY the facts in the "
            "context below. Cite every claim with its bracketed locator, e.g. "
            "[docs/aomine/brewery.md ¶1]. If the context does not answer the question, say so.\n\n"
            "{context}",
        ),
        ("human", "{question}"),
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


def format_docs(documents: list[Document]) -> str:
    """Retrieved Documents → the context block, each line locator-first so
    the model has something concrete to cite."""
    lines = []
    for document in documents:
        meta = document.metadata
        if meta.get("paragraph") is not None:
            where = f"{meta['source']} ¶{meta['paragraph']}"
        else:
            where = f"{meta['source']} (graph fact)" if meta.get("source") else "graph fact"
        lines.append(f"[{where}] {document.page_content}")
    return "\n".join(lines)


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

        # -- write: LLM-driven decomposition, one idempotent batch per source --
        ingester = TaguruIngester(
            context="aomine-qa",
            llm=make_llm([json.dumps(FAKE_EXTRACTIONS[source], ensure_ascii=False) for source in DOCS]),
            client=client,
            create_context=True,
            context_description="青嶺酒造という架空の酒蔵の知識",
            questions=2,
        )
        documents = [
            Document(page_content=text, metadata={"source": source})
            for source, text in DOCS.items()
        ]
        outcomes = ingester.ingest_documents(documents)
        failed = [outcome for outcome in outcomes if not outcome.ok]
        if failed:
            for outcome in failed:
                print(f"FAILED to ingest {outcome.source}: {outcome.error}", file=sys.stderr)
            return 1
        for outcome in outcomes:
            print(f"ingested {outcome.source}: {outcome.associations} facts, {outcome.aliases} aliases")

        # -- read: the retriever composes like any other LCEL component --------
        retriever = TaguruRetriever(context="aomine-qa", client=client, k=6)
        llm = make_llm(FAKE_ANSWERS)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )

        for question in QUESTIONS:
            print(f"\n== {question} ==")
            # Shown for the demo; the chain runs the same retrieval internally.
            for document in retriever.invoke(question):
                meta = document.metadata
                where = (
                    f"{meta['source']} ¶{meta['paragraph']}"
                    if meta.get("paragraph") is not None
                    else "graph fact"
                )
                print(f"  [{meta['lane']:>10}] ({where}) {document.page_content}")
            print(f"  answer: {chain.invoke(question)}")
        return 0
    finally:
        if spawned is not None:
            spawned.stop()


if __name__ == "__main__":
    sys.exit(main())
