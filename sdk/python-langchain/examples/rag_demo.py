"""End-to-end RAG over Taguru with LangChain: ingest → retrieve → cite.

Runs self-contained: with no TAGURU_URL set it spawns a real server binary
(builds it with cargo on first run), and with no OPENAI_API_KEY it drives the
ingester with a deterministic fake chat model — so the full write/read loop
works offline. Point TAGURU_URL/TAGURU_API_TOKEN at a running server and/or
export OPENAI_API_KEY (pip install langchain-openai) for the real thing.

    cd sdk/python-langchain && .venv/bin/python examples/rag_demo.py

What to look for in the output: the ingester reports what one document's
decomposition amounted to (facts, aliases, questions — and what merge
dropped); the retriever then answers a graph-shaped cue with verbatim,
per-paragraph citations, and an answer-shaped text query lands through the
BM25 lane.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from taguru import Taguru
from taguru_langchain import TaguruIngester, TaguruRetriever

DOCS = {
    "docs/aomine.md": """青嶺酒造は1907年創業の架空の酒蔵である。代表銘柄は「青嶺」。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。""",
}

# The decomposition a real chat model would produce under the extract
# discipline — canned so the demo runs without an API key.
FAKE_EXTRACTION = json.dumps(
    {
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1},
            {"subject": "高瀬", "label": "重視する", "object": "寒仕込み", "weight": 1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "paragraph": 2},
        ],
        "aliases": [{"alias": "Aomine Brewery", "canonical": "青嶺酒造", "kind": "concept"}],
        "questions": [{"paragraph": 1, "question": "誰が酒を仕込んでいるの?"}],
    },
    ensure_ascii=False,
)


def make_llm():  # type: ignore[no-untyped-def]
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError:
            print("(OPENAI_API_KEY set but langchain-openai not installed — using the fake model)")
        else:
            return ChatOpenAI(model="gpt-4.1", temperature=0)
    else:
        print("(no OPENAI_API_KEY — driving the ingester with a canned fake model)")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=[FAKE_EXTRACTION] * len(DOCS))


def main() -> int:
    spawned = None
    if not os.environ.get("TAGURU_URL"):
        from taguru.testing import SpawnedServer, default_binary

        repo_root = Path(__file__).resolve().parents[3]
        print("(no TAGURU_URL — spawning a local server)")
        spawned = SpawnedServer(default_binary(repo_root), tempfile.mkdtemp(), {})
        os.environ["TAGURU_URL"] = spawned.base_url
        os.environ.pop("TAGURU_API_TOKEN", None)

    try:
        client = Taguru()
        client.wait_until_ready()

        # -- write: LLM-driven decomposition, per-source replace (idempotent) --
        ingester = TaguruIngester(
            context="sake-demo",
            llm=make_llm(),
            client=client,
            create_context=True,
            context_description="青嶺酒造という架空の酒蔵の知識",
            questions=2,
        )
        documents = [
            Document(page_content=text, metadata={"source": source})
            for source, text in DOCS.items()
        ]
        for outcome in ingester.ingest_documents(documents):
            print(
                f"ingested {outcome.source}: {outcome.associations} facts, "
                f"{outcome.aliases} aliases, {outcome.questions_stored} questions "
                f"(model duplicates folded: {outcome.duplicates_dropped}, "
                f"invalid dropped: {outcome.invalid_dropped})"
            )

        # -- read: graph lane + text lane, RRF-merged ---------------------------
        retriever = TaguruRetriever(context="sake-demo", client=client, k=5)

        print("\n== graph-shaped cue: 青嶺酒造 ==")
        for document in retriever.invoke("青嶺酒造"):
            meta = document.metadata
            where = (
                f"{meta['source']}¶{meta['paragraph']}"
                if meta.get("paragraph") is not None
                else "graph fact"
            )
            print(f"  [{meta['lane']:>10}] ({where}) {document.page_content}")

        print("\n== answer-shaped text query: 1907年に創業した ==")
        for document in retriever.invoke("1907年に創業した"):
            print(f"  [{document.metadata['lane']:>10}] {document.page_content}")

        # Stuff the documents into any chain from here — e.g.
        # `create_stuff_documents_chain(llm, prompt) | retriever`.
        return 0
    finally:
        if spawned is not None:
            spawned.stop()


if __name__ == "__main__":
    sys.exit(main())
