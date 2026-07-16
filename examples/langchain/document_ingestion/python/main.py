"""Governed ingestion with TaguruIngester: dry-run review, apply, replace, retract.

Writing to a shared memory deserves the same ceremony as a code review.
This example runs the write path in stages against a fictional tea shop's
docs: (1) dry_run renders the exact NDJSON batch the server would apply —
nothing is written; (2) the reviewed NDJSON is applied byte-for-byte with
the core SDK's import_batches; (3) the core SDK reads the graph back;
(4) re-ingesting a REVISED document under the same source id retracts the
old contribution first (weights never double-count); (5) retract_source
withdraws a document outright.

Runs self-contained: with no TAGURU_URL set it spawns a real server binary
(builds it with cargo on first run), and with no OPENAI_API_KEY it drives
the extraction with a deterministic fake model.

    cd examples/langchain && .venv/bin/python document_ingestion/python/main.py

What to look for in the output: the NDJSON lines in step 1 (header with
create block, the verbatim passage, doc2query questions, facts with
paragraph locators, aliases — that stream IS the import payload); the
創業年 weight staying 1.0 through the step-4 re-ingest; and the label
vocabulary listing in step 3 — those labels seed the next ingest prompt so
the model reuses relation names instead of coining synonyms.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from taguru import Taguru
from taguru_langchain import TaguruIngester

REPO_ROOT = Path(__file__).resolve().parents[4]

ABOUT_V1 = """蒼月堂は1892年に京都で創業した架空の茶舗である。

看板商品は玉露の「朝霧」である。

当主は蒼井である。"""

TEA_GUIDE = """玉露は収穫前の数週間、茶園を覆って育てる被覆栽培の茶である。

「朝霧」の茶葉は宇治産である。蒼月堂は「朝霧」の水出しを推奨しない。"""

# The revision: the last paragraph gained a sentence about the new branch.
ABOUT_V2 = """蒼月堂は1892年に京都で創業した架空の茶舗である。

看板商品は玉露の「朝霧」である。

当主は蒼井である。2026年、東京・銀座に支店を開いた。"""

# The decompositions a real chat model would produce — canned so the demo
# runs without an API key, in call order: about v1, tea guide, about v2.
FAKE_ABOUT_V1 = {
    "associations": [
        {"subject": "蒼月堂", "label": "創業年", "object": "1892年", "weight": 1.0, "paragraph": 0},
        {"subject": "蒼月堂", "label": "創業地", "object": "京都", "weight": 1.0, "paragraph": 0},
        {"subject": "蒼月堂", "label": "看板商品", "object": "朝霧", "weight": 1.0, "paragraph": 1},
        {"subject": "朝霧", "label": "種類", "object": "玉露", "weight": 1.0, "paragraph": 1},
        {"subject": "蒼月堂", "label": "当主", "object": "蒼井", "weight": 1.0, "paragraph": 2},
    ],
    "aliases": [{"alias": "Sougetsudo", "canonical": "蒼月堂", "kind": "concept"}],
    "questions": [{"paragraph": 0, "question": "蒼月堂はいつ創業した?"}],
}
FAKE_TEA_GUIDE = {
    "associations": [
        {"subject": "玉露", "label": "栽培方法", "object": "被覆栽培", "weight": 1.0, "paragraph": 0},
        {"subject": "朝霧", "label": "産地", "object": "宇治", "weight": 1.0, "paragraph": 1},
        {"subject": "蒼月堂", "label": "推奨する", "object": "朝霧の水出し", "weight": -1.0, "paragraph": 1},
    ],
    "aliases": [],
    "questions": [{"paragraph": 0, "question": "玉露はどのように育てられる?"}],
}
FAKE_ABOUT_V2 = {
    "associations": [
        {"subject": "蒼月堂", "label": "創業年", "object": "1892年", "weight": 1.0, "paragraph": 0},
        {"subject": "蒼月堂", "label": "創業地", "object": "京都", "weight": 1.0, "paragraph": 0},
        # A repeat of the first triple — models do this; merge folds it and
        # reports it as duplicates_dropped instead of double-writing.
        {"subject": "蒼月堂", "label": "創業年", "object": "1892年", "weight": 1.0, "paragraph": 0},
        {"subject": "蒼月堂", "label": "看板商品", "object": "朝霧", "weight": 1.0, "paragraph": 1},
        {"subject": "朝霧", "label": "種類", "object": "玉露", "weight": 1.0, "paragraph": 1},
        {"subject": "蒼月堂", "label": "当主", "object": "蒼井", "weight": 1.0, "paragraph": 2},
        {"subject": "蒼月堂", "label": "支店", "object": "東京・銀座", "weight": 1.0, "paragraph": 2},
    ],
    "aliases": [{"alias": "Sougetsudo", "canonical": "蒼月堂", "kind": "concept"}],
    "questions": [{"paragraph": 0, "question": "蒼月堂はいつ創業した?"}],
}


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

        fakes = [FAKE_ABOUT_V1, FAKE_TEA_GUIDE, FAKE_ABOUT_V2]
        ingester = TaguruIngester(
            context="sougetsu-kb",
            llm=make_llm([json.dumps(fake, ensure_ascii=False) for fake in fakes]),
            client=client,
            create_context=True,
            context_description="蒼月堂という架空の茶舗の知識",
            questions=1,
        )
        documents = [
            Document(page_content=ABOUT_V1, metadata={"source": "docs/sougetsu/about.md"}),
            Document(page_content=TEA_GUIDE, metadata={"source": "docs/sougetsu/tea-guide.md"}),
        ]

        print("== 1. dry run: review the exact batch before anything is written ==")
        reviewed = ingester.ingest_documents(documents, dry_run=True)
        failed = [outcome for outcome in reviewed if not outcome.ok]
        if failed:
            for outcome in failed:
                print(f"FAILED to ingest {outcome.source}: {outcome.error}", file=sys.stderr)
            return 1
        for outcome in reviewed:
            print(f"--- NDJSON for {outcome.source} ---")
            assert outcome.ndjson is not None
            for line in outcome.ndjson.strip().splitlines():
                print(f"  {line}")
        print(f"context created yet? {client.contexts.exists('sougetsu-kb')}")

        print("\n== 2. apply the reviewed batches — the NDJSON *is* the import payload ==")
        for outcome in reviewed:
            assert outcome.ndjson is not None
            applied = client.import_batches(outcome.ndjson).batches[0]
            print(
                f"{outcome.source}: created={applied.created} "
                f"associations={applied.associations} aliases={applied.aliases} "
                f"questions={applied.questions_stored} passage={applied.passage_stored}"
            )

        ctx = client.context("sougetsu-kb")
        print("\n== 3. read the graph back with the core SDK ==")
        print(f"sources: {ctx.list_sources().sources}")
        description = ctx.describe("蒼月堂")
        if description is not None:
            usages = ", ".join(f"{usage.label}×{usage.count}" for usage in description.as_subject)
            print(f"蒼月堂 as subject: {usages}")
        founded = ctx.query(subject="蒼月堂", label="創業年").matches[0]
        print(f"創業年: {founded.object} (weight {founded.weight})")
        print(f"label vocabulary (seeds the next ingest prompt): {ctx.list_labels().labels}")

        print("\n== 4. re-ingest a REVISED document under the same source id ==")
        outcome = ingester.ingest_text(ABOUT_V2, source="docs/sougetsu/about.md")
        print(
            f"retracted {outcome.retracted} old associations, applied {outcome.associations} "
            f"(model repeated a triple {outcome.duplicates_dropped}× — merge folded it)"
        )
        founded = ctx.query(subject="蒼月堂", label="創業年").matches[0]
        print(f"創業年 after re-ingest: {founded.object} (weight {founded.weight} — replaced, not doubled)")
        branches = ctx.query(subject="蒼月堂", label="支店").matches
        print(f"支店 (new in the revision): {[match.object for match in branches]}")

        print("\n== 5. withdraw a whole document ==")
        gone = ctx.retract_source("docs/sougetsu/tea-guide.md")
        print(f"associations touched: {gone.associations_touched}, passage removed: {gone.passage_removed}")
        print(f"sources now: {ctx.list_sources().sources}")
        return 0
    finally:
        if spawned is not None:
            spawned.stop()


if __name__ == "__main__":
    sys.exit(main())
