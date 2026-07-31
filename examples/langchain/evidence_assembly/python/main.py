"""Budgeted evidence assembly with Taguru: rank, budget, then hand off to an answer model.

Ingests the same fictional-brewery corpus rag_qa/ uses (LLM-decomposed via
TaguruIngester), then calls the core SDK's assemble_evidence() directly —
not the retrieve()/TaguruRetriever loop the other examples use — under two
budgets for the same query: generous (nothing omitted) and tight
(max_items=1, forcing a choice). Prints the selection trace
(plan.selection/plan.reranker), the budget account, and every omission
before handing the generous package's items to a (fake) answer model.

Runs self-contained: with no TAGURU_URL set it spawns a real server binary
(builds it with cargo on first run), and with no OPENAI_API_KEY it drives
both LLM roles (extraction and answering) with deterministic fake models.

    cd examples/langchain && .venv/bin/python evidence_assembly/python/main.py

What to look for in the output: the tight-budget run's omitted_total/
omitted_by_reason accounting for exactly what the generous run admitted but
the smaller ceiling couldn't hold, and the closing line — everything above
"answer:" is Taguru's own budgeted, deduplicated evidence; everything after
it is the answer model's prose, which Taguru itself never generates.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from taguru import EvidencePackage, Taguru
from taguru_langchain import TaguruIngester

REPO_ROOT = Path(__file__).resolve().parents[4]

# The same fictional-brewery corpus rag_qa/ uses — already a proven
# fixture, and reusing it keeps this example's own code focused on the
# read side, which is what's new here.
DOCS = {
    "docs/aomine/brewery.md": """青嶺酒造は1907年創業の架空の酒蔵である。蔵は岩手県遠野市にある。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。""",
    "docs/aomine/lineup.md": """「青嶺 大吟醸」は精米歩合40%の山田錦で仕込まれる。

「青嶺 純米」は地元米の遠野錦を使う。""",
}

# The decomposition a real chat model would produce under the extract
# discipline — canned per source so the demo runs without an API key.
# The negatively-weighted 大量生産 fact (weight -1.0) is what
# EvidenceItem.contradicts/corroboration would report a disagreement
# over if a second source asserted the opposite; here it demonstrates
# that a denial is stored and surfaced like any other fact, never
# silently dropped.
FAKE_EXTRACTIONS = {
    "docs/aomine/brewery.md": {
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "所在地", "object": "岩手県遠野市", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "paragraph": 2},
        ],
        "aliases": [],
        "questions": [{"paragraph": 1, "question": "青嶺酒造の杜氏は誰?"}],
    },
    "docs/aomine/lineup.md": {
        "associations": [
            {"subject": "青嶺 大吟醸", "label": "精米歩合", "object": "40%", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺 大吟醸", "label": "原料米", "object": "山田錦", "weight": 1.0, "paragraph": 0},
        ],
        "aliases": [],
        "questions": [{"paragraph": 0, "question": "大吟醸の精米歩合はいくつ?"}],
    },
}

QUERY = "青嶺酒造の杜氏は誰ですか?"

FAKE_ANSWER = (
    "杜氏は高瀬です [graph fact]。青嶺酒造は大量生産を行っていません "
    "[docs/aomine/brewery.md ¶2]。"
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


def describe_item(item) -> str:
    """One admitted EvidenceItem, kind-first — the same locator vocabulary
    HitLocator/citation_refs use elsewhere in this tree, never the raw
    fused_score (ADR 0006 §7 never serializes it at all)."""
    if item.passage is not None:
        where = f"{item.passage.source} ¶{item.passage.paragraph}"
        snippet = item.passage.text
    elif item.association is not None:
        assoc = item.association
        where = "graph fact"
        snippet = f"{assoc.subject} —{assoc.label}→ {assoc.object} (weight {assoc.weight:+.1f})"
    elif item.community is not None:
        where = f"{item.community.community} ¶{item.community.paragraph}"
        snippet = item.community.text
    else:
        where, snippet = "?", "?"
    line = f"  [{item.kind:>11} rank={item.fused_rank}] ({where}) {snippet}"
    if item.corroboration is not None:
        line += f"  corroborated by: {sorted(item.corroboration.sources)}"
    if item.contradicts:
        line += f"  contradicts: {item.contradicts}"
    return line


def print_package(label: str, package: EvidencePackage) -> None:
    budget = package.budget
    print(f"-- {label} --")
    print(
        f"  budget: {budget.items_used}/{budget.limits.max_items} items, "
        f"{budget.bytes_used}/{budget.limits.max_bytes} bytes, "
        f"{budget.tokens_used}/{budget.limits.max_tokens} tokens (estimated)"
    )
    print(
        f"  selection: dedup_dropped={package.plan.selection.dedup_dropped} "
        f"contradiction_groups={package.plan.selection.contradiction_groups} "
        f"diversity_tier_width={package.plan.selection.diversity_tier_width}"
    )
    print(f"  reranker: configured={package.plan.reranker.configured} ran={package.plan.reranker.ran}")
    print(f"  omitted: {package.omitted_total} total, by reason: {package.omitted_by_reason}")
    for item in package.items:
        print(describe_item(item))


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
            context="aomine-evidence",
            llm=make_llm([json.dumps(FAKE_EXTRACTIONS[source], ensure_ascii=False) for source in DOCS]),
            client=client,
            create_context=True,
            context_description="青嶺酒造という架空の酒蔵の知識（evidence assembly デモ）",
            questions=1,
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

        # -- read: POST /contexts/{name}/evidence directly, no assembly-lane
        #    intermediary — this is the same call `taguru evaluate --assembly`
        #    drives for the equal-budget comparison documented on
        #    docs/evidence.html. Context.assemble_evidence(), not a bare
        #    client-level call — every read/write method is bound to one
        #    context, named after the server's own vocabulary.
        context = client.context("aomine-evidence")
        print(f"\n== {QUERY} ==")
        generous = context.assemble_evidence(origins=["青嶺酒造"], text_fallback_query=QUERY)
        print_package("generous budget (server defaults)", generous)

        tight = context.assemble_evidence(
            origins=["青嶺酒造"],
            text_fallback_query=QUERY,
            budget={"max_items": 1},
        )
        print_package("tight budget (max_items=1)", tight)

        # -- handing the generous package to an (fake) answer model --------
        evidence_context = "\n".join(describe_item(item) for item in generous.items)
        prompt = (
            f"Context (assembled by Taguru):\n{evidence_context}\n\n"
            f"Question: {QUERY}\n\nAnswer in Japanese using ONLY the context above, "
            "with bracketed citations."
        )
        llm = make_llm([FAKE_ANSWER])
        answer = llm.invoke(prompt).content
        print(f"\nanswer: {answer}")
        print(
            "(everything above 'answer:' is evidence Taguru assembled and budgeted; "
            "everything after it is the answer model's own prose — Taguru itself "
            "never generates it.)"
        )
        return 0
    finally:
        if spawned is not None:
            spawned.stop()


if __name__ == "__main__":
    sys.exit(main())
