"""Local RAG over PDFs: TaguruIngester (extract) -> Taguru server (embed) ->
TaguruRetriever + an LCEL chain (answer), fully local end to end.

Two short fictional papers (tanaka2024, sato2023) go in as PDFs, get split
into numbered sections by a plain regex, and each section becomes its own
Taguru context — TaguruIngester(context=...) switched per section — bundled
under its paper's group. TaguruRetriever then searches both papers' groups
at once, and a separate LCEL chain answers the question; retrieval and
generation are printed as two distinct phases on purpose.

Runs self-contained: with no TAGURU_URL set it spawns a real server binary
(builds it with cargo on first run), and with no OLLAMA_MODEL set it drives
every LLM role with a deterministic fake model — no Ollama required to see
the wiring work. Point OLLAMA_MODEL at a model already pulled locally
(check with `ollama list`) for the real thing; this script never pulls one
for you. Set TAGURU_EMBED_URL/TAGURU_EMBED_MODEL on the server (see the
walkthrough) to exercise the semantic lane too — retrieval and citations
below work the same either way, on BM25 and the graph alone if not.

    cd examples/langchain && .venv/bin/python local_rag/python/main.py

Full walkthrough, prerequisites, and the modeling behind the context/group
choice: https://t0k0sh1.github.io/taguru/local-rag-walkthrough.html
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pypdf import PdfReader
from taguru import ConflictError, Taguru
from taguru_langchain import TaguruIngester, TaguruRetriever

REPO_ROOT = Path(__file__).resolve().parents[4]
PAPERS_DIR = Path(__file__).resolve().parent.parent / "papers"

SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)

# paper id -> a human-readable byline, used only for group descriptions and
# building CITATION_LABELS below — Taguru itself never sees the byline.
PAPERS = {
    "tanaka2024": "Tanaka et al. 2024",
    "sato2023": "Sato et al. 2023",
}

# The decomposition a real chat model would produce under the extract
# discipline, canned per section so the demo runs without a local LLM.
# Keys are the source id each section is ingested under: "{paper}/{n}".
FAKE_EXTRACTIONS = {
    "tanaka2024/1": {
        "associations": [
            {
                "subject": "ginjo aroma",
                "label": "main component",
                "object": "isoamyl acetate",
                "weight": 1.0,
                "paragraph": 0,
            },
        ],
        "aliases": [],
    },
    "tanaka2024/2": {
        "associations": [
            {
                "subject": "tanaka2024 experiment",
                "label": "yeast strain used",
                "object": "Kyokai No. 901",
                "weight": 1.0,
                "paragraph": 0,
            },
        ],
        "aliases": [],
    },
    "tanaka2024/3": {
        "associations": [
            {
                "subject": "low-temperature long fermentation",
                "label": "effect on isoamyl acetate",
                "object": "about 1.8x higher",
                "weight": 1.0,
                "paragraph": 0,
            },
            {
                "subject": "low-temperature long fermentation",
                "label": "effect on yield",
                "object": "somewhat lower",
                "weight": 1.0,
                "paragraph": 0,
            },
        ],
        "aliases": [],
    },
    "sato2023/1": {
        "associations": [
            {
                "subject": "koji-mold protease activity",
                "label": "directly affects",
                "object": "amino acid content",
                "weight": 1.0,
                "paragraph": 0,
            },
        ],
        "aliases": [],
    },
    "sato2023/2": {
        "associations": [
            {
                "subject": "sato2023 strain comparison",
                "label": "strain count",
                "object": "three",
                "weight": 1.0,
                "paragraph": 0,
            },
        ],
        "aliases": [],
    },
}

QUESTION = "How does low-temperature fermentation affect ginjo aroma yield?"

FAKE_ANSWER = (
    "Low-temperature long fermentation raises isoamyl acetate concentration by "
    "about 1.8x [tanaka2024/3], though the trade-off is a somewhat lower yield "
    "[tanaka2024/3]."
)

# Taguru's API only ever deals in the source id (a machine key, chosen at
# ingest time). A human-readable citation label like "Tanaka et al. 2024,
# §3" is not an API concept at all — this mapping is entirely this script's
# own bookkeeping, built once from whatever a paper's front matter carries.
CITATION_LABELS = {source: f"{PAPERS[source.split('/')[0]]}, §{source.split('/')[1]}" for source in FAKE_EXTRACTIONS}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the facts in the context below. Cite every claim "
            "with its bracketed source id, e.g. [tanaka2024/3].\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def pdf_to_sections(pdf_path: Path, paper_id: str):
    """One dict per numbered section: paper, section number, title, text."""
    pages = PdfReader(str(pdf_path)).pages
    text = "\n".join(page.extract_text() for page in pages)
    marks = list(SECTION_RE.finditer(text))
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        yield {
            "paper": paper_id,
            "n": int(m.group(1)),
            "title": m.group(2).strip(),
            "text": text[start:end].strip(),
        }


def ensure_group(client: Taguru, name: str, description: str, contexts: list[str]) -> None:
    """Create the group, or fold new contexts into an existing one — group
    creation 409s on a rerun, so the idempotence has to be handled here."""
    try:
        client.groups.create(name, description=description, contexts=contexts)
    except ConflictError:
        client.groups.update(name, add_contexts=contexts)


def make_llm(fake_responses: list[str]) -> BaseChatModel:
    """A real local Ollama model when OLLAMA_MODEL is set, else a canned fake."""
    model = os.environ.get("OLLAMA_MODEL")
    if model:
        try:
            from langchain_ollama import ChatOllama
        except ModuleNotFoundError:
            print("(OLLAMA_MODEL set but langchain-ollama not installed — using the fake model)")
        else:
            return ChatOllama(model=model, temperature=0)
    else:
        print("(no OLLAMA_MODEL — using a canned fake model; this script never pulls one)")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=fake_responses)


def format_docs(documents: list[Document]) -> str:
    """Retrieved Documents -> the context block, source-id-first so the
    model has something concrete to cite."""
    return "\n".join(f"[{document.metadata['source']}] {document.page_content}" for document in documents)


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

        # -- write: one PDF -> N sections -> N contexts, one paper -> one group --
        all_sections = [
            section
            for paper_id in PAPERS
            for section in pdf_to_sections(PAPERS_DIR / f"{paper_id}.pdf", paper_id)
        ]
        extract_llm = make_llm(
            [json.dumps(FAKE_EXTRACTIONS[f"{s['paper']}/{s['n']}"]) for s in all_sections]
        )

        paper_contexts: dict[str, list[str]] = {paper_id: [] for paper_id in PAPERS}
        for section in all_sections:
            context = f"section/{section['paper']}/{section['n']}"
            ingester = TaguruIngester(
                context=context,  # switches every section
                llm=extract_llm,
                client=client,
                create_context=True,
                context_description=f"{section['paper']} §{section['n']} — {section['title']}",
            )
            doc = Document(
                page_content=section["text"],
                metadata={"source": f"{section['paper']}/{section['n']}"},
            )
            outcome = ingester.ingest_documents([doc])[0]
            if not outcome.ok:
                print(f"FAILED to ingest {outcome.source}: {outcome.error}", file=sys.stderr)
                return 1
            print(f"ingested {context}: {outcome.associations} facts, {outcome.aliases} aliases")
            paper_contexts[section["paper"]].append(context)

        for paper_id, byline in PAPERS.items():
            ensure_group(client, f"paper/{paper_id}", f"{byline}, full paper", paper_contexts[paper_id])

        # -- read: one retriever across both papers' groups, an independent answer model --
        retriever = TaguruRetriever(client=client, groups=[f"paper/{p}" for p in PAPERS], k=8)
        answer_llm = make_llm([FAKE_ANSWER])
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | PROMPT
            | answer_llm
            | StrOutputParser()
        )

        print(f"\n== {QUESTION} ==")
        # Phase 1: what retrieval brought back — shown before any generation runs.
        for document in retriever.invoke(QUESTION):
            meta = document.metadata
            label = CITATION_LABELS.get(meta["source"], meta["source"])
            where = f"{label} ¶{meta['paragraph']}" if meta.get("paragraph") is not None else label
            print(f"  [{meta['lane']:>10}] {where}: {document.page_content[:70]}...")

        # Phase 2: the answer, generated from exactly those retrieved documents.
        print(f"  answer: {chain.invoke(QUESTION)}")

        # Trace one claim in the answer back to its original PDF paragraph.
        citation = client.context("section/tanaka2024/3").cite_passage("tanaka2024/3", 0)
        print(f"\n  cited passage (tanaka2024 §3, ¶0): {citation.text[:120]}...")
        return 0
    finally:
        if spawned is not None:
            spawned.stop()


if __name__ == "__main__":
    sys.exit(main())
