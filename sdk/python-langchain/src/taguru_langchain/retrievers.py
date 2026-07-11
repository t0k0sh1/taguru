"""TaguruRetriever: the documented retrieval loop as a LangChain retriever.

The query string is treated as ONE opaque cue — decomposing a multi-entity
question, rephrasing it into the declarative shape the text lane prefers, and
judging containment/fuzzy lookalikes by their gloss are the calling LLM's
job; chain a query-rewriting step in front of this retriever for those.

Both lanes run by default (graph: resolve → activate → citations; text:
search_passages) and merge by Reciprocal Rank Fusion — activate's ``strength``
and search's ``score`` are each meaningful only within one call, so ranks are
the only comparable currency. Hits landing in both lanes collapse into one
Document tagged ``lane: "graph+text"``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from taguru import (
    Activation,
    AsyncTaguru,
    Citation,
    NotFoundError,
    PassageHit,
    Taguru,
)

RRF_K = 60


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


class TaguruRetriever(BaseRetriever):
    """Retrieve Documents from one Taguru context, graph lane + text lane.

    Passage-backed hits carry ``page_content`` = the verbatim paragraph and
    metadata ``{source, paragraph, section, lane, associations?, score?,
    lanes?}``. Graph facts whose attribution has no stored passage still
    become Documents (``page_content`` = "subject label object") when
    ``include_graph_only_facts`` is on, so pure-graph deployments retrieve too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Taguru | None = None
    async_client: AsyncTaguru | None = None
    context: str
    k: int = 8
    include_graph: bool = True
    include_text: bool = True
    include_graph_only_facts: bool = True
    activate_decay: float | None = None
    activate_limit: int = 20
    text_limit: int = 5
    resolve_limit: int = 5
    dice_floor: float | None = None
    semantic_floor: float | None = None

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        if "client" not in kwargs and "async_client" not in kwargs:
            client_kwargs: dict[str, Any] = {}
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            kwargs["client"] = Taguru(base_url, api_key, **client_kwargs)
            kwargs["async_client"] = AsyncTaguru(base_url, api_key, **client_kwargs)
        super().__init__(**kwargs)

    # -- sync lane ---------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int | None = None,
    ) -> list[Document]:
        if self.client is None:
            raise ValueError("this retriever was built with only an async client; use ainvoke")
        ctx = self.client.context(self.context)

        graph_docs: list[Document] = []
        if self.include_graph:
            candidates = ctx.resolve(
                query,
                dice_floor=self.dice_floor,
                semantic_floor=self.semantic_floor,
                limit=self.resolve_limit,
            )
            origins: list[str] = []
            for candidate in candidates:
                if candidate.name not in origins:
                    origins.append(candidate.name)
            if origins:
                page = ctx.activate(origins, decay=self.activate_decay, limit=self.activate_limit)
                citations: dict[tuple[str, int], Citation | None] = {}
                for wanted in _wanted_citations(page.matches):
                    try:
                        citations[wanted] = ctx.cite_passage(*wanted)
                    except NotFoundError:
                        citations[wanted] = None
                graph_docs = _graph_documents(
                    page.matches, citations, self.include_graph_only_facts
                )

        text_hits: list[PassageHit] = []
        if self.include_text:
            text_hits = ctx.search_passages(query, limit=self.text_limit)

        return _merge_lanes(graph_docs, text_hits, k if k is not None else self.k)

    # -- async lane ----------------------------------------------------------

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: int | None = None,
    ) -> list[Document]:
        if self.async_client is None:
            raise ValueError("this retriever was built with only a sync client; use invoke")
        ctx = self.async_client.context(self.context)

        graph_docs: list[Document] = []
        if self.include_graph:
            candidates = await ctx.resolve(
                query,
                dice_floor=self.dice_floor,
                semantic_floor=self.semantic_floor,
                limit=self.resolve_limit,
            )
            origins: list[str] = []
            for candidate in candidates:
                if candidate.name not in origins:
                    origins.append(candidate.name)
            if origins:
                page = await ctx.activate(
                    origins, decay=self.activate_decay, limit=self.activate_limit
                )
                citations: dict[tuple[str, int], Citation | None] = {}
                for wanted in _wanted_citations(page.matches):
                    try:
                        citations[wanted] = await ctx.cite_passage(*wanted)
                    except NotFoundError:
                        citations[wanted] = None
                graph_docs = _graph_documents(
                    page.matches, citations, self.include_graph_only_facts
                )

        text_hits: list[PassageHit] = []
        if self.include_text:
            text_hits = await ctx.search_passages(query, limit=self.text_limit)

        return _merge_lanes(graph_docs, text_hits, k if k is not None else self.k)


# -- lane assembly (pure functions shared by both entry points) -----------------


def _wanted_citations(activations: list[Activation]) -> list[tuple[str, int]]:
    wanted: list[tuple[str, int]] = []
    for activation in activations:
        for attribution in activation.association.attributions:
            if attribution.paragraph is None:
                continue
            key = (attribution.source, attribution.paragraph)
            if key not in wanted:
                wanted.append(key)
    return wanted


def _association_meta(activation: Activation) -> dict[str, Any]:
    association = activation.association
    return {
        "subject": association.subject,
        "label": association.label,
        "object": association.object,
        "weight": association.weight,
        "strength": activation.strength,
        "path": list(activation.path),
    }


def _graph_documents(
    activations: list[Activation],
    citations: dict[tuple[str, int], Citation | None],
    include_graph_only_facts: bool,
) -> list[Document]:
    """Activation matches → Documents, ranked by activation order (strongest
    first). Located attributions with a stored passage become passage-backed
    Documents (several associations citing one paragraph collapse into one
    Document); everything else becomes a graph-only fact Document."""
    passage_docs: dict[tuple[str, int], Document] = {}
    ordered: list[Document] = []
    for activation in activations:
        association = activation.association
        located: list[tuple[str, int]] = []
        for attribution in association.attributions:
            if attribution.paragraph is None:
                continue
            key = (attribution.source, attribution.paragraph)
            if citations.get(key) is not None:
                located.append(key)
        if located:
            for key in located:
                citation = citations[key]
                assert citation is not None
                existing = passage_docs.get(key)
                if existing is None:
                    document = Document(
                        page_content=citation.text,
                        metadata={
                            "source": key[0],
                            "paragraph": key[1],
                            "section": citation.section,
                            "lane": "graph",
                            "associations": [_association_meta(activation)],
                        },
                    )
                    passage_docs[key] = document
                    ordered.append(document)
                else:
                    existing.metadata["associations"].append(_association_meta(activation))
        elif include_graph_only_facts:
            # Real, retrievable knowledge with no verbatim excerpt to ground
            # it in — a triple is already a composed, atomic fact.
            best_source = association.attributions[0].source if association.attributions else None
            ordered.append(
                Document(
                    page_content=f"{association.subject} {association.label} {association.object}",
                    metadata={
                        "source": best_source,
                        "paragraph": None,
                        "section": None,
                        "lane": "graph",
                        "associations": [_association_meta(activation)],
                    },
                )
            )
    return ordered


def _text_document(hit: PassageHit) -> Document:
    lanes: dict[str, Any] = {}
    if hit.lanes.bm25 is not None:
        lanes["bm25"] = {"rank": hit.lanes.bm25.rank, "score": hit.lanes.bm25.score}
    if hit.lanes.vector is not None:
        lanes["vector"] = {"rank": hit.lanes.vector.rank, "score": hit.lanes.vector.score}
    return Document(
        page_content=hit.text,
        metadata={
            "source": hit.source,
            "paragraph": hit.paragraph,
            "section": None,
            "lane": "text",
            "score": hit.score,
            "lanes": lanes,
        },
    )


def _merge_lanes(graph_docs: list[Document], text_hits: list[PassageHit], k: int) -> list[Document]:
    """Reciprocal Rank Fusion across lanes; (source, paragraph) hits landing
    in both collapse into one Document tagged ``lane: "graph+text"``."""
    scored: dict[object, tuple[float, Document]] = {}

    def key_of(document: Document) -> object:
        if document.metadata.get("paragraph") is not None:
            return (document.metadata["source"], document.metadata["paragraph"])
        associations = document.metadata.get("associations") or [{}]
        first = associations[0]
        return ("fact", first.get("subject"), first.get("label"), first.get("object"))

    for rank, document in enumerate(graph_docs):
        scored[key_of(document)] = (_rrf(rank), document)

    for rank, hit in enumerate(text_hits):
        key = (hit.source, hit.paragraph)
        existing = scored.get(key)
        if existing is None:
            scored[key] = (_rrf(rank), _text_document(hit))
        else:
            score, document = existing
            document.metadata["lane"] = "graph+text"
            document.metadata["score"] = hit.score
            document.metadata["lanes"] = _text_document(hit).metadata["lanes"]
            scored[key] = (score + _rrf(rank), document)

    ranked = sorted(scored.values(), key=lambda entry: entry[0], reverse=True)
    return [document for _score, document in ranked[:k]]
