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

The retriever addresses one ``context``, several ``contexts``, or ``groups``
(each group reaches every member context, nested children included). Across
several contexts the graph lane runs per context and interleaves by
per-context rank — the posture the server itself takes for passage scores —
and the text lane rides the server's own cross-context search.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr, model_validator
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
    """Retrieve Documents from Taguru contexts, graph lane + text lane.

    Passage-backed hits carry ``page_content`` = the verbatim paragraph and
    metadata ``{context, source, paragraph, section, lane, associations?,
    score?, lanes?}``. Graph facts whose attribution has no stored passage
    still become Documents (``page_content`` = "subject label object") when
    ``include_graph_only_facts`` is on, so pure-graph deployments retrieve too.

    Name at least one target: ``context`` (one name), ``contexts`` (several),
    or ``groups`` (group names — each searches every context it reaches).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Taguru | None = None
    async_client: AsyncTaguru | None = None
    context: str | None = None
    contexts: list[str] | None = None
    groups: list[str] | None = None
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

    _owns_clients: bool = PrivateAttr(default=False)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        owns_clients = "client" not in kwargs and "async_client" not in kwargs
        if owns_clients:
            client_kwargs: dict[str, Any] = {}
            if timeout is not None:
                client_kwargs["timeout"] = timeout
            kwargs["client"] = Taguru(base_url, api_key, **client_kwargs)
            kwargs["async_client"] = AsyncTaguru(base_url, api_key, **client_kwargs)
        super().__init__(**kwargs)
        self._owns_clients = owns_clients

    @model_validator(mode="after")
    def _require_a_target(self) -> TaguruRetriever:
        if self.context is None and not self.contexts and not self.groups:
            raise ValueError("name a target: context, contexts, or groups")
        return self

    def _is_cross(self) -> bool:
        """Whether retrieval spans several contexts (or a group's worth)."""
        return bool(self.contexts) or bool(self.groups)

    def _direct_targets(self) -> list[str]:
        targets: list[str] = []
        if self.context is not None:
            targets.append(self.context)
        for name in self.contexts or []:
            if name not in targets:
                targets.append(name)
        return targets

    @staticmethod
    def _with_members(targets: list[str], members: set[str]) -> list[str]:
        """Direct contexts lead in declaration order; group-resolved members
        follow in name order, overlaps deduped — the server's own
        cross-search tie order."""
        merged = list(targets)
        for name in sorted(members):
            if name not in merged:
                merged.append(name)
        return merged

    # -- sync lane ---------------------------------------------------------

    def _resolve_targets(self, client: Taguru) -> list[str]:
        members: set[str] = set()
        seen: set[str] = set()
        stack = list(self.groups or [])
        while stack:
            group = stack.pop()
            if group in seen:
                continue
            seen.add(group)
            entry = client.groups.get(group)
            members.update(entry.contexts)
            stack.extend(entry.groups)
        return self._with_members(self._direct_targets(), members)

    def _graph_lane(self, client: Taguru, target: str, query: str) -> list[Document]:
        ctx = client.context(target)
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
        if not origins:
            return []
        page = ctx.activate(origins, decay=self.activate_decay, limit=self.activate_limit)
        citations: dict[tuple[str, int], Citation | None] = {}
        for wanted in _wanted_citations(page.matches):
            try:
                citations[wanted] = ctx.cite_passage(*wanted)
            except NotFoundError:
                citations[wanted] = None
        return _graph_documents(page.matches, citations, self.include_graph_only_facts, target)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int | None = None,
    ) -> list[Document]:
        if self.client is None:
            raise ValueError("this retriever was built with only an async client; use ainvoke")
        limit = k if k is not None else self.k

        if not self._is_cross():
            target = self.context
            assert target is not None  # _require_a_target
            graph_docs = self._graph_lane(self.client, target, query) if self.include_graph else []
            text_hits: list[PassageHit] = []
            if self.include_text:
                text_hits = self.client.context(target).search_passages(
                    query, limit=self.text_limit
                )
            return _merge_lanes(graph_docs, text_hits, limit, fallback_context=target)

        targets = self._resolve_targets(self.client)
        graph_docs = []
        if self.include_graph:
            graph_docs = _interleave(
                [self._graph_lane(self.client, target, query) for target in targets]
            )
        text_hits = []
        if self.include_text:
            text_hits = list(
                self.client.search_passages(query, contexts=targets, limit=self.text_limit)
            )
        return _merge_lanes(graph_docs, text_hits, limit)

    # -- async lane ----------------------------------------------------------

    async def _aresolve_targets(self, client: AsyncTaguru) -> list[str]:
        members: set[str] = set()
        seen: set[str] = set()
        # BFS by frontier: the groups of one level are independent
        # fetches, so each level resolves concurrently (nesting is at
        # most 3 levels deep server-side).
        frontier = list(self.groups or [])
        while frontier:
            fetch = [name for name in dict.fromkeys(frontier) if name not in seen]
            seen.update(fetch)
            entries = await asyncio.gather(*(client.groups.get(name) for name in fetch))
            frontier = []
            for entry in entries:
                members.update(entry.contexts)
                frontier.extend(entry.groups)
        return self._with_members(self._direct_targets(), members)

    async def _agraph_lane(self, client: AsyncTaguru, target: str, query: str) -> list[Document]:
        ctx = client.context(target)
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
        if not origins:
            return []
        page = await ctx.activate(origins, decay=self.activate_decay, limit=self.activate_limit)
        citations: dict[tuple[str, int], Citation | None] = {}
        for wanted in _wanted_citations(page.matches):
            try:
                citations[wanted] = await ctx.cite_passage(*wanted)
            except NotFoundError:
                citations[wanted] = None
        return _graph_documents(page.matches, citations, self.include_graph_only_facts, target)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: int | None = None,
    ) -> list[Document]:
        if self.async_client is None:
            raise ValueError("this retriever was built with only a sync client; use invoke")
        limit = k if k is not None else self.k

        if not self._is_cross():
            target = self.context
            assert target is not None  # _require_a_target
            graph_docs = (
                await self._agraph_lane(self.async_client, target, query)
                if self.include_graph
                else []
            )
            text_hits: list[PassageHit] = []
            if self.include_text:
                text_hits = await self.async_client.context(target).search_passages(
                    query, limit=self.text_limit
                )
            return _merge_lanes(graph_docs, text_hits, limit, fallback_context=target)

        targets = await self._aresolve_targets(self.async_client)
        graph_docs = []
        if self.include_graph:
            # Each target's lane is an independent chain of round
            # trips, and completion order is irrelevant (_interleave
            # sorts by rank, then target order) — run them concurrently.
            per_target = await asyncio.gather(
                *(self._agraph_lane(self.async_client, target, query) for target in targets)
            )
            graph_docs = _interleave(list(per_target))
        text_hits = []
        if self.include_text:
            text_hits = list(
                await self.async_client.search_passages(
                    query, contexts=targets, limit=self.text_limit
                )
            )
        return _merge_lanes(graph_docs, text_hits, limit)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the sync HTTP client, if this retriever built it itself.

        A client passed in via ``client``/``async_client`` stays the
        caller's to close. The async client needs a running event loop to
        close cleanly; use :meth:`aclose` (or ``async with``) once the
        async lane (``ainvoke``) has been used.
        """
        if self._owns_clients and self.client is not None:
            self.client.close()

    async def aclose(self) -> None:
        """Close both the sync and async HTTP clients this retriever owns."""
        if self._owns_clients:
            if self.client is not None:
                self.client.close()
            if self.async_client is not None:
                await self.async_client.close()

    def __enter__(self) -> TaguruRetriever:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def __aenter__(self) -> TaguruRetriever:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


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
    context: str,
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
                            "context": context,
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
                        "context": context,
                        "source": best_source,
                        "paragraph": None,
                        "section": None,
                        "lane": "graph",
                        "associations": [_association_meta(activation)],
                    },
                )
            )
    return ordered


def _interleave(per_target: list[list[Document]]) -> list[Document]:
    """Per-context rank interleaving — activation strengths are ordinal
    within one call only, so ranks are the currency across contexts (the
    posture the server's own cross-context passage merge takes)."""
    indexed = [
        (rank, index, document)
        for index, documents in enumerate(per_target)
        for rank, document in enumerate(documents)
    ]
    indexed.sort(key=lambda entry: (entry[0], entry[1]))
    return [document for _rank, _index, document in indexed]


def _hit_context(hit: PassageHit, fallback: str | None) -> str | None:
    """A cross-context hit names its context; a per-context hit inherits the
    retriever's own target."""
    return getattr(hit, "context", None) or fallback


def _text_document(hit: PassageHit, context: str | None) -> Document:
    lanes: dict[str, Any] = {}
    if hit.lanes.bm25 is not None:
        lanes["bm25"] = {"rank": hit.lanes.bm25.rank, "score": hit.lanes.bm25.score}
    if hit.lanes.vector is not None:
        lanes["vector"] = {"rank": hit.lanes.vector.rank, "score": hit.lanes.vector.score}
    return Document(
        page_content=hit.text,
        metadata={
            "context": context,
            "source": hit.source,
            "paragraph": hit.paragraph,
            "section": None,
            "lane": "text",
            "score": hit.score,
            "lanes": lanes,
        },
    )


def _merge_lanes(
    graph_docs: list[Document],
    text_hits: list[PassageHit],
    k: int,
    fallback_context: str | None = None,
) -> list[Document]:
    """Reciprocal Rank Fusion across lanes; (context, source, paragraph) hits
    landing in both collapse into one Document tagged ``lane: "graph+text"``."""
    scored: dict[object, tuple[float, Document]] = {}

    def key_of(document: Document) -> object:
        context = document.metadata.get("context")
        if document.metadata.get("paragraph") is not None:
            return (context, document.metadata["source"], document.metadata["paragraph"])
        associations = document.metadata.get("associations") or [{}]
        first = associations[0]
        return ("fact", context, first.get("subject"), first.get("label"), first.get("object"))

    for rank, document in enumerate(graph_docs):
        scored[key_of(document)] = (_rrf(rank), document)

    for rank, hit in enumerate(text_hits):
        context = _hit_context(hit, fallback_context)
        key = (context, hit.source, hit.paragraph)
        existing = scored.get(key)
        if existing is None:
            scored[key] = (_rrf(rank), _text_document(hit, context))
        else:
            score, document = existing
            document.metadata["lane"] = "graph+text"
            document.metadata["score"] = hit.score
            document.metadata["lanes"] = _text_document(hit, context).metadata["lanes"]
            scored[key] = (score + _rrf(rank), document)

    ranked = sorted(scored.values(), key=lambda entry: entry[0], reverse=True)
    return [document for _score, document in ranked[:k]]
