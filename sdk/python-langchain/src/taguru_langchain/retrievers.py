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

The retriever addresses one ``context_id``, several ``context_ids``, or ``group_ids``
(each ``group`` reaches every member ``context``, nested children included). Across
several ``contexts`` the graph lane runs per ``context`` and interleaves by
per-``context`` rank — the posture the server itself takes for passage scores —
and the text lane rides the server's own cross-``context`` search, naming the
``groups`` themselves so the server expands them (a ``group`` may reach more
``contexts`` than one request may list).

Every target is an id: ``context_id``/``context_ids`` take the ``id`` column of
``client.contexts.list()`` (#964, #965, #967) and ``group_ids`` the ``id`` column of
``client.groups.list()`` (#997). Look a name up with ``taguru contexts --name NAME`` /
``taguru groups --name NAME``, or filter the listing on ``name``. Every Document's
``context`` metadata is a context id.
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
    GroupEntry,
    NotFoundError,
    PassageHit,
    Taguru,
)

RRF_K = 60


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def _next_storey(entries: list[GroupEntry | None], seen: set[str], members: set[str]) -> list[str]:
    """Folds one storey of fetched ``group`` rows into ``members`` and answers
    the child ``group`` ids to fetch next — each id once, however many parents
    name it (``seen`` carries every id already asked for). A ``None`` entry is a
    ``group`` that answered 404 (deleted): it contributes nothing, and the rest
    of the walk still stands — one unresolvable ``group`` must not blank out
    every other member."""
    children: list[str] = []
    for entry in entries:
        if entry is None:
            continue
        members.update(entry.context_ids)
        for child in entry.group_ids:
            if child not in seen:
                seen.add(child)
                children.append(child)
    return children


class TaguruRetriever(BaseRetriever):
    """Retrieve Documents from Taguru ``contexts``, graph lane + text lane.

    Passage-backed hits carry ``page_content`` = the verbatim paragraph and
    metadata ``{context, source, paragraph, section, lane, associations?,
    score?, lanes?}``. Graph facts whose attribution has no stored passage
    still become Documents (``page_content`` = "subject label object") when
    ``include_graph_only_facts`` is on, so pure-graph deployments retrieve too.

    Name at least one target: ``context_id`` (one context id — the ``id``
    column of ``contexts.list()``), ``context_ids`` (several context ids), or
    ``group_ids`` (``group`` ids — the ``id`` column of ``groups.list()``; each
    searches every ``context`` it reaches). Display names are not accepted
    anywhere (#965, #997).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Taguru | None = None
    async_client: AsyncTaguru | None = None
    context_id: str | None = None
    context_ids: list[str] | None = None
    group_ids: list[str] | None = None
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
        if self.context_id is None and not self.context_ids and not self.group_ids:
            raise ValueError("name a target: context_id, context_ids, or group_ids")
        return self

    def _is_cross(self) -> bool:
        """Whether retrieval spans several ``contexts`` (or a ``group``'s worth)."""
        return bool(self.context_ids) or bool(self.group_ids)

    def _direct_targets(self) -> list[str]:
        targets: list[str] = []
        if self.context_id is not None:
            targets.append(self.context_id)
        for context_id in self.context_ids or []:
            if context_id not in targets:
                targets.append(context_id)
        return targets

    @staticmethod
    def _with_members(targets: list[str], members: set[str]) -> list[str]:
        """Direct ``contexts`` lead in declaration order; ``group``-resolved members
        follow in id order, overlaps deduped — the server's own
        cross-search tie order."""
        merged = list(targets)
        for context_id in sorted(members):
            if context_id not in merged:
                merged.append(context_id)
        return merged

    # -- sync lane ---------------------------------------------------------

    def _resolve_targets(self, client: Taguru) -> tuple[list[str], list[str]]:
        """The graph lane's per-``context`` targets, and the text lane's
        ``group_ids`` — the named ``groups`` minus any that answered 404.

        The text lane sends ``groups`` rather than their expansion: one
        ``group`` may reach far more ``contexts`` than a request's
        ``context_ids`` cap (1000), and an over-cap list is a 400 that
        would blank the lane outright. The server expands them itself, in
        the same direct-first, members-in-id-order tie order."""
        if not self.group_ids:
            return self._direct_targets(), []
        # Walk down from the named ``groups``, one ``groups.get`` per ``group``
        # reached (#1024): only the rows this search needs are read — never
        # the whole directory, whose every row costs the server a
        # fingerprint. A walk that fails partway keeps the members already
        # reached beside the direct targets — every sibling of a failed
        # ``group`` is still fetched and counted, so the outcome does not hang
        # on fetch order — and with no direct target, the failure is the
        # caller's to hear, never an empty search that looks complete.
        direct = self._direct_targets()
        members: set[str] = set()
        missing: set[str] = set()
        storey = list(dict.fromkeys(self.group_ids))
        seen = set(storey)
        try:
            while storey:
                entries: list[GroupEntry | None] = []
                failure: Exception | None = None
                for group_id in storey:
                    try:
                        entries.append(client.groups.get(group_id))
                    except NotFoundError:
                        missing.add(group_id)
                        entries.append(None)
                    except Exception as error:
                        failure = failure or error
                storey = _next_storey(entries, seen, members)
                if failure is not None:
                    raise failure
        except Exception:
            if not direct:
                raise
        return self._with_members(direct, members), self._text_groups(missing)

    def _text_groups(self, missing: set[str]) -> list[str]:
        """The named ``groups`` the text lane sends: a deleted one would
        refuse the whole search (``no_group``), so it drops out exactly as
        it contributes no graph targets."""
        named = dict.fromkeys(self.group_ids or [])
        return [group_id for group_id in named if group_id not in missing]

    def _graph_lane(self, client: Taguru, target: str, query: str) -> list[Document]:
        """``target`` is the context ID, which the Documents' metadata carries
        too (matching the text lane's own ``context_id`` tags)."""
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
            target = self.context_id
            assert target is not None  # _require_a_target
            graph_docs: list[Document] = []
            graph_error: Exception | None = None
            if self.include_graph:
                try:
                    graph_docs = self._graph_lane(self.client, target, query)
                except Exception as error:
                    # Isolated exactly like the multi-context branch
                    # isolates each target's graph lane below: one lane
                    # failing must not blank out whatever the other lane
                    # already found.
                    graph_error = error
            text_hits: list[PassageHit] = []
            text_error: Exception | None = None
            if self.include_text:
                try:
                    text_hits = (
                        self.client.context(target)
                        .search_passages(query, limit=self.text_limit)
                        .hits
                    )
                except Exception as error:
                    text_error = error
            # "Every ENABLED lane failed" — a disabled lane is not a
            # healthy one, so with a single lane enabled its failure alone
            # must raise: an empty result here would read as "nothing
            # found" instead of "retrieval is broken."
            _raise_if_every_lane_failed(
                self.include_graph and graph_error is None,
                graph_error,
                self.include_text and text_error is None,
                text_error,
            )
            return _merge_lanes(graph_docs, text_hits, limit, fallback_context=target)

        targets, text_groups = self._resolve_targets(self.client)
        direct = self._direct_targets()
        graph_docs = []
        graph_ok = False
        graph_error = None
        if self.include_graph:
            # One target erroring (a deleted context, a transient
            # failure) should not blank out the graph docs every other
            # target already found — it keeps its slot with no docs, so
            # the interleave order holds.
            per_target: list[list[Document]] = []
            for target in targets:
                try:
                    per_target.append(self._graph_lane(self.client, target, query))
                    graph_ok = True
                except Exception as error:
                    graph_error = graph_error or error
                    per_target.append([])
            graph_docs = _interleave(per_target)
        text_hits = []
        text_error = None
        if self.include_text and (direct or text_groups):
            # The server's cross_targets validation rejects the WHOLE
            # request if even one named context is gone — isolated so a
            # single deleted context does not also discard the graph
            # lane's results (see the per-target isolation just above).
            try:
                text_hits = list(
                    self.client.search_passages(
                        query,
                        context_ids=direct or None,
                        group_ids=text_groups or None,
                        limit=self.text_limit,
                    ).hits
                )
            except Exception as error:
                text_error = error
        # Isolation is per target, not a licence to hide an outage: when
        # NO target's graph lane answered and the text lane failed too
        # (401, server down), an empty list would read as "nothing found".
        _raise_if_every_lane_failed(
            graph_ok, graph_error, self.include_text and text_error is None, text_error
        )
        return _merge_lanes(graph_docs, text_hits, limit)

    # -- async lane ----------------------------------------------------------

    async def _aresolve_targets(self, client: AsyncTaguru) -> tuple[list[str], list[str]]:
        """:meth:`_resolve_targets`' async twin, fetching each storey's
        ``groups`` concurrently, so the round trips follow the nesting depth
        (at most 3), not the ``group`` count. ``except Exception`` lets a
        BaseException that is NOT an Exception — CancelledError above all —
        propagate, never swallowed as an unreadable ``group``."""
        if not self.group_ids:
            return self._direct_targets(), []
        direct = self._direct_targets()
        members: set[str] = set()
        missing: set[str] = set()
        storey = list(dict.fromkeys(self.group_ids))
        seen = set(storey)

        async def _fetch_one(group_id: str) -> GroupEntry | None:
            try:
                return await client.groups.get(group_id)
            except NotFoundError:
                missing.add(group_id)
                return None

        try:
            while storey:
                # `return_exceptions` waits for every sibling (none is left
                # orphaned) and keeps the successes of a storey one fetch
                # failed in; cancelling this task still cancels the gather.
                results = await asyncio.gather(
                    *(_fetch_one(group_id) for group_id in storey), return_exceptions=True
                )
                entries = [r for r in results if not isinstance(r, BaseException)]
                storey = _next_storey(entries, seen, members)
                failures = [r for r in results if isinstance(r, BaseException)]
                if failures:
                    # A sibling's cancellation outranks a plain failure.
                    raise next((f for f in failures if not isinstance(f, Exception)), failures[0])
        except Exception:
            if not direct:
                raise
        return self._with_members(direct, members), self._text_groups(missing)

    async def _agraph_lane(self, client: AsyncTaguru, target: str, query: str) -> list[Document]:
        """:meth:`_graph_lane`'s async twin."""
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
        wanted_pairs = _wanted_citations(page.matches)

        async def _fetch_one(pair: tuple[str, int]) -> Citation | None:
            # Wrapped per-pair (not gather's return_exceptions=True) so a
            # 404 on one citation resolves to None without also having to
            # sift real errors out of a mixed exceptions/results list.
            try:
                return await ctx.cite_passage(*pair)
            except NotFoundError:
                return None

        # `gather` raises on the first non-404 error but does NOT cancel the
        # siblings — they would run on as orphaned tasks, surfacing later as
        # "Task exception was never retrieved" once the client is closed. To
        # match the sync lane's sequential loop (which simply never starts
        # the later lookups once one raises), cancel and drain the rest
        # before re-raising.
        tasks = [asyncio.ensure_future(_fetch_one(pair)) for pair in wanted_pairs]
        try:
            fetched = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        citations: dict[tuple[str, int], Citation | None] = dict(
            zip(wanted_pairs, fetched, strict=True)
        )
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
            target = self.context_id
            assert target is not None  # _require_a_target
            graph_docs: list[Document] = []
            graph_error: Exception | None = None
            if self.include_graph:
                try:
                    graph_docs = await self._agraph_lane(self.async_client, target, query)
                except Exception as error:
                    # Same isolation as the sync lane above: one lane
                    # failing must not blank out whatever the other lane
                    # already found.
                    graph_error = error
            text_hits: list[PassageHit] = []
            text_error: Exception | None = None
            if self.include_text:
                try:
                    single = await self.async_client.context(target).search_passages(
                        query, limit=self.text_limit
                    )
                    text_hits = single.hits
                except Exception as error:
                    text_error = error
            # Same rule as the sync path above: every ENABLED lane failing
            # must raise, including when only one lane is enabled.
            _raise_if_every_lane_failed(
                self.include_graph and graph_error is None,
                graph_error,
                self.include_text and text_error is None,
                text_error,
            )
            return _merge_lanes(graph_docs, text_hits, limit, fallback_context=target)

        targets, text_groups = await self._aresolve_targets(self.async_client)
        direct = self._direct_targets()
        graph_docs = []
        graph_ok = False
        graph_error = None
        if self.include_graph:
            # Each target's lane is an independent chain of round
            # trips, and completion order is irrelevant (_interleave
            # sorts by rank, then target order) — run them concurrently.
            # return_exceptions=True: one target erroring (a deleted
            # context, a transient failure) should not blank out the
            # graph docs every other target already found.
            client = self.async_client
            per_target = await asyncio.gather(
                *(self._agraph_lane(client, target, query) for target in targets),
                return_exceptions=True,
            )
            lanes: list[list[Document]] = []
            for docs in per_target:
                if isinstance(docs, BaseException):
                    # As in `_aresolve_targets`: a plain failure blanks only
                    # that one target's lane, but a non-Exception
                    # BaseException (CancelledError) must propagate.
                    if not isinstance(docs, Exception):
                        raise docs
                    graph_error = graph_error or docs
                    lanes.append([])
                else:
                    graph_ok = True
                    lanes.append(docs)
            graph_docs = _interleave(lanes)
        text_hits = []
        text_error = None
        if self.include_text and (direct or text_groups):
            # Same isolation as the sync lane above: one gone target must
            # not blank out the graph lane's already-fetched results.
            try:
                crossed = await self.async_client.search_passages(
                    query,
                    context_ids=direct or None,
                    group_ids=text_groups or None,
                    limit=self.text_limit,
                )
                text_hits = list(crossed.hits)
            except Exception as error:
                text_error = error
        # Same rule as the sync path: every lane failing raises.
        _raise_if_every_lane_failed(
            graph_ok, graph_error, self.include_text and text_error is None, text_error
        )
        return _merge_lanes(graph_docs, text_hits, limit)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP client(s) this retriever built itself, best-effort.

        A client passed in via ``client``/``async_client`` stays the
        caller's to close. The sync client always closes cleanly here;
        closing the async client needs a running event loop, so one is
        spun up just for that when none is already running. Called from
        inside a running loop, the async client is left for :meth:`aclose`
        instead.
        """
        if not self._owns_clients:
            return
        if self.client is not None:
            self.client.close()
        if self.async_client is not None:
            try:
                asyncio.run(self.async_client.close())
            except RuntimeError:
                pass

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


def _raise_if_every_lane_failed(
    graph_ok: bool,
    graph_error: Exception | None,
    text_ok: bool,
    text_error: Exception | None,
) -> None:
    """Raise when some lane failed and none succeeded — a disabled lane
    (``*_ok`` False) is not a healthy one, so with one lane enabled its
    failure alone raises. ``graph_ok`` means at least one graph target
    answered: partial failures stay isolated, but an empty result built
    from nothing but failures would read as "nothing found" instead of
    "retrieval is broken." The graph error wins when both failed."""
    if graph_ok or text_ok:
        return
    if graph_error is not None:
        raise graph_error
    if text_error is not None:
        raise text_error


def _wanted_citations(activations: list[Activation]) -> list[tuple[str, int]]:
    wanted: list[tuple[str, int]] = []
    for activation in activations:
        for attribution in activation.association.attributions:
            if attribution.paragraph is None:
                continue
            key = (attribution.source_id, attribution.paragraph)
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
            key = (attribution.source_id, attribution.paragraph)
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
            best_source = (
                association.attributions[0].source_id if association.attributions else None
            )
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
    """Per-``context`` rank interleaving — activation strengths are ordinal
    within one call only, so ranks are the currency across ``contexts`` (the
    posture the server's own cross-``context`` passage merge takes)."""
    indexed = [
        (rank, index, document)
        for index, documents in enumerate(per_target)
        for rank, document in enumerate(documents)
    ]
    indexed.sort(key=lambda entry: (entry[0], entry[1]))
    return [document for _rank, _index, document in indexed]


def _hit_context(hit: PassageHit, fallback: str | None) -> str | None:
    """A cross-``context`` hit names its ``context_id``; a per-``context`` hit
    inherits the retriever's own target."""
    return getattr(hit, "context_id", None) or fallback


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
            "source": hit.source_id,
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
    """Reciprocal Rank Fusion across lanes; (``context``, source, paragraph) hits
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
        key = (context, hit.source_id, hit.paragraph)
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
