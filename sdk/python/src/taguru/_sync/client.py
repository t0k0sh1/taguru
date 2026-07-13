"""Sync client — GENERATED, do not edit.

This file is produced from ``taguru._async.client`` by
``scripts/generate_sync.py`` (unasync). Edit the async source and regenerate.
"""

from __future__ import annotations

import time
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from .._decode import decode
from .._errors import NotFoundError, TaguruError, TransportError
from .._models import (
    Activation,
    ActivationPage,
    AliasEntry,
    AliasPage,
    Association,
    BatchApplyResult,
    Citation,
    CompactOutcome,
    ConceptDescription,
    ContextMeta,
    ContextPage,
    CrossMatchPage,
    CrossPassageHit,
    DirectoryEntry,
    DriftAudit,
    ExplorePage,
    GroupEntry,
    GroupPage,
    ImportOutcome,
    LabelPage,
    MatchPage,
    PassageHit,
    PassageLookup,
    RefreshOutcome,
    RetractAssociationOutcome,
    RetractOutcome,
    RetrievalResult,
    SourcePage,
    StoredPassages,
    TieredResolution,
    VocabularyAudit,
)
from .._retry import (
    DEFAULT_RETRIES,
    RetryClass,
    backoff_delay,
    parse_retry_after,
    should_retry_status,
    should_retry_transport,
)
from .._shared import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECS,
    ENV_TOKEN,
    ENV_URL,
    MAX_CHUNK_BYTES,
    MAX_OPS_PER_REQUEST,
    chunk_associations,
    drop_none,
    dumps_compact,
    encode_name,
    normalize_import_outcomes,
    raise_for_response,
    unwrap_envelope,
)
from .._types import (
    AssocOp,
    CrossMatchCursor,
    ExploreCursor,
    MatchCursor,
    QuestionSpec,
    SectionSpec,
)

__all__ = ["Taguru", "Contexts", "Groups", "Context"]


class Taguru:
    """Client for one Taguru server.

    Args:
        base_url: Server base URL; defaults to ``$TAGURU_URL`` then
            ``http://127.0.0.1:8248``.
        api_key: Bearer token; defaults to ``$TAGURU_API_TOKEN`` (unset means
            the server runs unauthenticated — dev mode).
        timeout: Per-request budget in seconds. Matches the server's own
            default; raise both together when the server calls an embedding
            provider.
        retries: Additional attempts after the first, for retry-safe failures.
        headers: Extra headers sent on every request.
        http_client: Bring your own configured ``httpx`` client (proxies,
            mTLS, ...); its lifecycle stays yours.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECS,
        retries: int = DEFAULT_RETRIES,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get(ENV_URL) or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get(ENV_TOKEN)
        self._retries = retries
        self._headers: dict[str, str] = dict(headers or {})
        if self._api_key:
            self._headers["authorization"] = f"Bearer {self._api_key}"
        self._http = http_client if http_client is not None else httpx.Client(timeout=timeout)
        self._owns_http = http_client is None
        self.contexts = Contexts(self)
        self.groups = Groups(self)

    # -- transport ---------------------------------------------------------

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        content: bytes | None = None,
        content_type: str | None = None,
        retry: RetryClass = RetryClass.SAFE,
        retries: int | None = None,
    ) -> httpx.Response:
        url = self._base_url + path
        headers = dict(self._headers)
        body: bytes | None = content
        if json_body is not None:
            body = dumps_compact(json_body)
            headers["content-type"] = "application/json"
        elif content is not None and content_type is not None:
            headers["content-type"] = content_type
        max_attempts = (self._retries if retries is None else retries) + 1
        attempt = 0
        while True:
            try:
                response = self._http.request(
                    method,
                    url,
                    params=dict(params) if params else None,
                    content=body,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                # A pre-connect failure never reached the server; anything
                # after that is ambiguous (the request may have executed).
                ambiguous = not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
                attempt += 1
                if attempt < max_attempts and should_retry_transport(ambiguous, retry):
                    time.sleep(backoff_delay(attempt - 1))
                    continue
                raise TransportError(str(exc) or type(exc).__name__) from exc
            if response.status_code >= 400:
                attempt += 1
                if attempt < max_attempts and should_retry_status(response.status_code, retry):
                    delay = parse_retry_after(response.headers.get("retry-after"))
                    time.sleep(delay if delay is not None else backoff_delay(attempt - 1))
                    continue
                raise_for_response(response)
            return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        retry: RetryClass = RetryClass.SAFE,
    ) -> Any:
        response = self._send(method, path, params=params, json_body=json_body, retry=retry)
        return unwrap_envelope(response)

    # -- server-level operations -------------------------------------------

    def health(self) -> None:
        """Readiness probe: raises ``ServiceUnavailableError`` while degraded."""
        self._send("GET", "/health", retries=0)

    def live(self) -> None:
        """Liveness probe: succeeds as long as the process answers at all."""
        self._send("GET", "/live", retries=0)

    def metrics(self) -> str:
        """Prometheus exposition text (not the JSON envelope)."""
        response = self._send("GET", "/metrics")
        return response.text

    def protocol(self) -> str:
        """The client protocol document (markdown) this server ships."""
        response = self._send("GET", "/protocol")
        return response.text

    def flush(self) -> list[str]:
        """Persist every dirty context now; returns their names (admin role)."""
        result = self._request_json("POST", "/flush")
        return [str(name) for name in result]

    def import_batches(self, data: str | bytes) -> list[ImportOutcome]:
        """Apply an NDJSON batch stream (the format ``export`` produces).

        Each batch is one source's retract-then-apply, so re-importing is
        idempotent. The response is normalized to a list even for a single
        batch.
        """
        content = data.encode("utf-8") if isinstance(data, str) else data
        response = self._send(
            "POST", "/import", content=content, content_type="application/x-ndjson"
        )
        return normalize_import_outcomes(unwrap_envelope(response))

    def import_file(self, path: str | Path) -> list[ImportOutcome]:
        """Apply an NDJSON batch file (see ``import_batches``)."""
        return self.import_batches(Path(path).read_bytes())

    def wait_until_ready(self, *, timeout: float = 30.0, interval: float = 0.5) -> None:
        """Poll ``live`` then ``health`` until both pass or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        last_error: TaguruError | None = None
        while True:
            try:
                self.live()
                self.health()
                return
            except TaguruError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                if last_error is not None:
                    raise last_error
                raise TaguruError(f"server not ready after {timeout} seconds")
            time.sleep(interval)

    def context(self, name: str) -> Context:
        """A handle bound to one context (no network call)."""
        return Context(self, name)

    # -- cross-context search ------------------------------------------------

    def recall(
        self,
        cue: str,
        *,
        contexts: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
        limit: int | None = None,
        after: CrossMatchCursor | None = None,
    ) -> CrossMatchPage:
        """Recall across several contexts at once, every match tagged.

        ``contexts`` takes full names; each ``groups`` entry searches every
        context the group reaches (nested children included), overlaps
        deduped. At least one of the two must name something. Weights share
        one scale, so past the limit the strongest |weight| survives exactly
        as within one context. ``after`` resumes past the previous page's
        last match; ``total`` stays constant across pages.
        """
        body = drop_none(
            {
                "contexts": list(contexts) if contexts is not None else None,
                "groups": list(groups) if groups is not None else None,
                "cue": cue,
                "limit": limit,
                "after": after,
            }
        )
        result = self._request_json("POST", "/recall", json_body=body)
        return decode(CrossMatchPage, result)  # type: ignore[no-any-return]

    def query(
        self,
        *,
        contexts: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
        subject: str | Sequence[str] | None = None,
        label: str | Sequence[str] | None = None,
        object: str | Sequence[str] | None = None,
        limit: int | None = None,
        after: CrossMatchCursor | None = None,
    ) -> CrossMatchPage:
        """Exact-position query across several contexts at once, matches
        tagged; the same target contract as :meth:`recall`."""
        body = drop_none(
            {
                "contexts": list(contexts) if contexts is not None else None,
                "groups": list(groups) if groups is not None else None,
                "subject": subject,
                "label": label,
                "object": object,
                "limit": limit,
                "after": after,
            }
        )
        result = self._request_json("POST", "/query", json_body=body)
        return decode(CrossMatchPage, result)  # type: ignore[no-any-return]

    def search_passages(
        self,
        query: str,
        *,
        contexts: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[CrossPassageHit]:
        """Paragraph search across several contexts at once, hits tagged.

        Passage scores do NOT share a scale across contexts (BM25 statistics
        are corpus-local), so the merged order is rank interleaving — every
        context's best hit first; ``score`` compares within one context only.
        """
        body = drop_none(
            {
                "contexts": list(contexts) if contexts is not None else None,
                "groups": list(groups) if groups is not None else None,
                "query": query,
                "limit": limit,
            }
        )
        result = self._request_json("POST", "/sources/search", json_body=body)
        return decode(list[CrossPassageHit], result)  # type: ignore[no-any-return]

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Taguru:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class Contexts:
    """The context directory: collection-level CRUD."""

    def __init__(self, client: Taguru) -> None:
        self._client = client

    def list(self, *, limit: int | None = None, after: str | None = None) -> ContextPage:
        """One directory page (keyset cursor: ``after`` = last name shown)."""
        result = self._client._request_json(
            "GET", "/contexts", params=drop_none({"limit": limit, "after": after})
        )
        return decode(ContextPage, result)  # type: ignore[no-any-return]

    def iter(self, *, limit: int | None = None) -> Iterator[DirectoryEntry]:
        """Walk every directory page transparently."""
        after: str | None = None
        while True:
            page = self.list(limit=limit, after=after)
            if not page.contexts:
                return
            for entry in page.contexts:
                yield entry
            if limit is not None and len(page.contexts) < limit:
                return
            after = page.contexts[-1].name

    def get(self, name: str) -> DirectoryEntry:
        result = self._client._request_json("GET", f"/contexts/{encode_name(name)}")
        return decode(DirectoryEntry, result)  # type: ignore[no-any-return]

    def exists(self, name: str) -> bool:
        try:
            self.get(name)
        except NotFoundError:
            return False
        return True

    def create(
        self,
        name: str,
        *,
        description: str = "",
        pinned: bool = False,
        dice_floor: float | None = None,
        semantic_floor: float | None = None,
    ) -> bool:
        """Create a context (409 ``ConflictError`` if it already exists)."""
        body = drop_none(
            {
                "description": description,
                "pinned": pinned,
                "dice_floor": dice_floor,
                "semantic_floor": semantic_floor,
            }
        )
        result = self._client._request_json(
            "PUT",
            f"/contexts/{encode_name(name)}",
            json_body=body,
            retry=RetryClass.UNSAFE_ON_AMBIGUOUS,
        )
        return bool(result)

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        pinned: bool | None = None,
        dice_floor: float | None = None,
        semantic_floor: float | None = None,
    ) -> ContextMeta:
        """Update metadata; ``None`` leaves a field unchanged."""
        body = drop_none(
            {
                "description": description,
                "pinned": pinned,
                "dice_floor": dice_floor,
                "semantic_floor": semantic_floor,
            }
        )
        result = self._client._request_json(
            "PATCH", f"/contexts/{encode_name(name)}", json_body=body
        )
        return decode(ContextMeta, result)  # type: ignore[no-any-return]

    def delete(self, name: str) -> bool:
        """Delete a context, files included (admin role)."""
        result = self._client._request_json("DELETE", f"/contexts/{encode_name(name)}")
        return bool(result)


class Groups:
    """The group directory: flat context bundles (many-to-many) that may nest
    child groups — a shallow DAG, at most 3 storeys, never cyclic — as the
    addressing unit cross-context search builds on."""

    def __init__(self, client: Taguru) -> None:
        self._client = client

    def list(self, *, limit: int | None = None, after: str | None = None) -> GroupPage:
        """One directory page (keyset cursor: ``after`` = last name shown)."""
        result = self._client._request_json(
            "GET", "/groups", params=drop_none({"limit": limit, "after": after})
        )
        return decode(GroupPage, result)  # type: ignore[no-any-return]

    def iter(self, *, limit: int | None = None) -> Iterator[GroupEntry]:
        """Walk every directory page transparently."""
        after: str | None = None
        while True:
            page = self.list(limit=limit, after=after)
            if not page.groups:
                return
            for entry in page.groups:
                yield entry
            if limit is not None and len(page.groups) < limit:
                return
            after = page.groups[-1].name

    def get(self, name: str) -> GroupEntry:
        result = self._client._request_json("GET", f"/groups/{encode_name(name)}")
        return decode(GroupEntry, result)  # type: ignore[no-any-return]

    def exists(self, name: str) -> bool:
        try:
            self.get(name)
        except NotFoundError:
            return False
        return True

    def create(
        self,
        name: str,
        *,
        description: str = "",
        contexts: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
    ) -> bool:
        """Create a group (409 ``ConflictError`` if it already exists).

        Every listed member — context or child group — must already exist;
        contexts and groups are separate namespaces.
        """
        body = drop_none(
            {
                "description": description,
                "contexts": list(contexts) if contexts is not None else None,
                "groups": list(groups) if groups is not None else None,
            }
        )
        result = self._client._request_json(
            "PUT",
            f"/groups/{encode_name(name)}",
            json_body=body,
            retry=RetryClass.UNSAFE_ON_AMBIGUOUS,
        )
        return bool(result)

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        add_contexts: Sequence[str] | None = None,
        remove_contexts: Sequence[str] | None = None,
        add_groups: Sequence[str] | None = None,
        remove_groups: Sequence[str] | None = None,
    ) -> GroupEntry:
        """Delta membership update (removals first); returns the updated row.

        Removing a non-member is an idempotent no-op; only additions demand
        the member exists. The result holds at most 1,000 member contexts and
        1,000 child groups — past that, split into nested child groups.
        """
        body = drop_none(
            {
                "description": description,
                "add_contexts": list(add_contexts) if add_contexts is not None else None,
                "remove_contexts": list(remove_contexts) if remove_contexts is not None else None,
                "add_groups": list(add_groups) if add_groups is not None else None,
                "remove_groups": list(remove_groups) if remove_groups is not None else None,
            }
        )
        result = self._client._request_json("PATCH", f"/groups/{encode_name(name)}", json_body=body)
        return decode(GroupEntry, result)  # type: ignore[no-any-return]

    def delete(self, name: str) -> bool:
        """Delete the bundling only — member contexts and child groups stay."""
        result = self._client._request_json("DELETE", f"/groups/{encode_name(name)}")
        return bool(result)

    def export(self, name: str) -> str:
        """The group as one import-stream record (a ``taguru_group`` JSON
        line); ``import_batches`` restores it as a whole-record replace."""
        response = self._client._send("GET", f"/groups/{encode_name(name)}/export")
        return response.text


class Context:
    """Operations bound to one context, named after the server's own vocabulary.

    Method names mirror ``GET /protocol`` and the MCP tool names, so knowledge
    of one surface transfers to the others.
    """

    def __init__(self, client: Taguru, name: str) -> None:
        self._client = client
        self.name = name
        self._path = f"/contexts/{encode_name(name)}"

    def _post(self, suffix: str, json_body: Any = None, retry: RetryClass = RetryClass.SAFE) -> Any:
        return self._client._request_json(
            "POST", self._path + suffix, json_body=json_body, retry=retry
        )

    # -- entry resolution ---------------------------------------------------

    def resolve(
        self,
        cue: str,
        *,
        dice_floor: float | None = None,
        semantic_floor: float | None = None,
        limit: int | None = None,
    ) -> list[TieredResolution]:
        """Concept candidates for a cue. Read ``gloss`` before adopting a
        containment/fuzzy hit — never adopt a lookalike on score alone."""
        body = drop_none(
            {
                "cue": cue,
                "dice_floor": dice_floor,
                "semantic_floor": semantic_floor,
                "limit": limit,
            }
        )
        result = self._post("/resolve", body)
        return decode(list[TieredResolution], result)  # type: ignore[no-any-return]

    def resolve_label(
        self,
        cue: str,
        *,
        dice_floor: float | None = None,
        semantic_floor: float | None = None,
        limit: int | None = None,
    ) -> list[TieredResolution]:
        """Relation-label candidates for a cue."""
        body = drop_none(
            {
                "cue": cue,
                "dice_floor": dice_floor,
                "semantic_floor": semantic_floor,
                "limit": limit,
            }
        )
        result = self._post("/resolve_label", body)
        return decode(list[TieredResolution], result)  # type: ignore[no-any-return]

    # -- graph reads ---------------------------------------------------------

    def recall(
        self, cue: str, *, limit: int | None = None, after: MatchCursor | None = None
    ) -> MatchPage:
        """Associations whose subject/object entry-matches the cue.

        ``after`` resumes past the previous page's last match; ``total``
        stays constant across pages.
        """
        result = self._post("/recall", drop_none({"cue": cue, "limit": limit, "after": after}))
        return decode(MatchPage, result)  # type: ignore[no-any-return]

    def query(
        self,
        *,
        subject: str | Sequence[str] | None = None,
        label: str | Sequence[str] | None = None,
        object: str | Sequence[str] | None = None,
        limit: int | None = None,
        after: MatchCursor | None = None,
    ) -> MatchPage:
        """Exact-position query; each position takes one name or an OR-set."""
        body = drop_none(
            {
                "subject": subject,
                "label": label,
                "object": object,
                "limit": limit,
                "after": after,
            }
        )
        result = self._post("/query", body)
        return decode(MatchPage, result)  # type: ignore[no-any-return]

    def describe(self, concept: str) -> ConceptDescription | None:
        """Label outline (counts per role); ``None`` for an unknown concept."""
        result = self._post("/describe", {"concept": concept})
        return decode(ConceptDescription | None, result)  # type: ignore[no-any-return]

    def explore(
        self,
        origins: str | Sequence[str],
        *,
        max_depth: int | None = None,
        limit: int | None = None,
        after: ExploreCursor | None = None,
    ) -> ExplorePage:
        """Exhaustive hop-annotated walk (truncation keeps the nearest).

        ``after`` resumes past the previous page's last recollection;
        ``total`` stays constant across pages.
        """
        body = drop_none(
            {
                "origins": [origins] if isinstance(origins, str) else list(origins),
                "max_depth": max_depth,
                "limit": limit,
                "after": after,
            }
        )
        result = self._post("/explore", body)
        return decode(ExplorePage, result)  # type: ignore[no-any-return]

    def activate(
        self,
        origins: str | Sequence[str],
        *,
        decay: float | None = None,
        limit: int | None = None,
    ) -> ActivationPage:
        """Spreading activation from origins, strongest first."""
        body = drop_none(
            {
                "origins": [origins] if isinstance(origins, str) else list(origins),
                "decay": decay,
                "limit": limit,
            }
        )
        result = self._post("/activate", body)
        return decode(ActivationPage, result)  # type: ignore[no-any-return]

    def unreachable_from(
        self,
        origins: str | Sequence[str],
        *,
        limit: int | None = None,
        after: MatchCursor | None = None,
    ) -> MatchPage:
        """Coverage audit: associations not reachable from the origins.

        ``after`` resumes past the previous page's last match; ``total``
        stays constant across pages.
        """
        body = drop_none(
            {
                "origins": [origins] if isinstance(origins, str) else list(origins),
                "limit": limit,
                "after": after,
            }
        )
        result = self._post("/unreachable_from", body)
        return decode(MatchPage, result)  # type: ignore[no-any-return]

    def list_labels(self, *, limit: int | None = None, after: str | None = None) -> LabelPage:
        """One page of the relation vocabulary (canonical labels only)."""
        result = self._client._request_json(
            "GET", self._path + "/labels", params=drop_none({"limit": limit, "after": after})
        )
        return decode(LabelPage, result)  # type: ignore[no-any-return]

    def iter_labels(self, *, limit: int | None = None) -> Iterator[str]:
        after: str | None = None
        while True:
            page = self.list_labels(limit=limit, after=after)
            if not page.labels:
                return
            for label in page.labels:
                yield label
            if limit is not None and len(page.labels) < limit:
                return
            after = page.labels[-1]

    # -- graph writes ---------------------------------------------------------

    def add_associations(self, associations: Sequence[AssocOp]) -> int:
        """Assert a batch of associations; returns the applied count.

        Weight ACCUMULATES on re-assertion, so this call is never blindly
        retried after an ambiguous transport failure. Server cap: 10,000 per
        request (use ``add_associations_batched`` to auto-chunk).
        """
        result = self._post(
            "/associations", list(associations), retry=RetryClass.UNSAFE_ON_AMBIGUOUS
        )
        return int(result)

    def add_associations_batched(
        self,
        associations: Sequence[AssocOp],
        *,
        chunk_size: int = MAX_OPS_PER_REQUEST,
        max_chunk_bytes: int = MAX_CHUNK_BYTES,
    ) -> BatchApplyResult:
        """Chunked ``add_associations`` for arbitrarily large batches.

        Chunks are independent requests: a failure mid-way leaves earlier
        chunks applied (that is why this is a separate, opt-in method).
        """
        applied = 0
        chunks = 0
        for chunk in chunk_associations(list(associations), chunk_size, max_chunk_bytes):
            applied += self.add_associations(chunk)
            chunks += 1
        return BatchApplyResult(applied=applied, chunks=chunks)

    def retract_association(
        self, subject: str, label: str, object: str
    ) -> RetractAssociationOutcome:
        """Withdraw one (subject, label, object) association outright.

        Every source's contribution to that one edge goes — where
        ``retract_source`` withdraws a whole document's. Names resolve
        through aliases; ``retracted=False`` means the triple named no
        live edge and nothing changed. The surgical correction for a
        fact that should never have been asserted; a fact that is merely
        CONTESTED wants a negative-weight assertion instead.
        """
        body = {"subject": subject, "label": label, "object": object}
        result = self._post("/associations/retract", body)
        return decode(RetractAssociationOutcome, result)  # type: ignore[no-any-return]

    # -- passages / sources ----------------------------------------------------

    def store_passages(
        self,
        passages: Mapping[str, str],
        *,
        questions: Mapping[str, Sequence[QuestionSpec]] | None = None,
        sections: Mapping[str, Sequence[SectionSpec]] | None = None,
    ) -> StoredPassages:
        """Register source-id → full-text passages (replaces per source).

        Store the document as-is: the server splits paragraphs on blank
        lines. ``questions``/``sections`` attach per-paragraph doc2query
        questions and section labels.
        """
        body: dict[str, Any] = {"passages": dict(passages)}
        if questions is not None:
            body["questions"] = {key: list(value) for key, value in questions.items()}
        if sections is not None:
            body["sections"] = {key: list(value) for key, value in sections.items()}
        result = self._post("/sources", body)
        return decode(StoredPassages, result)  # type: ignore[no-any-return]

    def lookup_passages(self, sources: Sequence[str]) -> PassageLookup:
        """Fetch whole passages by source id."""
        result = self._post("/sources/lookup", {"sources": list(sources)})
        return decode(PassageLookup, result)  # type: ignore[no-any-return]

    def search_passages(self, query: str, *, limit: int | None = None) -> list[PassageHit]:
        """Paragraph search (BM25 fused with embeddings where configured).

        Phrase the query as an answer, not a question — a plausible
        declarative sentence lands nearer the text you hope to find.
        """
        result = self._post("/sources/search", drop_none({"query": query, "limit": limit}))
        return decode(list[PassageHit], result)  # type: ignore[no-any-return]

    def retract_source(self, source: str) -> RetractOutcome:
        """Withdraw one source's contributions (diff sync before re-ingest)."""
        result = self._post("/sources/retract", {"source": source})
        return decode(RetractOutcome, result)  # type: ignore[no-any-return]

    def list_sources(self, *, limit: int | None = None, after: str | None = None) -> SourcePage:
        result = self._client._request_json(
            "GET", self._path + "/sources", params=drop_none({"limit": limit, "after": after})
        )
        return decode(SourcePage, result)  # type: ignore[no-any-return]

    def iter_sources(self, *, limit: int | None = None) -> Iterator[str]:
        after: str | None = None
        while True:
            page = self.list_sources(limit=limit, after=after)
            if not page.sources:
                return
            for source in page.sources:
                yield source
            if limit is not None and len(page.sources) < limit:
                return
            after = page.sources[-1]

    def cite_passage(self, source: str, paragraph: int) -> Citation:
        """One verbatim paragraph by source and paragraph locator."""
        result = self._post("/citations", {"source": source, "paragraph": paragraph})
        return decode(Citation, result)  # type: ignore[no-any-return]

    # -- aliases -----------------------------------------------------------------

    def get_aliases(self, *, limit: int | None = None, after: str | None = None) -> AliasPage:
        """One alias page; the cursor spans both namespaces (concepts first),
        so ``after`` takes ``"concept:<alias>"`` or ``"label:<alias>"``."""
        result = self._client._request_json(
            "GET", self._path + "/aliases", params=drop_none({"limit": limit, "after": after})
        )
        return decode(AliasPage, result)  # type: ignore[no-any-return]

    def iter_aliases(self, *, limit: int | None = None) -> Iterator[AliasEntry]:
        """Walk both alias namespaces as a flat stream of entries."""
        after: str | None = None
        while True:
            page = self.get_aliases(limit=limit, after=after)
            count = len(page.concepts) + len(page.labels)
            if count == 0:
                return
            last = after
            for alias, canonical in page.concepts.items():
                yield AliasEntry(namespace="concept", alias=alias, canonical=canonical)
                last = f"concept:{alias}"
            for alias, canonical in page.labels.items():
                yield AliasEntry(namespace="label", alias=alias, canonical=canonical)
                last = f"label:{alias}"
            if limit is not None and count < limit:
                return
            after = last

    def add_aliases(
        self,
        *,
        concepts: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> int:
        """Register alias → canonical spellings; returns the applied count.

        Aliases are entry-only: results always carry the canonical spelling.
        Re-registering an identical pair succeeds as a no-op (verified against
        the server), so this call is retry-safe.
        """
        body = {"concepts": dict(concepts or {}), "labels": dict(labels or {})}
        result = self._post("/aliases", body)
        return int(result)

    def remove_aliases(
        self,
        *,
        concepts: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
    ) -> int:
        """Withdraw alias spellings (canonical names are refused)."""
        body = {"concepts": list(concepts or []), "labels": list(labels or [])}
        result = self._client._request_json("DELETE", self._path + "/aliases", json_body=body)
        return int(result)

    # -- maintenance ---------------------------------------------------------------

    def audit_vocabulary(
        self, *, dice_floor: float | None = None, cosine_floor: float | None = None
    ) -> VocabularyAudit:
        """Spelling/synonym fork candidates — candidates, not verdicts."""
        body = drop_none({"dice_floor": dice_floor, "cosine_floor": cosine_floor})
        result = self._post("/vocabulary/audit", body)
        return decode(VocabularyAudit, result)  # type: ignore[no-any-return]

    def audit_drift(
        self,
        *,
        unsourced_floor: float | None = None,
        limit: int | None = None,
        after: MatchCursor | None = None,
        include_twins: bool = False,
        dice_floor: float | None = None,
        cosine_floor: float | None = None,
    ) -> DriftAudit:
        """Graph-vs-archive drift audit: unsourced weight (worst-first,
        paginated) and dead-canonical aliases, always; vocabulary fork
        candidates too when ``include_twins`` is set.

        ``after`` resumes past the previous page's last unsourced match;
        ``total`` stays constant across pages.
        """
        body = drop_none(
            {
                "unsourced_floor": unsourced_floor,
                "limit": limit,
                "after": after,
                "include_twins": include_twins,
                "dice_floor": dice_floor,
                "cosine_floor": cosine_floor,
            }
        )
        result = self._post("/drift/audit", body)
        return decode(DriftAudit, result)  # type: ignore[no-any-return]

    def refresh_embeddings(self) -> RefreshOutcome:
        """Re-embed new/changed glosses (diff-only, idempotent).

        Raises ``EmbeddingUnavailableError`` (501) when the server has no
        provider configured.
        """
        result = self._post("/embeddings/refresh")
        return decode(RefreshOutcome, result)  # type: ignore[no-any-return]

    def compact(self) -> CompactOutcome:
        """Rebuild the image without dead records (admin role)."""
        result = self._post("/compact")
        return decode(CompactOutcome, result)  # type: ignore[no-any-return]

    # -- export ------------------------------------------------------------------------

    def export(self) -> str:
        """The context as an import batch stream (NDJSON text)."""
        response = self._client._send("GET", self._path + "/export")
        return response.text

    def export_stream(self) -> Iterator[bytes]:
        """Stream the export body without buffering it whole (no retry)."""
        url = self._client._base_url + self._path + "/export"
        headers = dict(self._client._headers)
        with self._client._http.stream("GET", url, headers=headers) as response:
            if response.status_code >= 400:
                response.read()
                raise_for_response(response)
            for chunk in response.iter_bytes():
                yield chunk

    def export_to_file(self, path: str | Path) -> None:
        """Stream the export straight to a file."""
        target = Path(path)
        with target.open("wb") as handle:
            for chunk in self.export_stream():
                handle.write(chunk)

    # -- high-level retrieval loop -------------------------------------------------------

    def retrieve(
        self,
        origins: str | Sequence[str],
        *,
        labels: str | Sequence[str] | None = None,
        dice_floor: float | None = None,
        semantic_floor: float | None = None,
        resolve_limit: int | None = None,
        auto_pick: bool = True,
        activate_decay: float | None = None,
        activate_limit: int | None = None,
        describe_first: bool = True,
        fetch_citations: bool = True,
        text_fallback_query: str | None = None,
        text_fallback_only_if_empty: bool = True,
        search_limit: int | None = None,
    ) -> RetrievalResult:
        """The documented retrieval loop as one call.

        resolve each cue → (describe) → activate (and ``query`` when
        ``labels`` pins the facets) → batch citations for every located
        attribution → optional text-lane fallback.

        The cues must already be extracted entity names — decomposing a
        question, judging lookalikes, and phrasing a declarative
        ``text_fallback_query`` are the calling LLM's job. Every resolve
        candidate (gloss included) is returned so an auto-picked anchor is
        never hidden.
        """
        cues = [origins] if isinstance(origins, str) else list(origins)
        resolved: dict[str, list[TieredResolution]] = {}
        anchors: list[str] = []
        for cue in cues:
            candidates = self.resolve(
                cue, dice_floor=dice_floor, semantic_floor=semantic_floor, limit=resolve_limit
            )
            resolved[cue] = candidates
            picked = (
                candidates[0].name if (auto_pick and candidates) else (None if auto_pick else cue)
            )
            if picked is not None and picked not in anchors:
                anchors.append(picked)

        outline: dict[str, ConceptDescription | None] = {}
        if describe_first:
            for anchor in anchors:
                outline[anchor] = self.describe(anchor)

        activations: list[Activation] = []
        associations: list[Association] = []
        seen_triples: set[tuple[str, str, str]] = set()
        if anchors:
            if labels is not None:
                matched = self.query(subject=anchors, label=labels)
                for match in matched.matches:
                    triple = (match.subject, match.label, match.object)
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        associations.append(match)
            page = self.activate(anchors, decay=activate_decay, limit=activate_limit)
            activations = page.matches
            for activation in activations:
                triple = (
                    activation.association.subject,
                    activation.association.label,
                    activation.association.object,
                )
                if triple not in seen_triples:
                    seen_triples.add(triple)
                    associations.append(activation.association)

        citations: dict[tuple[str, int], Citation] = {}
        if fetch_citations:
            wanted: list[tuple[str, int]] = []
            for association in associations:
                for attribution in association.attributions:
                    if attribution.paragraph is None:
                        continue
                    key = (attribution.source, attribution.paragraph)
                    if key not in citations and key not in wanted:
                        wanted.append(key)
            for source, paragraph in wanted:
                try:
                    citations[(source, paragraph)] = self.cite_passage(source, paragraph)
                except NotFoundError:
                    # The locator points at a passage that was never stored
                    # (or was retracted) — the graph fact itself still stands.
                    continue

        passage_hits: list[PassageHit] = []
        if text_fallback_query is not None and (
            not text_fallback_only_if_empty or not associations
        ):
            passage_hits = self.search_passages(text_fallback_query, limit=search_limit)

        return RetrievalResult(
            resolved=resolved,
            outline=outline,
            associations=associations,
            activations=activations,
            citations=citations,
            passage_hits=passage_hits,
        )
