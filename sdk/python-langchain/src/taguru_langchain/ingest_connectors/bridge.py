"""The bridge from a connector's normalized document to
:class:`~taguru_langchain.ingest.TaguruIngester` (ADR 0007, issue #347) —
the "end to end" wiring the issue asks for: a
:class:`~taguru_langchain.ingest_connectors.document.ConnectorDocument`
goes in, an :class:`~taguru_langchain.ingest.IngestOutcome` comes out,
through the exact same retract-then-apply batch/import path any other
source uses.

Depends on ``taguru_langchain.ingest`` in one direction only —
``TaguruIngester`` itself knows nothing about connectors; a connector never
grows special-cased methods on the ingester it feeds.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

from taguru import LocatorSpec, SectionSpec

from ..ingest import IngestOutcome, TaguruIngester, _normalize_stop
from .document import ConnectorDocument


def _specs(document: ConnectorDocument) -> tuple[list[SectionSpec], list[LocatorSpec]]:
    sections: list[SectionSpec] = [
        {"paragraph": entry.paragraph, "section": entry.section} for entry in document.sections
    ]
    locators: list[LocatorSpec] = [
        {
            "paragraph": entry.paragraph,
            "locator": {"kind": entry.locator.kind, "value": entry.locator.value},
        }
        for entry in document.locators
    ]
    return sections, locators


def _diagnostics_only_outcome(document: ConnectorDocument) -> IngestOutcome:
    summary = "; ".join(f"{d.code}: {d.message}" for d in document.diagnostics)
    return IngestOutcome(
        source=document.source,
        ok=False,
        error=summary or "connector document has empty text and no diagnostic",
    )


def ingest_connector_document(
    ingester: TaguruIngester,
    document: ConnectorDocument,
    *,
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
) -> IngestOutcome:
    """Ingests one :class:`ConnectorDocument` through ``ingester``.

    A document whose ``text`` is empty (diagnostics-only — nothing usable
    was extracted, ADR 0007 §5/§10) never reaches the network: it is
    reported as a failed :class:`~taguru_langchain.ingest.IngestOutcome`
    carrying every diagnostic's ``code``/``message`` in ``error`` — the same
    "never silently empty" posture the normalized document contract itself
    enforces at construction time.
    """
    if not document.text:
        return _diagnostics_only_outcome(document)
    sections, locators = _specs(document)
    return ingester.ingest_text(
        document.text,
        source=document.source,
        sections=sections,
        locators=locators,
        dry_run=dry_run,
        should_stop=should_stop,
    )


async def aingest_connector_document(
    ingester: TaguruIngester,
    document: ConnectorDocument,
    *,
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
) -> IngestOutcome:
    """Async twin of :func:`ingest_connector_document`."""
    if not document.text:
        return _diagnostics_only_outcome(document)
    sections, locators = _specs(document)
    return await ingester.aingest_text(
        document.text,
        source=document.source,
        sections=sections,
        locators=locators,
        dry_run=dry_run,
        should_stop=should_stop,
    )


def ingest_connector_documents(
    ingester: TaguruIngester,
    documents: Sequence[ConnectorDocument],
    *,
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
) -> list[IngestOutcome]:
    """Ingests each document independently — one failure never stops the
    rest (set ``ingester.raise_on_error=True`` to fail fast), the same
    posture :meth:`~taguru_langchain.ingest.TaguruIngester.ingest_documents`
    already takes for LangChain ``Document``s. ``should_stop`` (issue #211)
    is checked between documents too, and once a document's own
    :func:`ingest_connector_document` reports ``interrupted`` the run stops
    there.

    A diagnostics-only document (empty ``text``) is never a
    ``raise_on_error`` trigger — it is expected connector output, not a
    programming error, so the run always continues past it to the next
    document regardless of ``raise_on_error``.
    """
    stop = _normalize_stop(should_stop)
    outcomes: list[IngestOutcome] = []
    for document in documents:
        if stop():
            break
        try:
            outcome = ingest_connector_document(
                ingester, document, dry_run=dry_run, should_stop=should_stop
            )
        except Exception as error:
            if ingester.raise_on_error:
                raise
            outcomes.append(IngestOutcome(source=document.source, ok=False, error=str(error)))
            continue
        outcomes.append(outcome)
        if outcome.interrupted:
            break
    return outcomes


async def aingest_connector_documents(
    ingester: TaguruIngester,
    documents: Sequence[ConnectorDocument],
    *,
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
) -> list[IngestOutcome]:
    """Async twin of :func:`ingest_connector_documents`."""
    stop = _normalize_stop(should_stop)
    outcomes: list[IngestOutcome] = []
    for document in documents:
        if stop():
            break
        try:
            outcome = await aingest_connector_document(
                ingester, document, dry_run=dry_run, should_stop=should_stop
            )
        except Exception as error:
            if ingester.raise_on_error:
                raise
            outcomes.append(IngestOutcome(source=document.source, ok=False, error=str(error)))
            continue
        outcomes.append(outcome)
        if outcome.interrupted:
            break
    return outcomes
