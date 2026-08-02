"""The cross-connector local-file/URL driver (ADR 0007 §11, issue #353):
the one entry point every non-object-storage connector (``.md``/``.txt``,
PDF, HTML, DOCX, PPTX) syncs through, instead of a caller hand-rolling its
own loop over :meth:`~taguru_langchain.ingest_connectors.protocol.
Connector.read` calls with no shared event log, counts, or ``dry_run``
posture. :func:`~taguru_langchain.ingest_connectors.s3.sync_object_storage`
is this module's object-storage twin — the two share
:mod:`~taguru_langchain.ingest_connectors.observability` but not this
module, since object-storage listing (bucket/prefix enumeration, its own
fingerprint tiers) has no local-file/URL analogue.

Two ideas this module exists to get right, both real bugs found while
writing it rather than hypotheticals:

- **Dispatch must classify a reference's KIND before asking any connector
  whether it "supports" the reference.** Every non-HTML connector's own
  ``supports()`` looks at the extension alone
  (:meth:`~taguru_langchain.ingest_connectors.text.TextFileConnector.
  supports` is exactly ``Path(reference).suffix.lower() in
  _SUPPORTED_SUFFIXES``) — so ``supports("https://example.com/a.md")`` is
  ``True``, and hands a URL to ``open()`` as though it were a local path.
  :func:`plan_references` classifies ``path`` vs. ``url`` first (mirroring
  exactly :meth:`~taguru_langchain.ingest_connectors.html.HtmlConnector.
  read`'s own ``_is_windows_drive_path`` → scheme check) and only probes a
  URL-kind reference against connectors that are scheme-aware in the
  first place (see :func:`_connector_for_url`).
- **A duplicate reference must never be tallied under the source id it
  collides with.** The obvious approach — record the duplicate's
  ``skipped``/``duplicate_source`` event under the SAME source id the
  earlier reference already claimed — corrupts
  :class:`~taguru_langchain.ingest_connectors.observability.RunReport`'s
  last-phase-only tally: because :func:`plan_references` always resolves
  duplicates in input order, the earlier (claiming) reference is always
  processed first, so by the time the duplicate is reached the claimed
  source may already have reached ``imported`` — recording ``skipped``
  under that same key would silently turn a successful import back into a
  failure in the summary.
  :meth:`~taguru_langchain.ingest_connectors.observability.RunRecorder.
  duplicate` exists specifically to make this impossible.
"""

from __future__ import annotations

import os
import stat as stat_module
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TextIO
from urllib.parse import urlsplit

from ..checkpoints import CheckpointStore
from ..ingest import TaguruIngester, _normalize_stop
from .bridge import ingest_connector_document
from .checkpoint import ConnectorCheckpoint, FileProbe, FileProbeCheckpoint
from .document import ConnectorDocument, Diagnostic, DiagnosticCode
from .docx import DocxConnector
from .html import HtmlConnector, _is_windows_drive_path, _strip_fragment
from .observability import RunRecorder, RunReport
from .pdf import PdfConnector
from .pptx import PptxConnector
from .protocol import Connector
from .sources import SourceIdRegistry, canonicalize_url, check_source_id, file_source_id
from .text import TextFileConnector

PARSER_NAME: Final = "taguru-reference-sync"
"""This driver's own identity in :attr:`~taguru_langchain.
ingest_connectors.observability.RunReport.connector` — it is not itself a
format connector (the delegate's own ``fingerprint_inputs.parser`` is left
untouched, exactly as :mod:`~taguru_langchain.ingest_connectors.s3` leaves a
delegate's fingerprint untouched under its own S3 identity re-stamping)."""

ReferenceKind = Literal["path", "url"]
"""What kind of identity a reference resolves to (ADR 0007 §11's per-kind
``dry_run`` table is keyed on exactly this distinction) — a reference whose
scheme names an object-storage backend (``s3://``, ``gs://``, ``az://``,
...) is classified ``"url"`` for bookkeeping purposes but always carries an
``unsupported_format`` diagnostic pointing at
:func:`~taguru_langchain.ingest_connectors.s3.sync_object_storage`
instead: this driver has no listing/fingerprint-tier story of its own for
those schemes."""


def default_connectors() -> tuple[Connector, ...]:
    """The installed reference connectors, in ADR 0007's own PDF/HTML/DOCX/
    PPTX order plus the original ``.md``/``.txt`` reference — an optional
    dependency's connector (``pdf``, ``docx``, ``pptx``) is simply absent
    when its extra was never installed, so a reference in that format
    legitimately reports ``unsupported_format`` rather than raising at
    construction. Mirrors
    :mod:`~taguru_langchain.ingest_connectors.s3`'s own (private)
    ``_default_connectors`` exactly; that module re-exports this one."""
    connectors: list[Connector] = [TextFileConnector(), HtmlConnector()]
    try:
        connectors.append(PdfConnector())
    except ImportError:
        pass
    try:
        connectors.append(DocxConnector())
    except ImportError:
        pass
    try:
        connectors.append(PptxConnector())
    except ImportError:
        pass
    return tuple(connectors)


def _probe_reference(scheme: str) -> str:
    # No installed connector's `supports()` inspects anything about a
    # reference beyond its scheme (for a URL) or extension (for a path) —
    # a fabricated, suffix-less URL is therefore enough to ask "are you
    # scheme-aware at all," the same trick
    # `ingest_connectors.s3._connector_for_suffix` already plays in the
    # other direction with a fabricated `"object<suffix>"` name.
    return f"{scheme}://taguru.invalid/probe"


def _connector_for_path(connectors: Sequence[Connector], reference: str) -> Connector | None:
    for connector in connectors:
        if connector.supports(reference):
            return connector
    return None


def _connector_for_url(
    connectors: Sequence[Connector], reference: str, scheme: str
) -> Connector | None:
    probe = _probe_reference(scheme)
    for connector in connectors:
        # Both checks matter: `supports(probe)` rejects every connector
        # that (like every non-HTML connector today) only ever looks at a
        # reference's extension, since the probe carries none; `supports
        # (reference)` then leaves room for a future scheme-aware
        # connector that only claims a URL scheme for certain paths.
        if connector.supports(probe) and connector.supports(reference):
            return connector
    return None


@dataclass(slots=True, frozen=True, kw_only=True)
class ReferencePlan:
    """One reference's dispatch decision, made before any fetch — the
    "enumerate all discovered work before the first fetch" half of ADR 0007
    §11, materialized as data so :func:`sync_references` can report every
    ``discovered`` event up front regardless of how far the run gets.

    ``diagnostic`` set means this reference will never reach ``read()`` —
    an unreadable/oversized/duplicate/unrecognized reference, reported
    ``skipped`` (or, for a duplicate, not tallied at all — see
    :meth:`~taguru_langchain.ingest_connectors.observability.RunRecorder.
    duplicate`) without ever calling a connector. ``connector`` is
    ``None`` exactly when ``diagnostic`` is set.
    """

    reference: str
    source: str
    kind: ReferenceKind
    source_bytes: int
    connector: Connector | None
    diagnostic: Diagnostic | None


def _duplicate_diagnostic(source: str) -> Diagnostic:
    return Diagnostic(
        code="duplicate_source",
        message="source id already claimed earlier in this run",
        source=source,
    )


def _unsupported(source: str, message: str) -> Diagnostic:
    code: DiagnosticCode = "unsupported_format"
    return Diagnostic(code=code, message=message, source=source)


def _plan_path(reference: str, source: str, *, connectors: Sequence[Connector]) -> ReferencePlan:
    diagnostic = check_source_id(source)
    if diagnostic is not None:
        return ReferencePlan(
            reference=reference,
            source=source,
            kind="path",
            source_bytes=0,
            connector=None,
            diagnostic=diagnostic,
        )
    try:
        stat_result = os.stat(reference)
    except OSError as error:
        return ReferencePlan(
            reference=reference,
            source=source,
            kind="path",
            source_bytes=0,
            connector=None,
            diagnostic=Diagnostic(code="unreadable", message=str(error), source=source),
        )
    if not stat_module.S_ISREG(stat_result.st_mode):
        return ReferencePlan(
            reference=reference,
            source=source,
            kind="path",
            source_bytes=0,
            connector=None,
            diagnostic=Diagnostic(code="unreadable", message="not a regular file", source=source),
        )
    connector = _connector_for_path(connectors, reference)
    if connector is None:
        suffix = Path(reference).suffix or "(none)"
        return ReferencePlan(
            reference=reference,
            source=source,
            kind="path",
            source_bytes=stat_result.st_size,
            connector=None,
            diagnostic=_unsupported(
                source, f"no installed connector supports extension {suffix!r}"
            ),
        )
    return ReferencePlan(
        reference=reference,
        source=source,
        kind="path",
        source_bytes=stat_result.st_size,
        connector=connector,
        diagnostic=None,
    )


def _plan_url(
    reference: str, source: str, scheme: str, *, connectors: Sequence[Connector]
) -> ReferencePlan:
    diagnostic = check_source_id(source)
    if diagnostic is not None:
        return ReferencePlan(
            reference=reference,
            source=source,
            kind="url",
            source_bytes=0,
            connector=None,
            diagnostic=diagnostic,
        )
    connector = _connector_for_url(connectors, reference, scheme)
    if connector is None:
        return ReferencePlan(
            reference=reference,
            source=source,
            kind="url",
            source_bytes=0,
            connector=None,
            diagnostic=_unsupported(source, f"no installed connector supports scheme {scheme!r}"),
        )
    return ReferencePlan(
        reference=reference,
        source=source,
        kind="url",
        source_bytes=0,
        connector=connector,
        diagnostic=None,
    )


def plan_references(
    references: Iterable[str | Path], *, connectors: Sequence[Connector]
) -> tuple[ReferencePlan, ...]:
    """Classifies and dispatches every reference in ``references``, in
    order, before any of them is fetched.

    Classification mirrors :meth:`~taguru_langchain.ingest_connectors.
    html.HtmlConnector.read`'s own local-path-vs-URL check exactly
    (Windows drive path, then ``http``/``https`` scheme, then bare path) —
    an ``http``/``https`` reference's planned ``source`` is therefore the
    same value :meth:`~taguru_langchain.ingest_connectors.html.
    HtmlConnector._read_url` computes as its own pre-fetch ``pre_source``,
    so a fetch that hits no redirect needs no
    :meth:`~taguru_langchain.ingest_connectors.observability.RunRecorder.
    retarget` at all. A reference naming any OTHER scheme (``s3://``,
    ``gs://``, ``az://``, ...) is planned with an ``unsupported_format``
    diagnostic pointing at
    :func:`~taguru_langchain.ingest_connectors.s3.sync_object_storage`.

    A reference whose planned ``source`` was already claimed by an earlier
    reference in this same call is planned with a ``duplicate_source``
    diagnostic and no connector — :func:`sync_references` never calls
    :meth:`~taguru_langchain.ingest_connectors.observability.RunRecorder.
    discovered` for it, only
    :meth:`~taguru_langchain.ingest_connectors.observability.RunRecorder.
    duplicate`.
    """
    registry = SourceIdRegistry()
    plans: list[ReferencePlan] = []
    for raw in references:
        reference = str(raw)
        if _is_windows_drive_path(reference):
            source = file_source_id(reference)
            kind: ReferenceKind = "path"
        else:
            scheme = urlsplit(reference).scheme.lower()
            if scheme in ("http", "https"):
                source = canonicalize_url(_strip_fragment(reference))
                kind = "url"
            elif scheme:
                plans.append(
                    ReferencePlan(
                        reference=reference,
                        source=reference,
                        kind="url",
                        source_bytes=0,
                        connector=None,
                        diagnostic=_unsupported(
                            reference,
                            f"scheme {scheme!r} is not synced by sync_references "
                            "(use sync_object_storage for s3://, gs://, az://, ...)",
                        ),
                    )
                )
                continue
            else:
                source = file_source_id(reference)
                kind = "path"

        if not registry.claim(source):
            plans.append(
                ReferencePlan(
                    reference=reference,
                    source=source,
                    kind=kind,
                    source_bytes=0,
                    connector=None,
                    diagnostic=_duplicate_diagnostic(source),
                )
            )
            continue

        if kind == "path":
            plans.append(_plan_path(reference, source, connectors=connectors))
        else:
            plans.append(_plan_url(reference, source, scheme, connectors=connectors))

    return tuple(plans)


def _is_duplicate(plan: ReferencePlan) -> bool:
    return plan.diagnostic is not None and plan.diagnostic.code == "duplicate_source"


def _file_probe(document: ConnectorDocument, stat_result: os.stat_result) -> FileProbe:
    return FileProbe(
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        parser=document.fingerprint_inputs.parser,
        parser_version=document.fingerprint_inputs.parser_version,
        parse_options_digest=document.fingerprint_inputs.parse_options_digest,
    )


def _save_file_probe(
    file_probe_checkpoint: FileProbeCheckpoint, plan: ReferencePlan, document: ConnectorDocument
) -> None:
    if plan.kind != "path":
        return
    try:
        stat_result = os.stat(plan.reference)
    except OSError:
        # Best-effort: a probe write failure only costs the NEXT dry-run
        # an honest "parsed" instead of "unchanged" — never a false hit.
        return
    file_probe_checkpoint.save(document.source, _file_probe(document, stat_result))


def _dry_run_plan(
    plan: ReferencePlan, *, recorder: RunRecorder, file_probe_checkpoint: FileProbeCheckpoint
) -> None:
    if plan.kind == "url":
        # ADR 0007 §11: a URL performs NO network access under --dry-run,
        # not even a HEAD — "would attempt to fetch" is already the
        # correct, honest answer, and the only one available without one.
        recorder.record(plan.source, "parsed")
        return

    connector = plan.connector
    assert connector is not None  # plan.diagnostic is None, by construction
    try:
        stat_result = os.stat(plan.reference)
    except OSError:
        # The file vanished between planning and this dry-run pass — still
        # honestly "would attempt to fetch" (and would then fail), never a
        # manufactured `unchanged`.
        recorder.record(plan.source, "parsed")
        return
    probe = file_probe_checkpoint.load(plan.source)
    if (
        probe is not None
        and probe.size == stat_result.st_size
        and probe.mtime_ns == stat_result.st_mtime_ns
        and probe.parser == connector.parser
        and probe.parser_version == connector.parser_version
        and probe.parse_options_digest == connector.parse_options_digest
    ):
        recorder.record(plan.source, "unchanged")
    else:
        # No probe, or ANY mismatch (including a metadata match but a
        # changed parser option) — never a false `unchanged`, the one
        # direction ADR 0007 §11 allows dry-run's local-file check to err.
        recorder.record(plan.source, "parsed")


def sync_references(
    references: Iterable[str | Path],
    *,
    ingester: TaguruIngester,
    checkpoints: CheckpointStore,
    connectors: Sequence[Connector] | None = None,
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
    events_out: str | Path | TextIO | None = None,
) -> RunReport:
    """Reads every local-file or ``http(s)://`` reference in ``references``
    and syncs it into ``ingester``, applying the connector-native
    checkpoint (skip re-fetch/re-parse of byte-identical content) and the
    local-file dry-run probe (ADR 0007 §11), and reporting one
    :class:`~taguru_langchain.ingest_connectors.observability.RunReport`.

    ``dry_run`` here means exactly ADR 0007 §11's driver-level contract:
    no network fetch, no local file read beyond a cheap ``stat``, and no
    write to the corpus OR to either checkpoint store. This is a
    DIFFERENT, stricter meaning than
    :meth:`~taguru_langchain.ingest.TaguruIngester.ingest_text`'s own
    ``dry_run`` (which still calls the model and only skips the network
    ``import_batches`` call) — this driver's ``dry_run`` is therefore
    never forwarded into
    :func:`~taguru_langchain.ingest_connectors.bridge.
    ingest_connector_document`; a real (non-dry) call to that bridge
    always runs with its own ``dry_run=False``.

    ``events_out`` is honored even under ``dry_run=True``: the sidecar it
    names is the dry run's entire product, and the caller named the path
    explicitly, so writing to it does not violate dry-run's "touch
    nothing else" contract.

    Every reference is classified and dispatched (:func:`plan_references`)
    — and every non-duplicate reference reported ``discovered`` — before
    any of them is read, so an interruption
    (``should_stop``) after the Nth reference still leaves every later
    reference visible as ``discovered`` in the report rather than absent.
    """
    stop = _normalize_stop(should_stop)
    active_connectors = tuple(connectors) if connectors is not None else default_connectors()
    connector_checkpoint = ConnectorCheckpoint(checkpoints)
    file_probe_checkpoint = FileProbeCheckpoint(checkpoints)

    # A `with` block, not a bare construct-then-close: `events_out`'s file
    # handle must close even when something below raises (a connector's
    # `read()` catches its own exceptions, but `ingest_connector_document`,
    # `connector_checkpoint.load`/`save`, or `recorder.record` itself do
    # not) — `recorder.close()` at the end of a straight-line function body
    # never runs in that case.
    with RunRecorder(connector=PARSER_NAME, events_out=events_out) as recorder:
        plans = plan_references(references, connectors=active_connectors)

        work: list[ReferencePlan] = []
        for plan in plans:
            if _is_duplicate(plan):
                recorder.duplicate(plan.reference, of=plan.source)
                continue
            recorder.discovered(plan.source, source_bytes=plan.source_bytes)
            work.append(plan)

        interrupted = False
        for plan in work:
            if stop():
                interrupted = True
                break

            if plan.diagnostic is not None:
                recorder.record(plan.source, "skipped", diagnostic=plan.diagnostic)
                continue

            connector = plan.connector
            if connector is None:
                # Unreachable by construction (plan_references only omits a
                # diagnostic when it also resolved a connector) — handled
                # defensively rather than raising mid-run.
                recorder.record(
                    plan.source,
                    "failed",
                    diagnostic=Diagnostic(
                        code="unreadable", message="no connector resolved", source=plan.source
                    ),
                )
                continue

            if dry_run:
                _dry_run_plan(plan, recorder=recorder, file_probe_checkpoint=file_probe_checkpoint)
                continue

            try:
                document = connector.read(plan.reference)
            except Exception as error:  # noqa: BLE001 - a connector bug must not abort the run
                recorder.record(
                    plan.source,
                    "failed",
                    diagnostic=Diagnostic(
                        code="unreadable", message=str(error), source=plan.source
                    ),
                )
                continue

            if document.source != plan.source:
                recorder.retarget(plan.source, document.source)
            source = document.source

            if not document.text:
                diagnostic = document.diagnostics[0] if document.diagnostics else None
                recorder.record(source, "failed", diagnostic=diagnostic)
                continue

            recorder.record(source, "parsed", parser=document.fingerprint_inputs.parser)

            if connector_checkpoint.load(source, document.fingerprint_inputs) is not None:
                _save_file_probe(file_probe_checkpoint, plan, document)
                recorder.record(source, "unchanged")
                continue

            with recorder.attached(ingester):
                outcome = ingest_connector_document(ingester, document, should_stop=should_stop)

            recorder.add_dropped(
                locators=outcome.locators_dropped, sections=outcome.sections_dropped
            )

            if outcome.interrupted:
                recorder.record(source, "skipped")
                interrupted = True
                break
            if not outcome.ok:
                recorder.record(
                    source,
                    "failed",
                    diagnostic=Diagnostic(
                        code="unreadable", message=outcome.error or "ingest failed", source=source
                    ),
                )
                continue

            connector_checkpoint.save(document)
            _save_file_probe(file_probe_checkpoint, plan, document)
            recorder.record(source, "imported", parser=document.fingerprint_inputs.parser)

        return recorder.finish(interrupted=interrupted)


__all__ = [
    "PARSER_NAME",
    "ReferenceKind",
    "ReferencePlan",
    "default_connectors",
    "plan_references",
    "sync_references",
]
