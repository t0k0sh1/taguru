"""Cross-connector observability (ADR 0007 §11, issue #353): the one event
shape, run summary, and ``dry_run`` posture every connector driver —
:func:`~taguru_langchain.ingest_connectors.references.sync_references` and
:func:`~taguru_langchain.ingest_connectors.s3.sync_object_storage` alike —
emits, instead of each connector inventing its own.

Originally implemented once, scoped to S3 only (issue #351/#352,
``ingest_connectors.s3``). This module generalizes that implementation
unchanged in shape — :data:`Phase`/:class:`SourceEvent` are a verbatim
move, :class:`S3SyncReport` is now an alias for :class:`RunReport` (issue
#353's "Implementation note" in ADR 0007 §11.1 explains why an alias, not a
subclass) — plus the JSON serialization and the run-duration field ADR 0007
§11 always asked for but nothing had produced yet.

Two ideas:

- **Per-source record is an event log, not a snapshot** (ADR 0007 §11): one
  :class:`SourceEvent` per phase transition, append-only. :class:`RunReport`
  counts are a tally of each source's *last* phase event only — a source
  that reached ``imported`` counts once, under ``imported``, never
  separately under ``discovered``/``parsed`` too.
- **The summary references the events, it does not inline them** — a
  :class:`RunReport` from a 100k-object bucket sync stays O(1) whether or
  not ``events_out`` was given; :attr:`RunReport.events` is only populated
  when a caller asks for it (``keep_events=True``, the default, chosen so
  existing callers of ``sync_object_storage`` see no change) and
  :attr:`RunReport.events_path` names where the full JSONL sidecar landed,
  the same "summary points at the sidecar" shape
  ``taguru extract``'s own ``--diagnostics-out`` takes.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TextIO, get_args

from .document import Diagnostic

if TYPE_CHECKING:
    from ..events import IngestEvent
    from ..ingest import TaguruIngester

RUN_SUMMARY_VERSION: Final = 1
"""Checked for equality, like :data:`~taguru_langchain.ingest_connectors.
document.CONNECTOR_DOCUMENT_VERSION` — a protocol-breaking change to
:meth:`RunReport.to_dict`'s shape bumps this, never loosens the check to a
range."""

SOURCE_EVENT_KIND: Final = "source_event"
RUN_SUMMARY_KIND: Final = "run_summary"

Phase = Literal["discovered", "unchanged", "parsed", "extracted", "imported", "skipped", "failed"]
"""ADR 0007 §11's seven-state per-source phase vocabulary, verbatim."""

PHASES: Final[tuple[Phase, ...]] = get_args(Phase)


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


@dataclass(slots=True, frozen=True, kw_only=True)
class SourceEvent:
    """One phase transition for one source (ADR 0007 §11) — a source's
    full per-run history is the ordered subsequence of a run's events
    sharing its ``source``.

    ``elapsed_ms`` is milliseconds since this *source's* first event
    (always ``0.0`` on ``discovered``), not since the run started —
    :attr:`RunReport.duration_ms` covers the whole run. ``bytes`` is the
    raw byte count of the source object/file where a driver knows it for
    free (a listing's ``size``, ``os.stat().st_size``) — never the parsed
    text's byte count, which would make a sum over this column meaningless
    across phases that carry no text at all.
    """

    source: str
    phase: Phase
    elapsed_ms: float
    bytes: int = 0
    parser: str | None = None
    diagnostic: Diagnostic | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": SOURCE_EVENT_KIND,
            "source": self.source,
            "phase": self.phase,
            "elapsed_ms": self.elapsed_ms,
            "bytes": self.bytes,
            "parser": self.parser,
            "diagnostic": None if self.diagnostic is None else self.diagnostic.to_dict(),
        }


@dataclass(slots=True, frozen=True, kw_only=True)
class RunReport:
    """One run's machine-readable summary (ADR 0007 §11) — always
    constructed, never behind an opt-in flag (contrast ``taguru extract``'s
    ``--diagnostics-out``, where an unset flag means no record is ever
    assembled at all).

    Every connector's summary shares this one key set — ``tags_dropped``/
    ``deleted_detected``/``retracted`` are specific to the S3 connector
    (ADR 0007 §9) but stay present (structurally zero) for every other
    driver, so a consumer parsing one shape never has to branch on which
    connector produced it.
    """

    connector: str = ""
    duration_ms: float = 0.0
    interrupted: bool = False
    discovered: int = 0
    unchanged: int = 0
    parsed: int = 0
    extracted: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    locators_dropped: int = 0
    sections_dropped: int = 0
    tags_dropped: int = 0
    deleted_detected: int = 0
    retracted: int = 0
    events: tuple[SourceEvent, ...] = ()
    events_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": RUN_SUMMARY_KIND,
            "connector_run": RUN_SUMMARY_VERSION,
            "connector": self.connector,
            "duration_ms": self.duration_ms,
            "interrupted": self.interrupted,
            "discovered": self.discovered,
            "unchanged": self.unchanged,
            "parsed": self.parsed,
            "extracted": self.extracted,
            "imported": self.imported,
            "skipped": self.skipped,
            "failed": self.failed,
            "locators_dropped": self.locators_dropped,
            "sections_dropped": self.sections_dropped,
            "tags_dropped": self.tags_dropped,
            "deleted_detected": self.deleted_detected,
            "retracted": self.retracted,
            "events": len(self.events),
            "events_path": self.events_path,
        }

    def events_jsonl(self) -> str:
        """Renders :attr:`events` as newline-delimited JSON, one line per
        phase transition, each terminated by its own ``\\n`` — the same
        on-disk shape a :class:`SourceEventSink` writes incrementally
        (every :meth:`SourceEventSink.write` appends one line plus a
        trailing newline), so this method's output is byte-identical to
        what ``events_out=`` would have written for the same run. Only
        meaningful when ``keep_events=True`` collected them
        (:class:`RunRecorder`'s default); ``""`` when there are none."""
        return "".join(
            json.dumps(event.to_dict(), ensure_ascii=False) + "\n" for event in self.events
        )


S3SyncReport = RunReport
"""Deprecated alias kept for issue #351/#352's published name. ADR 0007
§11.1 folds S3's own counters (``tags_dropped``/``deleted_detected``/
``retracted``) into the one run summary every connector emits, rather than
a connector-specific subclass — a slotted, frozen dataclass subclass
re-declares its base's ``__slots__`` on Python 3.10 (the
``inherited_slots`` fix landed in 3.11, and this SDK's floor is 3.10), so
an alias is the only option that stays correct on the whole supported
Python range."""


class SourceEventSink:
    """Append-only per-source JSONL sidecar (ADR 0007 §11), one line per
    phase transition, written the moment it lands — mirroring
    ``taguru extract``'s own ``DiagnosticsSink`` (src/extract.rs:2372)
    field for field: truncate-on-open ("describes THIS run, never a prior
    one appended to"), one ``write`` + flush per record, no fsync
    (advisory), and a write/open failure warns exactly once for the run
    then silently drops every later record rather than raising mid-sync.

    A ``str``/``Path`` target is opened (truncating) and owned — ``close``
    closes it. A caller-supplied ``TextIO`` is written to but never closed
    — the caller opened it, the caller closes it, exactly as
    ``TaguruIngester.on_event`` never assumes ownership of a stream a
    caller handed it.
    """

    def __init__(self, target: str | Path | TextIO) -> None:
        self._owns_stream: bool
        self._path: str | None
        self._stream: TextIO | None
        self._warned = False
        if isinstance(target, (str, Path)):
            self._path = str(target)
            try:
                self._stream = open(target, "w", encoding="utf-8")
            except OSError as error:
                warnings.warn(
                    f"SourceEventSink: opening {target!r} raised {error!r}; "
                    "events will not be recorded to a file this run",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._stream = None
                self._warned = True
            self._owns_stream = True
        else:
            self._stream = target
            self._path = getattr(target, "name", None)
            self._owns_stream = False

    @property
    def path(self) -> str | None:
        return self._path

    def write(self, event: SourceEvent) -> None:
        if self._stream is None:
            return
        try:
            self._stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self._stream.flush()
        except (OSError, ValueError) as error:
            # `ValueError` covers "I/O operation on closed file" — a
            # caller-supplied `TextIO` this sink never owns (docstring
            # above) may already be closed by the time a later `write`
            # lands; that must degrade exactly like any other write
            # failure, never raise mid-sync.
            if not self._warned:
                warnings.warn(
                    f"SourceEventSink: writing to {self._path!r} raised {error!r}; "
                    "later events this run will not be recorded",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            self._stream = None

    def close(self) -> None:
        if self._owns_stream and self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None

    def __enter__(self) -> SourceEventSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(slots=True)
class _SourceState:
    started: float
    phase: Phase
    bytes: int = 0


class RunRecorder:
    """Collects one run's :class:`SourceEvent` history and derives its
    :class:`RunReport` (ADR 0007 §11) — the shared bookkeeping every
    connector driver uses instead of each maintaining its own tally.

    ``events_out``, when given, is honored even under ``dry_run=True``:
    the sidecar is the dry run's entire product, and the caller named the
    path explicitly — writing to it does not violate dry-run's "touch
    nothing" contract the way writing to the corpus or a checkpoint store
    would (ADR 0007 §11.1).
    """

    def __init__(
        self,
        *,
        connector: str,
        events_out: str | Path | TextIO | None = None,
        keep_events: bool = True,
    ) -> None:
        self._connector = connector
        self._keep_events = keep_events
        self._sink = None if events_out is None else SourceEventSink(events_out)
        self._events: list[SourceEvent] = []
        self._last_phase: dict[str, Phase] = {}
        self._sources: dict[str, _SourceState] = {}
        self._run_started = time.monotonic()
        self._locators_dropped = 0
        self._sections_dropped = 0
        self._tags_dropped = 0
        self._deleted_detected = 0
        self._retracted = 0

    def _emit(
        self,
        source: str,
        phase: Phase,
        *,
        elapsed_ms: float,
        source_bytes: int,
        parser: str | None,
        diagnostic: Diagnostic | None,
    ) -> None:
        event = SourceEvent(
            source=source,
            phase=phase,
            elapsed_ms=elapsed_ms,
            bytes=source_bytes,
            parser=parser,
            diagnostic=diagnostic,
        )
        if self._keep_events:
            self._events.append(event)
        if self._sink is not None:
            self._sink.write(event)
        self._last_phase[source] = phase

    def discovered(self, source: str, *, source_bytes: int = 0) -> None:
        """Records ``source``'s ``discovered`` event. Always ``elapsed_ms
        == 0.0`` — every later :meth:`record` for the same source is timed
        from this call, not from the run's own start. ``source_bytes`` is
        remembered and reused by :meth:`record` for later phases of the
        same source, since an object's own size does not change between
        phases (unless a later call overrides it explicitly)."""
        self._sources[source] = _SourceState(
            started=time.monotonic(), phase="discovered", bytes=source_bytes
        )
        self._emit(
            source,
            "discovered",
            elapsed_ms=0.0,
            source_bytes=source_bytes,
            parser=None,
            diagnostic=None,
        )

    def record(
        self,
        source: str,
        phase: Phase,
        *,
        source_bytes: int | None = None,
        parser: str | None = None,
        diagnostic: Diagnostic | None = None,
    ) -> None:
        """Records ``source``'s transition to ``phase``. ``source_bytes``
        defaults to whatever :meth:`discovered` recorded for this source
        when omitted (``None``) — pass it explicitly only when a later
        phase learns a size :meth:`discovered` did not have (e.g. after a
        redirect); the explicit value is then remembered for any phase
        still to come, the same way :meth:`discovered`'s own value is."""
        state = self._sources.get(source)
        started = state.started if state is not None else time.monotonic()
        elapsed = _elapsed_ms(started)
        resolved_bytes = source_bytes if source_bytes is not None else 0
        if state is not None:
            state.phase = phase
            if source_bytes is not None:
                state.bytes = source_bytes
            else:
                resolved_bytes = state.bytes
        else:
            self._sources[source] = _SourceState(started=started, phase=phase, bytes=resolved_bytes)
        self._emit(
            source,
            phase,
            elapsed_ms=elapsed,
            source_bytes=resolved_bytes,
            parser=parser,
            diagnostic=diagnostic,
        )

    def retarget(self, old: str, new: str) -> None:
        """Renames ``old`` to ``new`` in this run's tally and timing state
        without emitting a new event — for a source whose final identity
        is only known after a fetch (ADR 0007 §11.1, e.g. an HTTP redirect:
        :class:`~taguru_langchain.ingest_connectors.html.HtmlConnector`
        stamps ``document.source`` with the *post-redirect* URL, per ADR
        0007 §6.1, which can differ from the id :meth:`discovered` recorded
        before the fetch).

        ``old``'s phase history stops counting under ``old`` — the next
        :meth:`record` for either name only ever tallies once, as one
        reference. A no-op when ``old`` was never discovered under this
        run (nothing to rename) or ``old == new`` (nothing changed).

        Not safe when ``new`` is already a DIFFERENT source this run
        already has its own history for (two distinct inputs colliding
        onto the same post-fetch identity, e.g. two different URLs that
        both redirect to the same final URL) — that case silently
        clobbers ``new``'s prior state. Genuinely rare (it requires a
        post-fetch collision neither :func:`~taguru_langchain.
        ingest_connectors.references.plan_references`'s pre-fetch
        dedup nor :meth:`duplicate` can see coming) and out of scope for
        issue #353; flagged here rather than silently mishandled."""
        if old == new:
            return
        state = self._sources.pop(old, None)
        if state is None:
            return
        self._sources[new] = state
        phase = self._last_phase.pop(old, None)
        if phase is not None:
            self._last_phase[new] = phase

    def duplicate(self, reference: str, *, of: str) -> None:
        """Records ``reference`` as a rejected duplicate of the
        already-claimed source id ``of`` (ADR 0007 §11.1, issue #353's
        ``duplicate_source`` diagnostic) — visible in the JSONL trail as
        its own ``skipped`` event, but deliberately keyed by ``reference``
        itself and NEVER folded into :meth:`record`'s last-phase tally for
        ``of``.

        This is not an oversight: ``of`` already has its own, possibly
        still in-progress, phase history from the reference that claimed
        it first — a second reference resolving to the same id is not a
        distinct source in this run, so it must never be able to
        overwrite ``of``'s real outcome (e.g. turning an already-recorded
        ``imported`` back into ``skipped`` just because a later, redundant
        input happened to name the same thing). Calling :meth:`record`
        instead of this method for a known duplicate is the bug this
        method exists to make impossible.

        The event this writes is only guaranteed distinguishable from
        ``of``'s own events BY VALUE (``diagnostic.code ==
        "duplicate_source"``), not always by ``source``: for
        :func:`~taguru_langchain.ingest_connectors.references.
        sync_references`, ``reference`` is the rejected INPUT string,
        which differs from the canonical ``of`` whenever two distinct
        inputs collide (e.g. two URLs differing only by a stripped query
        parameter) — the common case this method was designed around.
        :mod:`~taguru_langchain.ingest_connectors.s3`'s own duplicate-key
        case (the exact same key appearing twice in one `store.list()`
        pass, a store bug/quirk) has no separate "input" identity to key
        by, so ``reference`` and ``of`` are the same string there; a
        JSONL consumer grouping strictly by ``source`` would see that
        source's history as ``discovered`` → ``skipped`` →
        ``parsed``/``extracted``/``imported`` and could misread it as an
        interrupted-then-recovered run rather than "one duplicate listing
        entry, one real import." The tally itself stays correct either
        way (this method never touches it); a consumer that needs to
        separate the two must check ``diagnostic.code`` rather than
        assuming ``source`` alone identifies one JSONL sub-sequence."""
        event = SourceEvent(
            source=reference,
            phase="skipped",
            elapsed_ms=0.0,
            diagnostic=Diagnostic(
                code="duplicate_source",
                message=f"already claimed as {of!r} earlier in this run",
                source=of,
            ),
        )
        if self._keep_events:
            self._events.append(event)
        if self._sink is not None:
            self._sink.write(event)
        # Deliberately no `_last_phase`/`_sources` write for either
        # `reference` or `of` — see the docstring above.

    def add_dropped(self, *, locators: int = 0, sections: int = 0, tags: int = 0) -> None:
        """Accumulates ADR 0007 §5's ``locators_dropped``/``sections_dropped``
        (surfaced via :class:`~taguru_langchain.ingest.IngestOutcome`, which
        a driver otherwise has no reason to inspect) and §9's
        ``tags_dropped`` into this run's totals."""
        self._locators_dropped += locators
        self._sections_dropped += sections
        self._tags_dropped += tags

    def add_deletions(self, *, detected: int, retracted: int) -> None:
        """Accumulates ADR 0007 §9's deletion-policy counters."""
        self._deleted_detected += detected
        self._retracted += retracted

    def on_ingest_event(self, event: IngestEvent) -> None:
        """Translates one :class:`~taguru_langchain.events.IngestEvent`
        into this run's ``extracted`` phase.

        ``ImportStarted`` is the only variant this cares about: it fires
        exactly once per document, exactly after every chunk has been
        extracted and the batch rendered, immediately before the network
        `import_batches` call (``ingest.py``'s ``ingest_text``/
        ``aingest_text``) — precisely the honest moment ``extracted`` needs,
        and the only one available without synthesizing a timestamp
        (ADR 0007 §11.1 rejects that as "the quietly degraded output #217
        forbids"). Every other event variant is ignored here — a driver
        that wants finer-grained reporting installs its own ``on_event``
        and calls this one from it, or uses :meth:`attached` to chain both.
        """
        if event.kind == "import_started":
            self.record(event.source, "extracted")

    @contextmanager
    def attached(self, ingester: TaguruIngester) -> Iterator[None]:
        """Routes ``ingester``'s events through :meth:`on_ingest_event` for
        the duration of the ``with`` block, chaining to whatever callback
        ``ingester.on_event`` already held (never replacing it) and
        restoring the original in ``finally`` even if the block raises.
        ``TaguruIngester._emit`` already wraps every callback in
        try/except + ``warnings.warn``, so a fault in this recorder can
        never break the ingest it is observing.

        Not re-entrant and not thread-safe: at most one driver attaches to
        a given ``ingester`` at a time. Every driver in this package is
        single-threaded and sequential, so this is never exercised
        concurrently in practice.
        """
        previous = ingester.on_event

        def combined(event: IngestEvent) -> None:
            self.on_ingest_event(event)
            if previous is not None:
                previous(event)

        ingester.on_event = combined
        try:
            yield
        finally:
            ingester.on_event = previous

    def finish(self, *, interrupted: bool = False) -> RunReport:
        """Derives this run's :class:`RunReport` — counts are a tally of
        each source's LAST phase event only (ADR 0007 §11), never a
        cumulative count across every phase it passed through."""
        tallies: dict[Phase, int] = dict.fromkeys(PHASES, 0)
        for phase in self._last_phase.values():
            tallies[phase] += 1
        return RunReport(
            connector=self._connector,
            duration_ms=_elapsed_ms(self._run_started),
            interrupted=interrupted,
            discovered=tallies["discovered"],
            unchanged=tallies["unchanged"],
            parsed=tallies["parsed"],
            extracted=tallies["extracted"],
            imported=tallies["imported"],
            skipped=tallies["skipped"],
            failed=tallies["failed"],
            locators_dropped=self._locators_dropped,
            sections_dropped=self._sections_dropped,
            tags_dropped=self._tags_dropped,
            deleted_detected=self._deleted_detected,
            retracted=self._retracted,
            events=tuple(self._events),
            events_path=None if self._sink is None else self._sink.path,
        )

    def close(self) -> None:
        """Closes the events sink, if this recorder owns one. Safe to call
        more than once; a caller-supplied stream is never closed here."""
        if self._sink is not None:
            self._sink.close()

    def __enter__(self) -> RunRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "PHASES",
    "RUN_SUMMARY_KIND",
    "RUN_SUMMARY_VERSION",
    "SOURCE_EVENT_KIND",
    "Phase",
    "RunRecorder",
    "RunReport",
    "S3SyncReport",
    "SourceEvent",
    "SourceEventSink",
]
