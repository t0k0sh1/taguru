"""Optional OpenTelemetry tracing for the ``retrieve()`` loop (ADR 0008; #224).

Shared verbatim between the sync and async clients: this module contains
no ``async``/``await``, so ``scripts/generate_sync.py`` never touches it
(unlike ``_async/client.py``, there is only ever one copy of this file).
:func:`span` is a plain — not async — context manager, so the exact same
``with tracing.span(...)`` line reads correctly whether the surrounding
method body is ``async def`` or the generated ``def`` (only ``async with``
needs unasync's token replacement; bare ``with`` does not).

Off by default, in two independent ways: (1) ``opentelemetry-api`` is not a
runtime dependency (see ``pyproject.toml``'s ``otel`` extra) — every name
from it is imported through a ``try/except ImportError`` below, so a
plain ``pip install taguru`` behaves identically whether or not the
package happens to be present; (2) even with the package installed, no
span is ever recorded until the *application* configures a
``TracerProvider`` — this SDK never calls ``set_tracer_provider`` itself.

Privacy (ADR 0008 §9 mirrored client-side): the only ways to put a value
on a span are :meth:`Span.count` (int), :meth:`Span.flag` (bool), and
:meth:`Span.skip` (a closed ``Reason`` string literal, recorded as a
``taguru.skip`` event — never a bare span attribute, so it cannot be
confused with the span's own status). There is deliberately no
``set_attribute(key, str)`` escape hatch: cue text, origins, labels, and
passage content cannot reach a span through this module, full stop —
see ``sdk/spec/tracing.yaml`` for the exact vocabulary both SDKs share.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Literal

# `[[tool.mypy.overrides]]` in pyproject.toml marks the whole
# `opentelemetry.*` namespace `ignore_missing_imports` — both names below
# typecheck as `Any` regardless of whether the package is actually
# installed in whatever environment is running mypy, which is what lets
# the `except` branch's fallback assignment need no `type: ignore` of its
# own here.
try:
    from opentelemetry import propagate as _propagate
    from opentelemetry import trace as _trace
except ImportError:  # pragma: no cover - exercised by the otel-less CI run
    _trace = None
    _propagate = None

__all__ = ["Span", "span", "inject_headers", "Reason"]

TRACER_NAME = "taguru"
ROOT_SPAN = "taguru.retrieve"
SKIP_EVENT = "taguru.skip"
REASON_FIELD = "taguru.reason"
CITATION_MISSING_FIELD = "taguru.citation.missing"

# `retrieve()`'s phase spans, in the order it may open them — named here
# once so `_async/client.py` and this SDK's own spec-parity test share a
# single source instead of three copies of the same six strings.
SPAN_RESOLVE = "taguru.resolve"
SPAN_DESCRIBE = "taguru.describe"
SPAN_QUERY = "taguru.query"
SPAN_ACTIVATE = "taguru.activate"
SPAN_CITATIONS = "taguru.citations"
SPAN_PASSAGE_FALLBACK = "taguru.passage_fallback"

# The root span's attribute keys `retrieve()` records — same reasoning.
ATTR_ORIGIN_COUNT = "taguru.origin.count"
ATTR_ANCHOR_COUNT = "taguru.anchor.count"
ATTR_ASSOCIATION_COUNT = "taguru.association.count"
ATTR_ACTIVATION_COUNT = "taguru.activation.count"
ATTR_CITATION_RETURNED = "taguru.citation.returned"
ATTR_PASSAGE_HIT_COUNT = "taguru.passage.hit_count"
ATTR_FALLBACK_RAN = "taguru.fallback.ran"

# ``retrieve()``'s closed skip-reason vocabulary — kept in lockstep with
# ``sdk/spec/tracing.yaml`` (and, by name only, with ADR 0008's server-side
# list; the SDK sees a narrower slice of the loop than the server does).
Reason = Literal[
    "describe_disabled",
    "no_anchors",
    "labels_absent",
    "citations_disabled",
    "citation_passage_missing",
    "fallback_not_requested",
    "fallback_suppressed",
]

# Test seam: production code never calls this. `opentelemetry.trace.
# set_tracer_provider` is a process-global, one-shot call (a second call
# just logs a warning and is ignored), which would make the test suite
# order-dependent if tests used it to install per-test exporters. Tests
# instead monkeypatch this module attribute directly to hand `_tracer()` a
# fresh local `TracerProvider`, leaving the real global provider (if any)
# untouched.
_provider_override: Any = None


def _tracer() -> Any:
    if _trace is None:
        return None
    return _trace.get_tracer(TRACER_NAME, tracer_provider=_provider_override)


class Span:
    """Wraps a live OTel span — or nothing — behind the three privacy-safe
    setters. Call sites never need to branch on whether tracing is active;
    an inert :class:`Span` silently drops everything."""

    __slots__ = ("_otel_span",)

    def __init__(self, otel_span: object | None) -> None:
        self._otel_span = otel_span

    def count(self, key: str, value: int) -> None:
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, value)  # type: ignore[attr-defined]

    def flag(self, key: str, value: bool) -> None:
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, value)  # type: ignore[attr-defined]

    def skip(self, reason: Reason) -> None:
        """Record a ``taguru.skip`` event: a planned step did not run."""
        if self._otel_span is not None:
            self._otel_span.add_event(SKIP_EVENT, {REASON_FIELD: reason})  # type: ignore[attr-defined]

    def citation_missing(self, count: int) -> None:
        """One aggregate ``taguru.skip`` event (reason
        ``citation_passage_missing``, the count in
        ``taguru.citation.missing``) for every citation lookup that
        404'd during this call — never one event per miss (ADR 0008's
        per-item aggregation rule: a citation locator is not
        attacker-controlled, but it is still unbounded, caller-shaped
        data). The identical shape the server's own retrieve
        composition emits, so one dashboard query reads both."""
        if count and self._otel_span is not None:
            self._otel_span.add_event(  # type: ignore[attr-defined]
                SKIP_EVENT,
                {
                    REASON_FIELD: "citation_passage_missing",
                    CITATION_MISSING_FIELD: count,
                },
            )


_NULL_SPAN = Span(None)


@contextmanager
def span(name: str) -> Iterator[Span]:
    """Open a child span under the current context, or no-op.

    Plain (non-async) context manager on purpose — see the module
    docstring for why that matters to unasync.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NULL_SPAN
        return
    with tracer.start_as_current_span(name) as otel_span:
        yield Span(otel_span)


def inject_headers(headers: MutableMapping[str, str]) -> None:
    """Inject the current span's W3C ``traceparent``/``tracestate`` into
    ``headers``, mutating it in place.

    A no-op with no ``opentelemetry-api`` installed, no active span, or no
    configured propagator — every one of those is the propagator's own
    established no-op behavior, not something this function special-cases.
    """
    if _propagate is None:
        return
    _propagate.inject(headers)
