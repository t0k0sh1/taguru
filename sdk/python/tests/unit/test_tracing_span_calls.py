"""Exact-argument tests for the privacy-safe ``Span`` seam and the tracer
plumbing, using plain fakes.

Unlike ``test_tracing.py`` (which ``importorskip``s the OpenTelemetry SDK
and inspects exported spans end-to-end), these run in the otel-less
environment too: what they pin is which key/event NAMES and values this
module hands to whatever span implementation is underneath — the closed
vocabulary of ``sdk/spec/tracing.yaml``, argument by argument.
"""

from __future__ import annotations

import pytest

import taguru._tracing as tracing


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: list[tuple[str, object]] = []
        self.events: list[tuple[str, dict[str, object]]] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes.append((key, value))

    def add_event(self, name: str, attributes: dict[str, object]) -> None:
        self.events.append((name, attributes))


def test_count_and_flag_set_exactly_the_given_attribute() -> None:
    fake = _RecordingSpan()
    span = tracing.Span(fake)
    span.count("taguru.origin.count", 3)
    span.flag("taguru.fallback.ran", True)
    assert fake.attributes == [("taguru.origin.count", 3), ("taguru.fallback.ran", True)]


def test_skip_records_the_reason_under_the_skip_event() -> None:
    fake = _RecordingSpan()
    tracing.Span(fake).skip("no_anchors")
    assert fake.events == [(tracing.SKIP_EVENT, {tracing.REASON_FIELD: "no_anchors"})]


def test_citation_missing_records_one_aggregate_event_and_none_for_zero() -> None:
    fake = _RecordingSpan()
    span = tracing.Span(fake)
    span.citation_missing(0)
    assert fake.events == []
    span.citation_missing(2)
    assert fake.events == [
        (
            tracing.SKIP_EVENT,
            {
                tracing.REASON_FIELD: "citation_passage_missing",
                tracing.CITATION_MISSING_FIELD: 2,
            },
        )
    ]


def test_tracer_is_named_and_honors_the_provider_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    class _FakeTrace:
        def get_tracer(self, name: str, tracer_provider: object = None) -> str:
            recorded["name"] = name
            recorded["provider"] = tracer_provider
            return "tracer"

    sentinel = object()
    monkeypatch.setattr(tracing, "_trace", _FakeTrace())
    monkeypatch.setattr(tracing, "_provider_override", sentinel)
    assert tracing._tracer() == "tracer"
    assert recorded == {"name": tracing.TRACER_NAME, "provider": sentinel}


def test_inject_headers_hands_the_very_mapping_to_the_propagator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected: list[object] = []

    class _FakePropagate:
        def inject(self, carrier: object) -> None:
            injected.append(carrier)

    monkeypatch.setattr(tracing, "_propagate", _FakePropagate())
    headers: dict[str, str] = {}
    tracing.inject_headers(headers)
    assert injected == [headers]
    assert injected[0] is headers
