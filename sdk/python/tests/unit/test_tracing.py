"""Tracing (ADR 0008, #224): span tree shape, skip reasons, attribute
types, `traceparent` propagation, and the privacy sentinel.

Two-phase in CI (.github/workflows/sdk.yml): the default job installs no
`opentelemetry-*` package at all, and every test elsewhere in this suite
must keep passing that way — `taguru._tracing` degrades to a silent no-op
with `opentelemetry-api` absent (see its module docstring). This whole
module needs `opentelemetry-sdk` (an in-memory exporter, a real
`TracerProvider`) to assert anything about the spans it produces, so it
opens with `importorskip`: skipped entirely in the default job, then run
for real in the second job that installs the SDK package and re-runs the
suite. `sdk/spec/tracing.yaml` is this module's contract with the
TypeScript SDK's own `tracing.test.ts` — read together, not just here.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import taguru._tracing as tracing  # noqa: E402

from .conftest import ok_response, sync_client  # noqa: E402

ASSOCIATION = {
    "subject": "青嶺酒造",
    "label": "杜氏",
    "object": "高瀬",
    "weight": 2.0,
    "count": 2,
    "attributions": [
        {"source": "docs/aomine.md", "weight": 2.0, "count": 2, "paragraph": 1, "section": None},
        {"source": "unstored.md", "weight": 1.0, "count": 1, "paragraph": 0, "section": None},
    ],
}


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """A fresh, local `TracerProvider` per test — never the process-global
    one (see `taguru._tracing._provider_override`'s docstring for why)."""
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    monkeypatch.setattr(tracing, "_provider_override", provider)
    return sink


def routed_handler(calls: list[tuple[str, dict[str, str]]]):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        calls.append((path, dict(req.headers)))
        body = json.loads(req.content) if req.content else None
        if path.endswith("/resolve"):
            return ok_response(
                [{"name": "青嶺酒造", "score": 1.0, "tier": "lexical", "kind": "exact"}]
                if body["cue"] == "青嶺"
                else []
            )
        if path.endswith("/describe"):
            return ok_response(
                {
                    "concept": "青嶺酒造",
                    "as_subject": [{"label": "杜氏", "count": 1}],
                    "as_object": [],
                }
            )
        if path.endswith("/query"):
            return ok_response({"total": 1, "matches": [ASSOCIATION]})
        if path.endswith("/activate"):
            return ok_response(
                {
                    "total": 1,
                    "matches": [
                        {"strength": 0.9, "path": ["青嶺酒造"], "association": ASSOCIATION}
                    ],
                }
            )
        if path.endswith("/citations"):
            if body["source"] == "unstored.md":
                return httpx.Response(
                    404, json={"status": "error", "error": "no stored passage", "time": 0.0}
                )
            return ok_response(
                {"text": "杜氏は高瀬。", "source": body["source"], "section": "人物"}
            )
        if path.endswith("/sources/search"):
            return ok_response(
                {
                    "plan": {
                        "contexts": [
                            {
                                "context": "sake",
                                "lanes": {
                                    "bm25": {"ran": True},
                                    "vector": {"ran": False, "reason": "no provider"},
                                },
                            }
                        ]
                    },
                    "hits": [],
                }
            )
        raise AssertionError(path)

    return handler


def by_name(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def one(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = by_name(exporter, name)
    assert len(spans) == 1, f"expected exactly one {name!r}, found {spans}"
    return spans[0]


def skip_reasons(span: ReadableSpan) -> list[str]:
    return [
        event.attributes[tracing.REASON_FIELD]
        for event in span.events
        if event.name == tracing.SKIP_EVENT
    ]


def test_full_run_produces_root_and_every_phase_span(exporter: InMemorySpanExporter) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve(
        "青嶺", labels="杜氏", text_fallback_query="杜氏は誰か", text_fallback_only_if_empty=False
    )

    root = one(exporter, tracing.ROOT_SPAN)
    assert root.attributes["taguru.origin.count"] == 1
    assert root.attributes["taguru.anchor.count"] == 1
    assert root.attributes["taguru.association.count"] == 1
    assert root.attributes["taguru.activation.count"] == 1
    assert root.attributes["taguru.fallback.ran"] is True
    assert skip_reasons(root) == []

    for phase in [
        "taguru.resolve",
        "taguru.describe",
        "taguru.query",
        "taguru.activate",
        "taguru.citations",
        "taguru.passage_fallback",
    ]:
        span = one(exporter, phase)
        assert span.parent is not None
        assert span.parent.span_id == root.context.span_id  # type: ignore[union-attr]
        assert span.context.trace_id == root.context.trace_id  # type: ignore[union-attr]


def test_skip_reasons_for_disabled_steps(exporter: InMemorySpanExporter) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve(
        "青嶺", describe_first=False, fetch_citations=False, text_fallback_query=None
    )

    root = one(exporter, tracing.ROOT_SPAN)
    assert set(skip_reasons(root)) == {
        "describe_disabled",
        "labels_absent",
        "citations_disabled",
        "fallback_not_requested",
    }
    absent_phases = [
        "taguru.describe",
        "taguru.query",
        "taguru.citations",
        "taguru.passage_fallback",
    ]
    for absent in absent_phases:
        assert by_name(exporter, absent) == []
    # anchors resolved and labels absent, but activate still runs unconditionally.
    assert len(by_name(exporter, "taguru.activate")) == 1


def test_no_anchors_skips_the_whole_graph_cluster(exporter: InMemorySpanExporter) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve("無関係", labels="杜氏")

    root = one(exporter, tracing.ROOT_SPAN)
    assert root.attributes["taguru.anchor.count"] == 0
    assert "no_anchors" in skip_reasons(root)
    for absent in ["taguru.describe", "taguru.query", "taguru.activate"]:
        assert by_name(exporter, absent) == []
    # fetch_citations defaults True, so the (empty) phase still opens.
    assert len(by_name(exporter, "taguru.citations")) == 1


def test_citation_misses_aggregate_to_one_event(exporter: InMemorySpanExporter) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve("青嶺")

    citations_span = one(exporter, "taguru.citations")
    missing = [
        event for event in citations_span.events if event.name == tracing.CITATION_MISSING_EVENT
    ]
    assert len(missing) == 1, missing
    assert missing[0].attributes[tracing.CITATION_MISSING_FIELD] == 1


def test_fallback_suppressed_when_graph_answered(exporter: InMemorySpanExporter) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve("青嶺", text_fallback_query="杜氏は誰か")

    root = one(exporter, tracing.ROOT_SPAN)
    assert root.attributes["taguru.fallback.ran"] is False
    assert "fallback_suppressed" in skip_reasons(root)
    assert by_name(exporter, "taguru.passage_fallback") == []


def test_outbound_requests_carry_traceparent_from_the_active_phase_span(
    exporter: InMemorySpanExporter,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve("青嶺", describe_first=False, fetch_citations=False)

    resolve_span = one(exporter, "taguru.resolve")
    resolve_call = next(headers for path, headers in calls if path.endswith("/resolve"))
    traceparent = resolve_call.get("traceparent")
    assert traceparent is not None, calls
    trace_id, span_id = traceparent.split("-")[1:3]
    assert trace_id == f"{resolve_span.context.trace_id:032x}"  # type: ignore[union-attr]
    assert span_id == f"{resolve_span.context.span_id:016x}"  # type: ignore[union-attr]


def test_no_raw_text_reaches_any_span(exporter: InMemorySpanExporter) -> None:
    """Sentinel: cue/label/fallback-query nonces must never appear as a
    span name or an attribute/event value — only counts, flags, and the
    closed reason vocabulary may."""
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    client.context("sake").retrieve(
        "青嶺", labels="杜氏", text_fallback_query="SENTINEL-QUERY-9f2c"
    )

    serialized = json.dumps(
        [
            {
                "name": span.name,
                "attributes": dict(span.attributes or {}),
                "events": [
                    {"name": e.name, "attributes": dict(e.attributes or {})} for e in span.events
                ],
            }
            for span in exporter.get_finished_spans()
        ],
        ensure_ascii=False,
    )
    for nonce in ["青嶺", "青嶺酒造", "杜氏", "SENTINEL-QUERY-9f2c"]:
        assert nonce not in serialized, f"{nonce!r} leaked into a span"


def test_with_no_tracer_provider_configured_retrieve_still_works() -> None:
    """`opentelemetry-api` installed but the application never called
    `set_tracer_provider` — the global default is itself a no-op
    provider. No test fixture here on purpose: this is the state every
    real caller starts in before opting into tracing."""
    calls: list[tuple[str, dict[str, str]]] = []
    client = sync_client(routed_handler(calls), retries=0)
    result = client.context("sake").retrieve("青嶺")
    assert result.associations
