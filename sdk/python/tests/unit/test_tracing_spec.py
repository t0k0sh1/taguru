"""`taguru._tracing`'s vocabulary against `sdk/spec/tracing.yaml` — the
shared source of truth both SDKs' tracing test suites check themselves
against (see that file's own docstring, and the TypeScript SDK's
`tests/unit/tracing-spec.test.ts`). Deliberately independent of
`test_tracing.py`: this file needs no `opentelemetry-sdk` (no spans are
ever created), just `taguru._tracing`'s own constants — so it runs in
CI's default, OTel-less job too, not only the second OTel-installed pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import yaml

from taguru import _tracing

SPEC_PATH = Path(__file__).resolve().parents[3] / "spec" / "tracing.yaml"


def load_spec() -> dict[str, object]:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def test_names_and_reason_vocabulary_match_the_shared_spec() -> None:
    spec = load_spec()
    assert _tracing.TRACER_NAME == spec["tracer_name"]
    assert _tracing.ROOT_SPAN == spec["root_span"]
    assert _tracing.SKIP_EVENT == spec["skip_event"]
    assert _tracing.REASON_FIELD == spec["reason_field"]
    assert _tracing.CITATION_MISSING_EVENT == spec["citation_missing_event"]
    assert _tracing.CITATION_MISSING_FIELD == spec["citation_missing_field"]
    assert set(get_args(_tracing.Reason)) == set(spec["skip_reasons"])


def test_phase_span_names_match_the_shared_spec() -> None:
    spec = load_spec()
    phase_spans = {
        _tracing.SPAN_RESOLVE,
        _tracing.SPAN_DESCRIBE,
        _tracing.SPAN_QUERY,
        _tracing.SPAN_ACTIVATE,
        _tracing.SPAN_CITATIONS,
        _tracing.SPAN_PASSAGE_FALLBACK,
    }
    assert phase_spans == set(spec["phase_spans"])


def test_root_attribute_keys_match_the_shared_spec() -> None:
    spec = load_spec()
    root_attributes = {
        _tracing.ATTR_ORIGIN_COUNT,
        _tracing.ATTR_ANCHOR_COUNT,
        _tracing.ATTR_ASSOCIATION_COUNT,
        _tracing.ATTR_ACTIVATION_COUNT,
        _tracing.ATTR_CITATION_RETURNED,
        _tracing.ATTR_PASSAGE_HIT_COUNT,
        _tracing.ATTR_FALLBACK_RAN,
    }
    assert root_attributes == set(spec["root_attributes"])


def test_privacy_vocabulary_matches_the_shared_spec() -> None:
    spec = load_spec()
    assert spec["privacy"]["allowed_value_kinds"] == ["count", "flag", "reason"]
    assert spec["privacy"]["no_per_item_signals"] is True
    # `taguru._tracing.Span`'s public surface IS that closed vocabulary —
    # `count`/`flag`/`skip` (this SDK's name for the spec's "reason" kind)
    # /`citation_missing`, nothing else.
    public_methods = {
        name
        for name in vars(_tracing.Span)
        if not name.startswith("_") and callable(getattr(_tracing.Span, name))
    }
    assert public_methods == {"count", "flag", "skip", "citation_missing"}
