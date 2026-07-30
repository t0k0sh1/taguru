"""``ResponseShapeError``: the two `_decode.py` failure modes (ADR 0005 §9.3).

Both were bare ``ValueError`` before this SDK's contract-compatibility work;
they now raise a dedicated, catchable `TaguruError` subclass instead, purely
additively (`ResponseShapeError` is still a `ValueError` too).
"""

from __future__ import annotations

import pytest

from taguru import PassagePage, ResponseShapeError, TaguruError
from taguru._decode import decode

from .conftest import ok_response, sync_client


def test_container_shape_mismatch_raises_response_shape_error() -> None:
    """The literal 0.4.0 `PassagePage` symptom (ADR 0005 §2.1): the
    message text is pinned verbatim — it's the string
    docs/troubleshooting.html names as the canonical skew signal."""
    with pytest.raises(ResponseShapeError, match=r"^expected an object for PassagePage, got list$"):
        decode(PassagePage, [])


def test_response_shape_error_is_also_a_value_error() -> None:
    """Purely additive: code written against the old bare `ValueError`
    behavior keeps working unchanged."""
    with pytest.raises(ValueError):
        decode(PassagePage, [])


def test_missing_required_field_raises_response_shape_error() -> None:
    with pytest.raises(ResponseShapeError, match=r"^missing required field 'plan'"):
        decode(PassagePage, {})


def test_shape_mismatch_from_a_real_call_is_caught_by_bare_taguru_error() -> None:
    """ADR 0005 §9.3's acceptance case: a server answering the pre-0.4.0
    bare-array shape must not escape an `except TaguruError` handler."""
    client = sync_client(lambda _req: ok_response([]))
    with pytest.raises(TaguruError):
        client.context("sake").search_passages("cue")
