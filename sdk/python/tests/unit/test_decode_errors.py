"""``ResponseShapeError``: the two `_decode.py` failure modes (ADR 0005 §9.3).

Both were bare ``ValueError`` before this SDK's contract-compatibility work;
they now raise a dedicated, catchable `TaguruError` subclass instead, purely
additively (`ResponseShapeError` is still a `ValueError` too).
"""

from __future__ import annotations

import dataclasses
import typing

import pytest

from taguru import AuditNames, PassageLookup, PassagePage, ResponseShapeError, TaguruError
from taguru._decode import decode

from .conftest import ok_response, sync_client


@dataclasses.dataclass
class _Wrapped:
    value: int


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


def test_list_field_fed_a_dict_raises_response_shape_error_instead_of_returning_its_keys() -> None:
    """Without the isinstance check, iterating a dict yields its keys, so a
    `list[str]` field silently decoded a dict into a list of its key names
    instead of rejecting the shape mismatch."""
    with pytest.raises(ResponseShapeError, match=r"^expected a list for list\[str\], got dict$"):
        decode(AuditNames, {"total": 1, "names": {"a": 1}})


def test_dict_field_fed_a_list_raises_response_shape_error_instead_of_attribute_error() -> None:
    """Without the isinstance check, `.items()` on a list raised a bare
    `AttributeError` instead of the dedicated, catchable shape error."""
    with pytest.raises(
        ResponseShapeError, match=r"^expected a dict for dict\[str, str\], got list$"
    ):
        decode(PassageLookup, {"passages": ["oops"]})


def test_shape_mismatch_from_a_real_call_is_caught_by_bare_taguru_error() -> None:
    """ADR 0005 §9.3's acceptance case: a server answering the pre-0.4.0
    bare-array shape must not escape an `except TaguruError` handler."""
    client = sync_client(lambda _req: ok_response([]))
    with pytest.raises(TaguruError):
        client.context("sake").search_passages("cue")


def test_an_optional_dataclass_unwraps_through_the_union_branch() -> None:
    """`X | None` must decode through `X`, not fall through as raw data —
    falling through would hand callers a plain dict where a dataclass
    instance is promised."""
    decoded = decode(_Wrapped | None, {"value": 3})
    assert decoded == _Wrapped(value=3)
    assert decode(_Wrapped | None, None) is None


def test_type_hints_are_resolved_once_per_class_then_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`typing.get_type_hints` re-evaluates string annotations on every call
    — the cache is what keeps decoding hot paths off that cost, so pin that
    a second decode of the same class never resolves hints again."""
    real = typing.get_type_hints
    calls = 0

    def counting(cls: type) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return real(cls)

    monkeypatch.setattr("taguru._decode.typing.get_type_hints", counting)

    @dataclasses.dataclass
    class Fresh:
        value: int

    assert decode(Fresh, {"value": 1}) == Fresh(value=1)
    assert decode(Fresh, {"value": 2}) == Fresh(value=2)
    assert calls == 1
