"""Tolerant response decoding.

The server promises additive-only response evolution (see ``GET /protocol``):
unknown fields must be ignored, and absent optional fields are omitted rather
than null. This decoder walks dataclass type hints recursively and filters
input to declared fields, so new server fields never break old clients.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any

_HINTS_CACHE: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    cached = _HINTS_CACHE.get(cls)
    if cached is None:
        cached = typing.get_type_hints(cls)
        _HINTS_CACHE[cls] = cached
    return cached


def decode(cls: Any, data: Any) -> Any:
    """Decode ``data`` (parsed JSON) as ``cls``, ignoring unknown fields."""
    if data is None:
        return None
    origin = typing.get_origin(cls)
    if origin is list:
        (item_type,) = typing.get_args(cls)
        return [decode(item_type, item) for item in data]
    if origin is dict:
        _key_type, value_type = typing.get_args(cls)
        return {key: decode(value_type, value) for key, value in data.items()}
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(cls) if arg is not type(None)]
        if len(args) == 1:
            return decode(args[0], data)
        return data
    if isinstance(cls, type) and dataclasses.is_dataclass(cls):
        if not isinstance(data, dict):
            raise ValueError(f"expected an object for {cls.__name__}, got {type(data).__name__}")
        hints = _hints(cls)
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if field.name in data:
                kwargs[field.name] = decode(hints[field.name], data[field.name])
            elif (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                raise ValueError(f"missing required field {field.name!r} for {cls.__name__}")
        return cls(**kwargs)
    return data
