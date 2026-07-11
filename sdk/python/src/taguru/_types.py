"""Request-shaped input types.

These are TypedDicts rather than dataclasses: association batches are
assembled in bulk and serialized as-is, so plain dicts avoid a conversion on
the hottest write path. Field names are the wire names.
"""

from __future__ import annotations

from typing import TypedDict

__all__ = ["AssocOp", "QuestionSpec", "SectionSpec"]


class _AssocOpRequired(TypedDict):
    subject: str
    label: str
    object: str
    weight: float


class AssocOp(_AssocOpRequired, total=False):
    """One association to assert: ``{subject, label, object, weight, source?, paragraph?}``.

    ``paragraph`` locates the fact within ``source`` (its blank-line paragraph
    index) and is ignored by the server without a ``source``.
    """

    source: str
    paragraph: int


class QuestionSpec(TypedDict):
    """A doc2query question attached to one paragraph of a stored passage."""

    paragraph: int
    question: str


class SectionSpec(TypedDict):
    """A section label governing paragraphs from ``paragraph`` onward."""

    paragraph: int
    section: str
