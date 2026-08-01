"""Small text/heading helpers shared by more than one connector — first
extracted out of ``html.py`` (issue #349) when ``docx.py`` (issue #350)
needed the same paragraph-sanitization and breadcrumb-fitting behavior for
its own heading hierarchy. Not a connector itself, and not part of the
protocol (`protocol.py`) — purely internal plumbing.
"""

from __future__ import annotations

import re
from typing import Final

# Collapses an interior blank-line run to a single `\n` — mirrors exactly
# what a connector's paragraph splitting must never let slip through: a
# stray blank line INSIDE what the connector considers one paragraph (e.g.
# two consecutive `<br><br>`, or a `<w:br/>` pair) would otherwise silently
# register as an extra paragraph boundary once
# `taguru_langchain._extract.split_paragraphs` re-derives paragraphs from
# the final `"\n\n".join(...)` text, offsetting every locator/section
# paragraph index after it.
_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n+")


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def sanitize_paragraph_text(text: str) -> str:
    return _BLANK_RUN_RE.sub("\n", text).strip()


def fit_breadcrumb(crumbs: list[str], *, separator: str, max_bytes: int) -> str | None:
    """Joins ``crumbs`` (outermost first) with ``separator``, dropping the
    outermost (least specific) ancestor first until the result fits within
    ``max_bytes`` — ``None`` if even the innermost crumb alone doesn't fit
    (or ``crumbs`` is empty)."""
    while crumbs:
        candidate = separator.join(crumbs)
        if byte_len(candidate) <= max_bytes:
            return candidate
        crumbs = crumbs[1:]
    return None
