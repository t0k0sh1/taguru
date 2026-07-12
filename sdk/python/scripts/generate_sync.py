#!/usr/bin/env python3
"""Generate taguru._sync.client from taguru._async.client via unasync.

Run from anywhere; paths are resolved relative to this script. CI re-runs
this and fails if the committed sync client is stale.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import unasync

ROOT = Path(__file__).resolve().parent.parent
ASYNC_DIR = ROOT / "src" / "taguru" / "_async"
SYNC_DIR = ROOT / "src" / "taguru" / "_sync"

REPLACEMENTS = {
    "AsyncTaguru": "Taguru",
    "AsyncContexts": "Contexts",
    "AsyncGroups": "Groups",
    "AsyncContext": "Context",
    "AsyncClient": "Client",
    "AsyncIterator": "Iterator",
    "aiter_bytes": "iter_bytes",
    "aread": "read",
    "aclose": "close",
    "asyncio": "time",
    "__aenter__": "__enter__",
    "__aexit__": "__exit__",
}

SYNC_DOCSTRING = '''"""Sync client — GENERATED, do not edit.

This file is produced from ``taguru._async.client`` by
``scripts/generate_sync.py`` (unasync). Edit the async source and regenerate.
"""'''


def main() -> int:
    rule = unasync.Rule(
        fromdir=str(ASYNC_DIR) + "/",
        todir=str(SYNC_DIR) + "/",
        additional_replacements=REPLACEMENTS,
    )
    unasync.unasync_files([str(ASYNC_DIR / "client.py")], [rule])

    target = SYNC_DIR / "client.py"
    text = target.read_text(encoding="utf-8")

    # Replace the module docstring (unasync only rewrites NAME tokens).
    text = re.sub(r'\A""".*?"""', SYNC_DOCSTRING, text, count=1, flags=re.DOTALL)

    # The asyncio -> time replacement leaves `import time` twice.
    lines = text.splitlines(keepends=True)
    seen_import_time = False
    kept: list[str] = []
    for line in lines:
        if line.strip() == "import time":
            if seen_import_time:
                continue
            seen_import_time = True
        kept.append(line)
    target.write_text("".join(kept), encoding="utf-8")

    # unasync keeps the async source's line wrapping, but dropping `async `
    # lets some signatures fit on one line — and CI holds the committed file
    # to `ruff format --check`. Format here so both checks agree.
    subprocess.run([sys.executable, "-m", "ruff", "format", "--quiet", str(target)], check=True)

    init = SYNC_DIR / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")
    print(f"generated {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
