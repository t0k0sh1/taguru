"""Repository-root lookup for tests that read files outside the package.

A fixed ``Path(__file__).parents[N]`` depth breaks under mutation runs:
mutmut copies the whole project into ``mutants/`` (one directory deeper)
and runs pytest from there, so the same test file sits at two different
depths depending on who invoked it. Walking up to the ``Cargo.toml``
marker resolves the real repository root in both layouts.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while not (p / "Cargo.toml").is_file():
        if p.parent == p:
            raise RuntimeError("Cargo.toml not found above tests/unit; cannot locate repo root")
        p = p.parent
    return p
