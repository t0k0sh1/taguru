#!/usr/bin/env python3
"""Router <-> sdk/spec/surface.yaml completeness (issue #625 finding 1).

`check_contract.py` classifies wire-fixture drift; `check_surface.py`
(Python) and `check-surface.ts` (TypeScript) verify each SDK's own method
signatures match `surface.yaml`. None of the three ever compares
`surface.yaml` against the HTTP router itself (`fn routes()` in
`src/main.rs`) — so an endpoint can go live with no SDK exposure at all
and nothing catches it (issue #625 found three this way: `promote`, `GET
.../communities`, `GET .../embeddings`, all closed earlier in the same
series). This script closes that gap: every path template `fn routes()`
registers must appear in `surface.yaml`'s `route:` values, and vice versa,
unless explicitly allowlisted below with a reason.

Scoped to `fn routes()`'s own body only — deliberately not the whole
file. `/mcp` (registered separately, after `routes()` returns) and the
OAuth discovery endpoints (`src/oauth_http.rs`) are a different
transport/protocol layer, not domain operations an SDK method wraps; the
sharding router's own smaller table (`src/route.rs`) is a different
binary mode entirely. None of those belong in an SDK surface, so none of
them are in scope for this check.

    check_surface_completeness.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_RS = ROOT / "src" / "main.rs"
SURFACE_YAML = ROOT / "sdk" / "spec" / "surface.yaml"

# Endpoints deliberately absent from the SDK surface, each with why —
# checked against fn routes()'s current path set every run, so a stale
# entry (the route itself later removed) is flagged rather than silently
# forgiven forever.
ALLOWLIST: dict[str, str] = {
    "/version": (
        "internal contract-probe plumbing only (both SDKs' "
        "_ensure_contract/_probe_contract preflight) — never called as a "
        "public SDK operation, so it has no method to declare a route for"
    ),
    "/maintenance/compact": (
        "found alongside the router/surface.yaml gap this script closes "
        "(issue #625), but out of scope for that issue's three named "
        "endpoints — likely another oversight (POST /flush and POST "
        "/contexts/{name}/compact, equally admin, are both in the SDK); "
        "candidate for a follow-up issue rather than silently added here"
    ),
}

ROUTE_CALL = re.compile(r'\.route\(\s*"([^"]+)"')
ROUTE_YAML_VALUE = re.compile(r'route:\s*"(?:[A-Z]+ )?([^"?]+)')
ROUTES_FN_START = re.compile(r"^fn routes\(", re.MULTILINE)
# A bare `}` (column 0) followed by another line or straight to EOF —
# Rust does not require a trailing newline, so `\n}\n` alone would miss
# `fn routes()` were it ever the file's last item.
ROUTES_FN_END = re.compile(r"\n}(?:\n|$)")


def routes_fn_body(text: str) -> str:
    start_match = ROUTES_FN_START.search(text)
    if start_match is None:
        raise ValueError("no top-level `fn routes(` found")
    start = start_match.start()
    # `fn routes()` is the only top-level item in this range, so its own
    # closing brace is the first bare `}` (column 0) after the opening —
    # unambiguous without a real brace-matching parser.
    end_match = ROUTES_FN_END.search(text, start)
    if end_match is None:
        raise ValueError("no closing `}` found for `fn routes(`")
    return text[start : end_match.start()]


def extract_router_paths(main_rs_text: str) -> set[str]:
    body = routes_fn_body(main_rs_text)
    return {match.group(1) for match in ROUTE_CALL.finditer(body)}


def extract_surface_paths(surface_yaml_text: str) -> set[str]:
    return {match.group(1) for match in ROUTE_YAML_VALUE.finditer(surface_yaml_text)}


def router_paths() -> set[str]:
    return extract_router_paths(MAIN_RS.read_text(encoding="utf-8"))


def surface_paths() -> set[str]:
    return extract_surface_paths(SURFACE_YAML.read_text(encoding="utf-8"))


def find_problems(router: set[str], surface: set[str], allowlist: dict[str, str]) -> list[str]:
    missing_from_surface = sorted((router - surface) - set(allowlist))
    missing_from_router = sorted(surface - router)
    stale_allowlist = sorted(path for path in allowlist if path not in router)

    problems: list[str] = []
    for path in missing_from_surface:
        problems.append(f"{path}: registered in fn routes() but absent from surface.yaml")
    for path in missing_from_router:
        problems.append(f"{path}: declared in surface.yaml but no longer in fn routes()")
    for path in stale_allowlist:
        problems.append(
            f"{path}: allowlisted as router-only, but fn routes() no longer registers it "
            "— drop the stale ALLOWLIST entry"
        )
    return problems


def main() -> int:
    router = router_paths()
    surface = surface_paths()
    problems = find_problems(router, surface, ALLOWLIST)

    if problems:
        print("router <-> surface.yaml completeness check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"fn routes() and surface.yaml agree on {len(router)} paths ({len(ALLOWLIST)} allowlisted)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
