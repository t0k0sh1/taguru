#!/usr/bin/env python3
"""Breaking-change guard over `tests/fixtures/wire/` (#301, ADR 0005 §4/§9).

Classifies every difference between the golden wire-contract fixtures
committed at HEAD and at some earlier ref, using the same compatible/
breaking table ADR 0005 §4 defines, and fails unless a genuinely
breaking change lands beside a matching `HTTP_CONTRACT`/`MCP_CONTRACT`
bump in `src/api.rs`. This does not replace human classification (see
`tests/fixtures/wire/README.md`); it is the mechanical backstop for the
cases ADR 0005 §4 already gives an unambiguous answer for — a removed
or renamed field, a container-shape change (array ↔ object), a known
enum value disappearing, a newly required request field, or a removed
operation.

Deliberately NOT a general-purpose schema-diff engine (ADR 0005 §1
scopes that out to #220's v0.7.0+ follow-up): array elements are
compared representative-element-only (index 0), and a scalar value
simply changing (a string becoming a different string, a null becoming
a number) is not itself classified — only container-shape changes
(array ↔ object ↔ scalar), field add/remove, and the two `shapes.json`-
declared cases (enum value removed, request field newly required) are.

    check_contract.py --check              # fixtures <-> shapes.json self-consistency
    check_contract.py --base origin/main   # classify HEAD's changes against that ref
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Sibling script, not an installed package — `check_versions` is only
# reliably importable once its own directory is on sys.path (plain
# `python3 check_contract.py` already puts sdk/spec there, but a
# caller that loads this file by path, like
# sdk/python/tests/unit/test_wire_contract.py, would not).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_versions import rel  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WIRE_DIR = ROOT / "tests/fixtures/wire"
API_RS = ROOT / "src/api.rs"

CONTRACT_CONST = re.compile(r"pub\(crate\) const (HTTP|MCP)_CONTRACT: u64 = (\d+);")


# --- path matching: "foo.bar[].baz" — "[]" means "every array element" ---


def collect_by_path(value: object, segments: list[str]) -> list[object]:
    if not segments:
        return [value]
    head, *rest = segments
    is_array = head.endswith("[]")
    key = head[:-2] if is_array else head
    if not isinstance(value, dict) or key not in value:
        return []
    nxt = value[key]
    if is_array:
        if not isinstance(nxt, list):
            return []
        collected: list[object] = []
        for item in nxt:
            collected += collect_by_path(item, rest)
        return collected
    return collect_by_path(nxt, rest)


# --- fixture discovery ---


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def fixture_paths() -> list[Path]:
    return sorted((WIRE_DIR / "http").glob("*.json")) + sorted(
        (WIRE_DIR / "mcp").glob("*.json")
    )


# --- --check: fixtures <-> shapes.json self-consistency ---


def run_check() -> None:
    shapes = load_json(WIRE_DIR / "shapes.json")
    enums: dict[str, list[str]] = shapes.get("enums", {})
    required: dict[str, list[str]] = shapes.get("required_request_fields", {})
    problems = []
    routes_seen: set[str] = set()

    paths = fixture_paths()
    for path in paths:
        fixture = load_json(path)
        for path_expr, allowed in enums.items():
            allowed_set = set(allowed)
            for value in collect_by_path(fixture, path_expr.split(".")):
                if isinstance(value, str) and value not in allowed_set:
                    problems.append(
                        f"{rel(path)}: {path_expr} carries {value!r}, not declared "
                        "in shapes.json's enums"
                    )
        route = fixture.get("route")
        if isinstance(route, str):
            routes_seen.add(route)
        request = fixture.get("request")
        if route in required and isinstance(request, dict):
            for field in required[route]:
                if field not in request:
                    problems.append(
                        f"{rel(path)}: shapes.json marks {field!r} required for "
                        f"{route}, but this fixture's request omits it"
                    )

    # The reverse direction: a route named in required_request_fields
    # with no fixture left to check it against is a stale entry (a
    # renamed or removed route) that nothing else would catch.
    for route in required:
        if route not in routes_seen:
            problems.append(
                f"shapes.json's required_request_fields names {route!r}, which no "
                "fixture's route matches"
            )

    if problems:
        print("fixtures <-> shapes.json are out of sync:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(1)
    print(f"shapes.json self-consistency: {len(paths)} fixtures agree")


# --- --base REF: classify HEAD's changes against REF ---


def git_show(ref: str, relpath: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{relpath}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def git_ls_tree(ref: str, *relpaths: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", *relpaths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def kind(value: object) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return "unknown"


def bucket(value: object) -> str:
    """[`kind`], collapsed to the three buckets a container-shape change
    is classified over — object, array, or everything else."""
    return kind(value) if isinstance(value, (dict, list)) else "scalar"


Finding = tuple[str, str, str]  # (severity, path, detail)


def maybe_parse(value: object) -> object:
    """A string that itself decodes to a JSON object/array is treated as
    that object/array — an MCP tool result carries the whole HTTP body a
    second time this way, as `content[].text` (ADR 0005 §2.4's pass-
    through convention), and a shape change hiding inside that text
    must be diffed structurally, not skipped as an opaque scalar.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, (dict, list)):
            return parsed
    return value


def classify(base: object, head: object, path: str) -> list[Finding]:
    """Structural diff between `base` and `head`: field add/remove and
    container-shape changes only (ADR 0005 §4) — see the module
    docstring for what this deliberately does not attempt.
    """
    base = maybe_parse(base)
    head = maybe_parse(head)
    if bucket(base) != bucket(head):
        return [("BREAKING", path, f"{kind(base)} -> {kind(head)}")]
    findings: list[Finding] = []
    if bucket(base) == "object":
        assert isinstance(base, dict) and isinstance(head, dict)
        for key in base:
            if key not in head:
                findings.append(("BREAKING", f"{path}.{key}", "field removed"))
            else:
                findings += classify(base[key], head[key], f"{path}.{key}")
        for key in head:
            if key not in base:
                findings.append(("compatible", f"{path}.{key}", "field added"))
    elif bucket(base) == "array":
        assert isinstance(base, list) and isinstance(head, list)
        if base and head:
            findings += classify(base[0], head[0], f"{path}[]")
    return findings


def classify_request(
    base: object, head: object, path: str, route: str | None, required_by_route: dict
) -> list[Finding]:
    """[`classify`], with ADR 0005 §4's asymmetric request rule applied:
    a newly added field is only compatible if it is optional — a field
    `shapes.json` marks required for this route is breaking instead
    ("old clients never send it").
    """
    findings = classify(base, head, path)
    required = set(required_by_route.get(route, [])) if route else set()
    upgraded = []
    for severity, finding_path, detail in findings:
        if severity == "compatible" and detail == "field added":
            field = finding_path.rsplit(".", 1)[-1]
            if field in required:
                severity = "BREAKING"
                detail = "field added AND required — old clients never send it"
        upgraded.append((severity, finding_path, detail))
    return upgraded


def diff_fixture(
    base: dict, head: dict, operation: str, required_by_route: dict
) -> list[Finding]:
    findings: list[Finding] = []
    for field in ("route", "method", "status"):
        if field in base and field in head and base[field] != head[field]:
            findings.append(
                (
                    "BREAKING",
                    f"{operation}.{field}",
                    f"{base[field]!r} -> {head[field]!r}",
                )
            )
    route = head.get("route")
    findings += classify_request(
        base.get("request"),
        head.get("request"),
        f"{operation}.request",
        route,
        required_by_route,
    )
    findings += classify(
        base.get("response"), head.get("response"), f"{operation}.response"
    )
    return findings


def diff_shapes(base_shapes: dict, head_shapes: dict) -> list[Finding]:
    findings: list[Finding] = []
    base_enums = base_shapes.get("enums", {})
    head_enums = head_shapes.get("enums", {})
    for path_expr, head_values in head_enums.items():
        if path_expr not in base_enums:
            continue
        missing = set(base_enums[path_expr]) - set(head_values)
        for value in sorted(missing):
            findings.append(
                (
                    "BREAKING",
                    f"shapes.enums[{path_expr}]",
                    f"known value {value!r} removed",
                )
            )
    base_required = base_shapes.get("required_request_fields", {})
    head_required = head_shapes.get("required_request_fields", {})
    for route, head_fields in head_required.items():
        if route not in base_required:
            continue
        added = set(head_fields) - set(base_required[route])
        for field in sorted(added):
            findings.append(
                (
                    "BREAKING",
                    f"shapes.required_request_fields[{route}]",
                    f"{field!r} newly required",
                )
            )
    return findings


def contract_versions(text: str) -> dict[str, int]:
    return {
        match.group(1): int(match.group(2)) for match in CONTRACT_CONST.finditer(text)
    }


def run_base(ref: str) -> None:
    head_shapes = load_json(WIRE_DIR / "shapes.json")
    base_shapes_text = git_show(ref, rel(WIRE_DIR / "shapes.json"))
    base_shapes = json.loads(base_shapes_text) if base_shapes_text else {}
    required_by_route = head_shapes.get("required_request_fields", {})

    head_files = {rel(path) for path in fixture_paths()}
    base_files = set(git_ls_tree(ref, rel(WIRE_DIR / "http"), rel(WIRE_DIR / "mcp")))
    base_files = {
        path
        for path in base_files
        if path.endswith(".json") and "shapes.json" not in path
    }

    findings: list[Finding] = []
    for removed in sorted(base_files - head_files):
        findings.append(("BREAKING", removed, "operation removed"))
    for added in sorted(head_files - base_files):
        findings.append(("compatible", added, "operation added"))

    for relpath in sorted(head_files & base_files):
        head_fixture = load_json(ROOT / relpath)
        base_text = git_show(ref, relpath)
        assert base_text is not None
        base_fixture = json.loads(base_text)
        operation = head_fixture.get("operation", relpath)
        findings += diff_fixture(
            base_fixture, head_fixture, operation, required_by_route
        )

    findings += diff_shapes(base_shapes, head_shapes)

    breaking = [finding for finding in findings if finding[0] == "BREAKING"]
    compatible = [finding for finding in findings if finding[0] == "compatible"]

    for severity, path, detail in findings:
        print(f"{severity:10} {path}\n           {detail}")

    if not breaking:
        print(
            f"\nno breaking wire-contract changes against {ref} ({len(compatible)} compatible)"
        )
        return

    head_versions = contract_versions(API_RS.read_text())
    base_api_text = git_show(ref, rel(API_RS)) or ""
    base_versions = contract_versions(base_api_text)

    bumped = {
        dimension
        for dimension in ("HTTP", "MCP")
        if head_versions.get(dimension, 0) > base_versions.get(dimension, 0)
    }
    if not bumped:
        print(
            f"\n{len(breaking)} breaking change(s) against {ref}, but neither "
            "HTTP_CONTRACT nor MCP_CONTRACT was bumped in src/api.rs — see "
            "ADR 0005 §4/§7 and tests/fixtures/wire/README.md",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"\n{len(breaking)} breaking change(s) against {ref}, matched by a bump to "
        f"{'/'.join(sorted(bumped))}_CONTRACT — see the CHANGELOG entry and migration "
        "note this PR must also carry (ADR 0005 §7)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="verify fixtures <-> shapes.json self-consistency",
    )
    group.add_argument(
        "--base",
        metavar="REF",
        help="classify HEAD's fixture changes against this git ref",
    )
    args = parser.parse_args()
    if args.check:
        run_check()
    else:
        run_base(args.base)


if __name__ == "__main__":
    main()
