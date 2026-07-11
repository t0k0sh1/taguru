#!/usr/bin/env python3
"""One version everywhere: the SDK packages release in lockstep with the server.

The reference is Cargo.toml's [package] version. Every other place a
version lives must carry the same number, and the langchain packages'
dependency on the core `taguru` package must span the matching
next-breaking interval (`>=X.Y.Z,<NEXT` on PyPI, `^X.Y.Z` on npm — the
same interval, spelled in each ecosystem's idiom).

    check_versions.py                 # verify (CI)
    check_versions.py --expect 0.2.0  # verify + pin the reference (release tag)
    check_versions.py --set 0.2.0     # rewrite every site, then print the
                                      # lockfile refreshes that must follow

sdk/package-lock.json is checked but never rewritten — npm owns it. A
bump is not finished until `npm install` in sdk/ has refreshed it; the
Cargo.lock side needs no check here because CI builds with --locked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CARGO = ROOT / "Cargo.toml"
PY_CORE = ROOT / "sdk/python/pyproject.toml"
PY_LANGCHAIN = ROOT / "sdk/python-langchain/pyproject.toml"
TS_CORE = ROOT / "sdk/typescript/package.json"
TS_LANGCHAIN = ROOT / "sdk/typescript-langchain/package.json"
INIT_CORE = ROOT / "sdk/python/src/taguru/__init__.py"
INIT_LANGCHAIN = ROOT / "sdk/python-langchain/src/taguru_langchain/__init__.py"
NPM_LOCK = ROOT / "sdk/package-lock.json"

PLAIN_VERSION = re.compile(r"\d+\.\d+\.\d+")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def next_breaking(version: str) -> str:
    """Lowest version the next breaking release may take (the npm-caret rule)."""
    major, minor, _patch = (int(part) for part in version.split("."))
    return f"0.{minor + 1}.0" if major == 0 else f"{major + 1}.0.0"


def pypi_range(version: str) -> str:
    return f">={version},<{next_breaking(version)}"


def npm_range(version: str) -> str:
    return f"^{version}"


def scan_line(path: Path, pattern: re.Pattern[str], where: str) -> tuple[int, str]:
    """Index and captured value of the single line matching `pattern`."""
    hits = [
        (index, match.group(1))
        for index, line in enumerate(path.read_text().splitlines())
        if (match := pattern.fullmatch(line.strip()))
    ]
    if len(hits) != 1:
        sys.exit(f"{rel(path)}: expected exactly one {where}, found {len(hits)}")
    return hits[0]


def toml_section_version(path: Path, section: str) -> tuple[int, str]:
    """Index and value of the `version = "…"` line inside [section]."""
    current = None
    for index, line in enumerate(path.read_text().splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
        elif current == section and (
            match := re.fullmatch(r'version\s*=\s*"([^"]+)"', stripped)
        ):
            return index, match.group(1)
    sys.exit(f"{rel(path)}: no version line in [{section}]")


PYPROJECT_DEP = re.compile(r'"taguru(>=[^"]*)",?')
PACKAGE_JSON_VERSION = re.compile(r'"version"\s*:\s*"([^"]+)",?')
PACKAGE_JSON_DEP = re.compile(r'"taguru"\s*:\s*"([^"]+)",?')
INIT_VERSION = re.compile(r'__version__ = "([^"]+)"')


def gather(ref: str) -> list[tuple[str, str, str]]:
    """Every (site, actual, wanted) pair the lockstep governs, beyond the reference."""
    sites = []
    for path in (PY_CORE, PY_LANGCHAIN):
        _, version = toml_section_version(path, "project")
        sites.append((f"{rel(path)} version", version, ref))
    _, dep = scan_line(PY_LANGCHAIN, PYPROJECT_DEP, "taguru dependency")
    sites.append((f"{rel(PY_LANGCHAIN)} taguru dependency", dep, pypi_range(ref)))
    for path in (TS_CORE, TS_LANGCHAIN):
        sites.append(
            (f"{rel(path)} version", json.loads(path.read_text())["version"], ref)
        )
    ts_dep = json.loads(TS_LANGCHAIN.read_text())["dependencies"]["taguru"]
    sites.append((f"{rel(TS_LANGCHAIN)} taguru dependency", ts_dep, npm_range(ref)))
    for path in (INIT_CORE, INIT_LANGCHAIN):
        _, version = scan_line(path, INIT_VERSION, "__version__")
        sites.append((f"{rel(path)} __version__", version, ref))
    packages = json.loads(NPM_LOCK.read_text())["packages"]
    for key in ("typescript", "typescript-langchain"):
        sites.append((f"{rel(NPM_LOCK)} {key} version", packages[key]["version"], ref))
    lock_dep = packages["typescript-langchain"].get("dependencies", {}).get("taguru")
    if lock_dep is not None:
        sites.append(
            (
                f"{rel(NPM_LOCK)} typescript-langchain taguru dependency",
                lock_dep,
                npm_range(ref),
            )
        )
    return sites


def check(expect: str | None) -> None:
    _, ref = toml_section_version(CARGO, "package")
    if not PLAIN_VERSION.fullmatch(ref):
        sys.exit(
            f"{rel(CARGO)}: version {ref!r} is not plain X.Y.Z — "
            "the lockstep defines no prerelease mapping across ecosystems"
        )
    mismatches = [
        f"{site}: {actual} (want {wanted})"
        for site, actual, wanted in gather(ref)
        if actual != wanted
    ]
    if expect is not None and ref != expect:
        mismatches.insert(0, f"{rel(CARGO)} [package] version: {ref} (want {expect})")
    if mismatches:
        print(f"version lockstep broken against {ref}:", file=sys.stderr)
        for line in mismatches:
            print(f"  {line}", file=sys.stderr)
        if any(rel(NPM_LOCK) in line for line in mismatches):
            print(
                f"  ({rel(NPM_LOCK)} is refreshed by `npm install` in sdk/)",
                file=sys.stderr,
            )
        sys.exit(1)
    sites = len(gather(ref)) + 1
    print(f"version lockstep: {sites} sites agree at {ref}")


def patch_line(path: Path, index: int, old: str, new: str) -> None:
    lines = path.read_text().splitlines(keepends=True)
    patched = lines[index].replace(old, new, 1)
    if patched == lines[index]:
        sys.exit(f"{rel(path)}: line {index + 1} does not contain {old!r}")
    lines[index] = patched
    path.write_text("".join(lines))


def set_all(target: str) -> None:
    index, version = toml_section_version(CARGO, "package")
    patch_line(CARGO, index, f'"{version}"', f'"{target}"')
    for path in (PY_CORE, PY_LANGCHAIN):
        index, version = toml_section_version(path, "project")
        patch_line(path, index, f'"{version}"', f'"{target}"')
    index, dep = scan_line(PY_LANGCHAIN, PYPROJECT_DEP, "taguru dependency")
    patch_line(PY_LANGCHAIN, index, f'"taguru{dep}"', f'"taguru{pypi_range(target)}"')
    for path in (TS_CORE, TS_LANGCHAIN):
        index, version = scan_line(path, PACKAGE_JSON_VERSION, "version")
        patch_line(path, index, f'"{version}"', f'"{target}"')
        json.loads(path.read_text())  # the patch must leave valid JSON behind
    index, dep = scan_line(TS_LANGCHAIN, PACKAGE_JSON_DEP, "taguru dependency")
    patch_line(TS_LANGCHAIN, index, f'"{dep}"', f'"{npm_range(target)}"')
    json.loads(TS_LANGCHAIN.read_text())
    for path in (INIT_CORE, INIT_LANGCHAIN):
        index, version = scan_line(path, INIT_VERSION, "__version__")
        patch_line(path, index, f'"{version}"', f'"{target}"')
    print(
        f"set {target} in {rel(CARGO)} and every SDK manifest; now refresh the lockfiles:"
    )
    print("  cargo check               # Cargo.lock")
    print("  (cd sdk && npm install)   # sdk/package-lock.json")
    print(f"then re-verify: python3 {rel(Path(__file__).resolve())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--expect",
        metavar="X.Y.Z",
        help="fail unless the lockstep version is exactly this",
    )
    group.add_argument(
        "--set",
        dest="target",
        metavar="X.Y.Z",
        help="rewrite every version site to this",
    )
    args = parser.parse_args()
    for raw in (args.expect, args.target):
        if raw is not None and not PLAIN_VERSION.fullmatch(raw):
            sys.exit(f"{raw!r} is not a plain X.Y.Z version")
    if args.target is not None:
        set_all(args.target)
    else:
        check(args.expect)


if __name__ == "__main__":
    main()
