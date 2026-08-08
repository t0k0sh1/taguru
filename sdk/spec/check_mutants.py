#!/usr/bin/env python3
"""Mutation-testing gate for the core Python SDK (`sdk/python`).

`mutmut` seeds deliberate faults into the source and reports which ones the
test suite fails to catch ("survived"). A survivor is a hole in the tests:
a line whose behavior nothing actually asserts. This script runs the suite
against every mutant and fails CI when a survivor appears that is not on the
reviewed allowlist of genuinely-equivalent mutants (`mutation-baseline.txt`).

    check_mutants.py            # run mutmut, then verify against the baseline
    check_mutants.py --verify   # verify only (assumes `mutmut run` already ran)

Scope note: only the core SDK is gated. `sdk/python-langchain` carries a
`[tool.mutmut]` config too (so `mutmut run` works there on demand), but its
mutation coverage leans on the server-backed integration suite, so a
per-PR hermetic gate would be mostly noise — it stays a manual tool, the
same posture the Rust side takes with its full `mutants-sweep` workflow.

Determinism: mutmut numbers mutants per function, stably for unchanged
source, so a baseline entry names one exact mutant (e.g.
`taguru._shared.x_encode_name__mutmut_5`). Editing a function renumbers its
mutants; regenerate that function's baseline entries when that happens.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SDK = Path(__file__).resolve().parents[1] / "python"
BASELINE = SDK / "mutation-baseline.txt"


def _load_baseline() -> set[str]:
    lines = BASELINE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _survivors() -> set[str]:
    """Every mutant `mutmut` reports as not caught — survived, plus the
    timeout/suspicious statuses, which are equally "the suite did not
    cleanly kill this" and must not slip through unreviewed."""
    result = subprocess.run(
        [sys.executable, "-m", "mutmut", "results"],
        cwd=SDK,
        capture_output=True,
        text=True,
        check=True,
    )
    escaped: set[str] = set()
    for line in result.stdout.splitlines():
        name, _, status = line.rpartition(": ")
        if status.strip() in {"survived", "timeout", "suspicious"}:
            escaped.add(name.strip())
    return escaped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="skip `mutmut run` and only compare existing results to the baseline",
    )
    args = parser.parse_args()

    if not args.verify:
        subprocess.run([sys.executable, "-m", "mutmut", "run"], cwd=SDK, check=False)

    baseline = _load_baseline()
    escaped = _survivors()

    new = sorted(escaped - baseline)
    stale = sorted(baseline - escaped)

    if stale:
        print(
            "Baseline entries that no longer escape (regenerate them — a function's "
            "mutants renumber when it is edited):",
            file=sys.stderr,
        )
        for name in stale:
            print(f"  - {name}", file=sys.stderr)

    if new:
        print(
            f"\n{len(new)} mutant(s) survived the test suite and are NOT in the "
            "reviewed baseline — add a test that kills each, or, if it is a genuine "
            "equivalent mutant, document it in mutation-baseline.txt:",
            file=sys.stderr,
        )
        for name in new:
            print(f"  - {name}", file=sys.stderr)
            show = subprocess.run(
                [sys.executable, "-m", "mutmut", "show", name],
                cwd=SDK,
                capture_output=True,
                text=True,
                check=False,
            )
            for detail in show.stdout.splitlines():
                print(f"      {detail}", file=sys.stderr)
        return 1

    # A stale baseline is worth surfacing but not worth failing a merge over:
    # it means the tests got STRICTER, never weaker.
    print(f"OK: every surviving mutant ({len(escaped)}) is on the reviewed baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
