#!/usr/bin/env python3
"""Check the Python SDK against sdk/spec/surface.yaml.

Verifies, for both the sync and Async class variants: every declared method
exists with exactly the declared positional (args) and keyword-only (options)
parameter names, and no undeclared public method exists. Run with the `taguru`
package importable (pip install -e sdk/python).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import yaml

import taguru

SPEC_PATH = Path(__file__).resolve().parents[2] / "spec" / "surface.yaml"


def params_of(func: object) -> tuple[list[str], list[str]]:
    signature = inspect.signature(func)  # type: ignore[arg-type]
    args: list[str] = []
    options: list[str] = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            options.append(name)
    return args, options


def main() -> int:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    for func_name, declared in spec.get("functions", {}).items():
        func = getattr(taguru, func_name, None)
        if func is None:
            errors.append(f"missing module function: taguru.{func_name}")
            continue
        args, _options = params_of(func)
        if args != declared.get("args", []):
            errors.append(f"taguru.{func_name}: args {args} != spec {declared.get('args', [])}")

    for class_name, methods in spec.get("classes", {}).items():
        for prefix in ("", "Async"):
            variant = prefix + class_name
            cls = getattr(taguru, variant, None)
            if cls is None:
                errors.append(f"missing class: taguru.{variant}")
                continue
            declared_names = set(methods)
            actual_names = {
                name
                for name, member in vars(cls).items()
                if not name.startswith("_") and callable(member)
            }
            for missing in sorted(declared_names - actual_names):
                errors.append(f"{variant}.{missing}: declared in spec but not implemented")
            for extra in sorted(actual_names - declared_names):
                errors.append(f"{variant}.{extra}: public method not declared in spec")
            for method_name, entry in methods.items():
                if method_name not in actual_names:
                    continue
                entry = entry or {}
                args, options = params_of(getattr(cls, method_name))
                want_args = entry.get("args", [])
                want_options = entry.get("options", [])
                if args != want_args:
                    errors.append(f"{variant}.{method_name}: args {args} != spec {want_args}")
                if options != want_options:
                    errors.append(
                        f"{variant}.{method_name}: options {options} != spec {want_options}"
                    )

    if errors:
        print(f"surface parity check FAILED ({len(errors)} problems):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("python surface matches sdk/spec/surface.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
