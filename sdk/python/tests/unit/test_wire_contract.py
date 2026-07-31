"""Golden wire-contract fixtures (#301): the Python SDK's own read of
``tests/fixtures/wire/``, alongside Rust's generator/verifier
(``tests/http_api/contract.rs``) and TypeScript's
(``sdk/typescript/tests/unit/wire-contract.test.ts``). Two checks:

- every fixture whose response this SDK already has a typed model for
  (``MatchPage``, ``PassagePage``, ``ContextPage``, ``ExplorePage``,
  ``ActivationPage``, ``CommunityPage``, and the five ``evidence_*``
  operations decoding into ``EvidencePackage``, #306) decodes through the
  real ``taguru._decode.decode`` — the same function every live call
  uses — without error;
- every fixture's declared enum-like fields only carry values
  ``shapes.json`` knows about, reusing ``sdk/spec/check_contract.py``'s
  own path matcher so the two checkers cannot silently disagree about
  what a path expression means.

The MCP-specific envelope (``assemble_evidence_call``/``tool_schema``/
``tool_error``) has no SDK model — the whole HTTP body rides inside
``content[0].text`` as an opaque string (ADR 0005 §2.4) — so those three
fixtures are covered here only by the shapes-driven structural check
below, not a typed decode.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from taguru._decode import decode
from taguru._models import (
    ActivationPage,
    CommunityPage,
    ContextPage,
    EvidencePackage,
    ExplorePage,
    MatchPage,
    PassagePage,
)

# sdk/python/tests/unit/test_wire_contract.py -> repo root: same depth
# sdk/python-langchain/tests/unit/test_extract.py's own comment climbs
# (unit, tests, python, sdk).
REPO_ROOT = Path(__file__).resolve().parents[4]
WIRE_DIR = REPO_ROOT / "tests" / "fixtures" / "wire"

# sdk/spec/check_contract.py is a script, not an installed package —
# loaded by path so this test reuses its `collect_by_path` matcher
# instead of a second copy that could silently drift from it.
_spec = importlib.util.spec_from_file_location(
    "check_contract", REPO_ROOT / "sdk" / "spec" / "check_contract.py"
)
assert _spec is not None and _spec.loader is not None
check_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_contract)


def _load_fixtures() -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((WIRE_DIR / "http").glob("*.json")) + sorted((WIRE_DIR / "mcp").glob("*.json"))
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


FIXTURES = _load_fixtures()
FIXTURES_BY_STEM = {path.stem: fixture for path, fixture in FIXTURES}
SHAPES = json.loads((WIRE_DIR / "shapes.json").read_text(encoding="utf-8"))

# operation -> the model this SDK already decodes its response into.
TYPED_OPERATIONS = {
    "recall": MatchPage,
    "contexts_list": ContextPage,
    "sources_search": PassagePage,
    "explore": ExplorePage,
    "activate": ActivationPage,
    "communities_search": CommunityPage,
    "evidence_mixed_lanes": EvidencePackage,
    "evidence_budget_constrained": EvidencePackage,
    "evidence_duplicate_passage": EvidencePackage,
    "evidence_contradiction_group": EvidencePackage,
    "evidence_communities_degrade_and_rerank_reason": EvidencePackage,
}


def test_wire_fixture_corpus_is_not_empty() -> None:
    assert FIXTURES, "tests/fixtures/wire must carry at least one fixture"
    assert set(TYPED_OPERATIONS) <= {path.stem for path, _ in FIXTURES}


@pytest.mark.parametrize("operation", sorted(TYPED_OPERATIONS), ids=sorted(TYPED_OPERATIONS))
def test_typed_operations_decode_through_the_real_sdk_decoder(operation: str) -> None:
    fixture = FIXTURES_BY_STEM[operation]
    model = TYPED_OPERATIONS[operation]
    decoded = decode(model, fixture["response"]["result"])
    assert decoded is not None


@pytest.mark.parametrize("path,fixture", FIXTURES, ids=[path.name for path, _ in FIXTURES])
def test_every_declared_enum_only_carries_known_values(path: Path, fixture: dict[str, Any]) -> None:
    for path_expr, allowed in SHAPES["enums"].items():
        allowed_set = set(allowed)
        for value in check_contract.collect_by_path(fixture, path_expr.split(".")):
            if isinstance(value, str):
                assert value in allowed_set, (
                    f"{path.name}: {path_expr} carries {value!r}, which is not "
                    "declared in shapes.json's enums"
                )
