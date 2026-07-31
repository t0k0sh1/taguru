"""Unit coverage for `sdk/spec/check_contract.py` (#301, ADR 0005 §4):
the breaking-change guard `contract-guard` (CI) runs against every PR.
A regression here silently changes what that guard enforces, so this
locks in the classification rules the module's own docstring documents
— loaded by path the same way `test_wire_contract.py` already does,
since the script is not an installed package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

_spec = importlib.util.spec_from_file_location(
    "check_contract", REPO_ROOT / "sdk" / "spec" / "check_contract.py"
)
assert _spec is not None and _spec.loader is not None
check_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_contract)


# --- classify(): field add/remove, container-shape changes ---


def test_classify_flags_a_removed_field_as_breaking() -> None:
    findings = check_contract.classify({"a": 1, "b": 2}, {"a": 1}, "root")
    assert findings == [("BREAKING", "root.b", "field removed")]


def test_classify_flags_an_added_field_as_compatible() -> None:
    findings = check_contract.classify({"a": 1}, {"a": 1, "b": 2}, "root")
    assert findings == [("compatible", "root.b", "field added")]


def test_classify_flags_array_to_object_as_breaking() -> None:
    findings = check_contract.classify({"hits": []}, {"hits": {}}, "root")
    assert findings == [("BREAKING", "root.hits", "array -> object")]


def test_classify_ignores_a_bare_scalar_value_change() -> None:
    # ADR 0005 §4 classifies field add/remove and container-shape
    # changes; a same-typed value simply changing is not itself a
    # wire-shape change (the module docstring's own stated scope).
    findings = check_contract.classify({"status": "ok"}, {"status": "degraded"}, "root")
    assert findings == []


def test_classify_recurses_into_a_shared_nested_object() -> None:
    base = {"plan": {"ran": True}}
    head = {"plan": {"ran": True, "reason": "skipped"}}
    findings = check_contract.classify(base, head, "root")
    assert findings == [("compatible", "root.plan.reason", "field added")]


def test_classify_compares_only_the_first_array_element() -> None:
    # Documented, deliberate scope limit (module docstring) — a shape
    # change only visible past index 0 is out of scope for this guard.
    base = {"items": [{"a": 1}]}
    head = {"items": [{"a": 1}, {"a": 1, "b": 2}]}
    assert check_contract.classify(base, head, "root") == []


# --- classify_request(): ADR 0005 §4's asymmetric request rule ---


def test_classify_request_flags_a_new_top_level_required_field() -> None:
    base = {"origins": ["x"]}
    head = {"origins": ["x"], "query": "y"}
    required = {"/contexts/{name}/evidence": ["query"]}
    findings = check_contract.classify_request(
        base, head, "op.request", "/contexts/{name}/evidence", required
    )
    assert findings == [
        (
            "BREAKING",
            "op.request.query",
            "field added AND required — old clients never send it",
        )
    ]


def test_classify_request_leaves_a_new_top_level_optional_field_compatible() -> None:
    base = {"origins": ["x"]}
    head = {"origins": ["x"], "query": "y"}
    required = {"/contexts/{name}/evidence": ["origins"]}
    findings = check_contract.classify_request(
        base, head, "op.request", "/contexts/{name}/evidence", required
    )
    assert findings == [("compatible", "op.request.query", "field added")]


def test_classify_request_does_not_confuse_a_nested_field_with_a_required_top_level_one() -> None:
    # Regression: a route requires top-level `query`; a NEW nested
    # `filter.query` must not be misclassified as that same field —
    # only the exact top-level path counts, not just the last segment.
    base = {"origins": ["x"], "filter": {}}
    head = {"origins": ["x"], "filter": {"query": "nested, not top-level"}}
    required = {"/contexts/{name}/evidence": ["query"]}
    findings = check_contract.classify_request(
        base, head, "op.request", "/contexts/{name}/evidence", required
    )
    assert findings == [("compatible", "op.request.filter.query", "field added")]


def test_classify_request_with_no_route_never_promotes_to_breaking() -> None:
    base = {"origins": ["x"]}
    head = {"origins": ["x"], "query": "y"}
    required = {"/contexts/{name}/evidence": ["query"]}
    findings = check_contract.classify_request(base, head, "op.request", None, required)
    assert findings == [("compatible", "op.request.query", "field added")]


# --- diff_shapes(): the two shapes.json-declared breaking cases ---


def test_diff_shapes_flags_a_removed_known_enum_value() -> None:
    base_shapes = {"enums": {"result.kind": ["a", "b"]}}
    head_shapes = {"enums": {"result.kind": ["a"]}}
    findings = check_contract.diff_shapes(base_shapes, head_shapes)
    assert findings == [("BREAKING", "shapes.enums[result.kind]", "known value 'b' removed")]


def test_diff_shapes_does_not_flag_a_newly_added_enum_value() -> None:
    base_shapes = {"enums": {"result.kind": ["a"]}}
    head_shapes = {"enums": {"result.kind": ["a", "b"]}}
    assert check_contract.diff_shapes(base_shapes, head_shapes) == []


def test_diff_shapes_flags_a_newly_required_request_field() -> None:
    base_shapes = {"required_request_fields": {"/x": ["a"]}}
    head_shapes = {"required_request_fields": {"/x": ["a", "b"]}}
    findings = check_contract.diff_shapes(base_shapes, head_shapes)
    assert findings == [("BREAKING", "shapes.required_request_fields[/x]", "'b' newly required")]


def test_diff_shapes_does_not_flag_a_field_no_longer_required() -> None:
    base_shapes = {"required_request_fields": {"/x": ["a", "b"]}}
    head_shapes = {"required_request_fields": {"/x": ["a"]}}
    assert check_contract.diff_shapes(base_shapes, head_shapes) == []


# --- collect_by_path(): MCP pass-through unwrap (ADR 0005 §2.4) ---


def test_collect_by_path_finds_a_value_behind_a_plain_object_walk() -> None:
    fixture = {"response": {"result": {"items": [{"kind": "passage"}]}}}
    values = check_contract.collect_by_path(fixture, "response.result.items[].kind".split("."))
    assert values == ["passage"]


def test_collect_by_path_unwraps_an_mcp_tool_results_embedded_json_text() -> None:
    # The MCP pass-through convention: the same body lives a second
    # time as JSON text inside content[].text, not as a nested object.
    embedded = json.dumps({"result": {"items": [{"kind": "association"}]}})
    fixture = {"response": {"content": [{"type": "text", "text": embedded}]}}
    values = check_contract.collect_by_path(fixture, "response.result.items[].kind".split("."))
    assert values == ["association"]


def test_collect_by_path_returns_nothing_for_a_genuinely_missing_path() -> None:
    fixture = {"response": {"status": "ok"}}
    values = check_contract.collect_by_path(fixture, "response.result.items[].kind".split("."))
    assert values == []


# --- bucket()/kind(): container-shape classification ---


def test_bucket_collapses_scalars_kind_stays_specific() -> None:
    assert check_contract.bucket("x") == "scalar"
    assert check_contract.bucket(1) == "scalar"
    assert check_contract.bucket(None) == "scalar"
    assert check_contract.bucket({}) == "object"
    assert check_contract.bucket([]) == "array"
    assert check_contract.kind("x") == "string"
    assert check_contract.kind(1) == "number"
    assert check_contract.kind(None) == "null"
