"""Unit coverage for `sdk/spec/check_surface_completeness.py` (issue
#625): the router <-> surface.yaml completeness guard `contract-guard`
(CI) runs against every PR. A regression here silently changes what
that guard enforces, so this locks in the extraction/diff logic
against synthetic `fn routes()`/`surface.yaml` text rather than the
live repo state — loaded by path the same way `test_check_contract.py`
already does, since the script is not an installed package.
"""

from __future__ import annotations

import importlib.util

from tests.unit._repo import repo_root

REPO_ROOT = repo_root()

_spec = importlib.util.spec_from_file_location(
    "check_surface_completeness",
    REPO_ROOT / "sdk" / "spec" / "check_surface_completeness.py",
)
assert _spec is not None and _spec.loader is not None
check_surface_completeness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_surface_completeness)


# --- routes_fn_body() / extract_router_paths(): fn routes()'s own scope ---


def test_extract_router_paths_collects_every_route_call_in_the_function() -> None:
    text = """\
fn other_fn() {
    let ignored = Router::new().route("/should-not-appear", get(x));
}

fn routes(state: AppState) -> Router<AppState> {
    let heavy_routes = Router::new()
        .route("/contexts/{name}/compact", post(api::compact_context))
        .route(
            "/contexts/{name}/promote",
            post(api::promote_sources),
        );
    Router::new()
        .route("/health", get(metrics::health))
        .merge(heavy_routes)
}

fn mcp_dispatch() {
    let also_ignored = app.route("/mcp", post(x));
}
"""
    paths = check_surface_completeness.extract_router_paths(text)
    assert paths == {
        "/contexts/{name}/compact",
        "/contexts/{name}/promote",
        "/health",
    }
    assert "/should-not-appear" not in paths
    assert "/mcp" not in paths


def test_extract_router_paths_handles_a_multi_line_route_call() -> None:
    text = """\
fn routes() -> Router<AppState> {
    Router::new().route(
        "/contexts/{name}/schema/validate",
        post(api::validate_schema),
    )
}
"""
    assert check_surface_completeness.extract_router_paths(text) == {
        "/contexts/{name}/schema/validate"
    }


# --- extract_surface_paths(): both YAML block/flow styles, query strings stripped ---


def test_extract_surface_paths_reads_flow_and_block_style_entries() -> None:
    text = """\
classes:
  Taguru:
    flush: { route: "POST /flush" }
    promote:
      route: "POST /contexts/{name}/promote"
      args: [into, sources]
"""
    assert check_surface_completeness.extract_surface_paths(text) == {
        "/flush",
        "/contexts/{name}/promote",
    }


def test_extract_surface_paths_strips_a_query_string_suffix() -> None:
    # promote_dry_run.json's own `route` field carries "?dry_run=true" —
    # the completeness check compares bare path templates only.
    text = 'route: "POST /contexts/{name}/promote?dry_run=true"'
    assert check_surface_completeness.extract_surface_paths(text) == {"/contexts/{name}/promote"}


# --- find_problems(): the three-way diff (forward, reverse, stale allowlist) ---


def test_find_problems_is_empty_when_router_and_surface_agree() -> None:
    router = {"/health", "/contexts/{name}/compact"}
    surface = {"/health", "/contexts/{name}/compact"}
    assert check_surface_completeness.find_problems(router, surface, {}) == []


def test_find_problems_flags_a_router_path_missing_from_surface() -> None:
    router = {"/health", "/contexts/{name}/promote"}
    surface = {"/health"}
    problems = check_surface_completeness.find_problems(router, surface, {})
    assert len(problems) == 1
    assert "/contexts/{name}/promote" in problems[0]
    assert "absent from surface.yaml" in problems[0]


def test_find_problems_allows_an_allowlisted_router_only_path() -> None:
    router = {"/health", "/maintenance/compact"}
    surface = {"/health"}
    allowlist = {"/maintenance/compact": "admin sweep, deliberately SDK-less"}
    assert check_surface_completeness.find_problems(router, surface, allowlist) == []


def test_find_problems_flags_a_surface_path_missing_from_router() -> None:
    router = {"/health"}
    surface = {"/health", "/contexts/{name}/renamed-away"}
    problems = check_surface_completeness.find_problems(router, surface, {})
    assert len(problems) == 1
    assert "/contexts/{name}/renamed-away" in problems[0]
    assert "no longer in fn routes()" in problems[0]


def test_find_problems_flags_a_stale_allowlist_entry() -> None:
    # The route the allowlist excuses has itself been removed from the
    # router — the exclusion is now dead weight, not a live decision.
    router = {"/health"}
    surface = {"/health"}
    allowlist = {"/gone": "used to be admin-only"}
    problems = check_surface_completeness.find_problems(router, surface, allowlist)
    assert len(problems) == 1
    assert "/gone" in problems[0]
    assert "drop the stale ALLOWLIST entry" in problems[0]


def test_the_real_repo_passes_its_own_completeness_check() -> None:
    """Not a synthetic fixture: the live guard `contract-guard` runs.
    Kept last/separate from the synthetic-fixture tests above so a
    failure here reads as "the real repo drifted", not "the checker's
    logic broke" — the synthetic tests already cover the latter.
    """
    router = check_surface_completeness.router_paths()
    surface = check_surface_completeness.surface_paths()
    problems = check_surface_completeness.find_problems(
        router, surface, check_surface_completeness.ALLOWLIST
    )
    assert problems == []
