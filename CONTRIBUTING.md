# Contributing

## Building and testing

Every change must pass the same three checks CI's `check` job runs,
in order:

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

`--all-targets` reaches the examples too (`cargo test` alone never
builds them); `-D warnings` holds the tree at its current zero-warning
baseline. `cargo test --locked` runs the unit tests (lib, server, MCP
bridge), the HTTP integration tests (which spawn the real binary on a
free port), and the QA recall floor — all against the committed
`Cargo.lock`.

## SDK surface parity

`taguru` ships one identical client surface in Python and TypeScript,
specified once in [sdk/spec/surface.yaml](sdk/spec/surface.yaml) and
checked mechanically so the two languages cannot drift apart:

```sh
python sdk/python/scripts/check_surface.py       # Python vs. the spec
npm run check:surface --workspace=taguru         # from sdk/: TypeScript vs. the spec
python sdk/spec/check_versions.py                # SDK package versions vs. the server's Cargo.toml
```

All three run in CI ([.github/workflows/sdk.yml](.github/workflows/sdk.yml));
a change that adds or renames a public SDK method updates
`surface.yaml` in the same commit.

## Running the examples

[examples/](examples/) holds library-level examples that drive
`taguru::context::Context` directly, no server required. Run them in
the order [examples/README.md](examples/README.md) suggests —
`network_retrieval` → `accurate_retrieval` → `paragraph_corpus` — with
`cargo run --example <name>`.

[examples/langchain/](examples/langchain/) is different: it exercises
the SDK packages against a real, spawned server binary and doubles as
their end-to-end smoke test (deterministic fake chat models stand in
for the LLM, so it still runs offline). CI runs every one of them
after the SDK test suites; see the `python`/`typescript` jobs in
`sdk.yml` for the exact invocation.

## Changelog

[CHANGELOG.md](CHANGELOG.md) follows [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/): user-visible changes
go under `[Unreleased]`, in the matching `### Added` / `### Changed` /
`### Fixed` / `### Security` subsection, as part of the same PR that
makes the change. Note any on-disk format or response-shape change
explicitly, the way existing entries do.
[.github/workflows/release.yml](.github/workflows/release.yml) enforces
this at tag time — it fails the release outright if the version being
tagged has no `## [x.y.z]` section in CHANGELOG.md yet, so promote
`[Unreleased]` to a versioned section (and bump `Cargo.toml`) as part
of cutting the release, not after.

## Platform support

CI builds and tests on `ubuntu-latest` and `ubuntu-24.04-arm` only,
and that is what's verified. Three call sites are unix-specific:
`shutdown_signal` in [src/main.rs](src/main.rs) (SIGTERM, beside
Ctrl-C, triggers graceful shutdown), and in
[src/registry.rs](src/registry.rs), `stage_bytes` (creates private
files — e.g. the OAuth grant store — mode 0600 at open time, so no
readable window exists between create and the secret write) and
`fsync_parent_dir` (fsyncs the parent directory after an atomic
rename, for durability against power loss, not just crash safety).
Each falls back on non-unix targets behind `#[cfg(not(unix))]`: no
SIGTERM handling (Ctrl-C still works), no enforced 0600 mode, no
parent-directory fsync — the server still runs, with weaker
guarantees. macOS takes the unix code path but is not covered by CI.
Windows binaries are not published; see [README.md's "Running in
production"](README.md#running-in-production).
