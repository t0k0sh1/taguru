# Taguru client SDKs

Four packages, one surface. The Python and TypeScript SDKs expose the
identical structure, vocabulary, arguments, and returns — method names differ
only by each language's casing convention (`search_passages` ↔
`searchPassages`), and data fields are the wire's own snake_case in both.

| Package | Registry | Directory | What it is |
|---|---|---|---|
| `taguru` | PyPI | [`python/`](python/) | Core client: sync `Taguru` + async `AsyncTaguru`, typed models for every endpoint, idempotency-aware retry, keyset auto-pagination, batched writes, the `retrieve()` loop. Depends on `httpx` only. |
| `taguru` | npm | [`typescript/`](typescript/) | The same core client, `fetch`-based, zero runtime dependencies, ESM+CJS, Node 20+. |
| `langchain-taguru` | PyPI | [`python-langchain/`](python-langchain/) | `TaguruRetriever` (graph lane + text lane, RRF-merged) and `TaguruIngester` (the LangChain twin of `taguru extract`: an LLM decomposes Documents into the graph via `POST /import`). |
| `langchain-taguru` | npm | [`typescript-langchain/`](typescript-langchain/) | The LangChain.js mirror of the above. |

## Parity, enforced

[`spec/surface.yaml`](spec/surface.yaml) declares every logical operation
once; `python/scripts/check_surface.py` and
`typescript/scripts/check-surface.ts` verify each SDK against it in CI
(`.github/workflows/sdk.yml`) — a method added, removed, or renamed in one
language only fails the build.

## Examples

Runnable use-case examples — RAG QA with citations, governed ingestion,
conversational long-term memory — live in
[examples/langchain/](../examples/langchain/), each as a Python and a
TypeScript program mirrored line for line. They run offline (server
spawned, deterministic fake LLMs) and CI executes all of them.

## Behavioral ground truth

There is deliberately no OpenAPI spec. The contract is the server's own
protocol document (`GET /protocol`, i.e. `src/llm-protocol.md`), and both
SDKs' integration suites spawn the real `cargo build` server binary — the
same harness the server's own `tests/http_api.rs` uses. The ingestion
prompt/validation in the langchain packages mirror `src/extract.rs`
(`PROMPT_VERSION` is kept in sync deliberately).

## Development

```sh
# Python (each package has its own venv)
cd sdk/python            && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cd sdk/python-langchain   && python3 -m venv .venv && .venv/bin/pip install -e ../python -e ".[dev]"
.venv/bin/python -m pytest tests            # integration tests build/spawn the real server
                                            # (or set TAGURU_TEST_BIN to a built binary)

# TypeScript (npm workspace rooted here)
cd sdk && npm install
npm run build --workspace=taguru            # langchain-taguru consumes the built dist
npm test --workspaces
```

Regenerate the Python sync client after editing the async source:
`python/scripts/generate_sync.py` (unasync; CI verifies freshness).

## Releasing

The packages version in lockstep with the server — one release, one
number, no per-package tags. The `v*` tag that cuts the server release
(`release.yml`: GitHub Release + crates.io) also fires
`sdk-release.yml`, which publishes `taguru` and `langchain-taguru` to
PyPI and npm. [`spec/check_versions.py`](spec/check_versions.py) pins
every version site — Cargo.toml, both pyproject.toml and package.json
plus their `taguru` dependency ranges, both `__version__`, the npm
lockfile — in SDK CI, and both release jobs check it against the tag.

To prepare a release:

1. `python3 sdk/spec/check_versions.py --set X.Y.Z`
2. `cargo check` and `(cd sdk && npm install)` — refresh both lockfiles
3. `python3 sdk/spec/check_versions.py` — must pass
4. Move the CHANGELOG's Unreleased entries under `## [X.Y.Z]`
5. Commit, tag `vX.Y.Z`, push the tag

One-time registry setup:

- **PyPI** — a `PYPI_API_TOKEN` repository secret (account-scoped until
  the packages exist, then swap in a project-scoped token). While it is
  missing, a release tag skips the PyPI publish loudly instead of
  failing.
- **npm** — [trusted publishing](https://docs.npmjs.com/trusted-publishers):
  no secret. On npmjs.com, each package's Settings → Trusted Publisher
  takes user `t0k0sh1`, repository `taguru`, workflow
  `sdk-release.yml`, environment blank. npmjs.com only offers this to
  packages that already exist, so bootstrap the first version locally:
  `cd sdk && npm install && npm login`, then
  `npm publish --access public --workspace=taguru` and again for
  `--workspace=langchain-taguru` (`prepublishOnly` builds them).
  Configure the trusted publishers, re-run the tag's npm job — it skips
  the versions just published — and every later release is hands-off.
