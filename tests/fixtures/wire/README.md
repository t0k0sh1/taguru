# Golden wire-contract fixtures

Machine-readable pins of `http_contract: 1` / `mcp_contract: 1` (ADR
0005 §3, §9) — a representative request/response for every wire shape
this repository has committed to keeping stable within the current
contract version, including #216's evidence-assembly package
(`POST /contexts/{name}/evidence`, `assemble_evidence`). Rust, Python,
and TypeScript all read the same files; `sdk/spec/check_contract.py`
diffs them across a base ref to fail CI on an unclassified breaking
change. This is #301's answer to "no golden wire-shape fixture exists
in the repository today" (ADR 0005 §2.5) — inventory and pinning, not
a schema-generation pivot (ADR 0005 §9).

## Layout

- `shapes.json` — the one thing a fixture file cannot say about itself:
  which request fields are *required* (`required_request_fields`, keyed
  by each fixture's own `route` field — a bare path template, e.g.
  `"/contexts/{name}/evidence"`, since every route this covers today is
  POST-only) and the full known-value set for every
  open-string enum-like field (`enums`, keyed by a dotted path into a
  fixture's `response`, with `[]` meaning "every array element").
  `volatile_fields` names the field names blanked to a fixed placeholder
  before a fixture is written or compared — `time` (elapsed seconds,
  different every call), `server`/`version` (this build's own
  `Cargo.toml` version, e.g. `GET /version`'s `server` and `GET
  /health`'s `version`), and `last_read_epoch`/`last_write_epoch` (a
  directory entry's own wall-clock usage stamps) — so a slow CI run, a
  routine version bump, or the literal moment a fixture happened to
  regenerate never shows up as wire drift. Applies inside an MCP tool
  result's `content[].text` too, which carries the whole HTTP body a
  second time as a JSON string (ADR 0005 §2.4's pass-through
  convention). **Documentation only** — the actual blanking is
  `tests/http_api/contract.rs`'s `normalize_volatile`, which hardcodes
  the same five field names (each needs its own placeholder *type*:
  `time` a float, `server`/`version` a string, the two epoch fields an
  integer, which a flat list can't express); nothing reads this array
  back. A new volatile field needs an entry both here and in
  `normalize_volatile` — no check ties the two together, so keep them
  in sync by hand.
- `http/*.json` — one file per HTTP operation: `{operation, contract,
  method, route, status, request, response}`. `response` is the exact
  parsed JSON body, envelope included where the real endpoint has one.
- `mcp/*.json` — one file per MCP operation, same shape with `route`
  instead of `method`/`route` split (e.g. `"tools/call
  assemble_evidence"`), covering the MCP-specific envelope
  (`isError`/`structuredContent`) and the `assemble_evidence` tool's
  `inputSchema` — not a second copy of every HTTP shape, since 45 of 46
  tools inherit `http_contract` by pass-through (ADR 0005 §2.4) and are
  already pinned under `http/`.

## Updating a fixture

1. Change the implementation.
2. Regenerate every fixture this module owns from a live server:
   ```sh
   TAGURU_UPDATE_WIRE_FIXTURES=1 cargo test --test http_api contract
   ```
   This rewrites `http/*.json`/`mcp/*.json` in place from the real
   server binary's real responses — nothing here is hand-typed.
3. `git diff tests/fixtures/wire/` and classify each change against
   [ADR 0005 §4](/adr/0005-wire-contract-compatibility.md#4-compatible--breaking-change-table):
   an added optional field or a new operation is compatible; a removed
   or renamed field, a container-shape change (array ↔ object), a
   pagination-envelope change, or a known enum value disappearing from
   `shapes.json`'s `enums` is breaking.
4. If anything is breaking: bump `HTTP_CONTRACT` and/or `MCP_CONTRACT`
   in `src/api.rs` in the **same PR**, and add the CHANGELOG
   `[Unreleased]` → `### Changed` entry plus a migration note (ADR 0005
   §7) — a caller reading the fixture diff alone should be able to act
   on it.
5. Adding a field that closes over a fixed set of values (a new `kind`,
   `lane`, `reason`, or `ErrorCode`)? Add it to `shapes.json`'s `enums`
   in the same commit — `sdk/spec/check_contract.py --base <ref>`
   otherwise cannot tell "a known value is now missing" (breaking) from
   "the fixture just didn't happen to exercise it" (nothing changed).
   Similarly, a newly *required* request field goes in
   `required_request_fields`; a newly optional one does not.
6. Confirm the guard agrees with your own classification:
   ```sh
   python sdk/spec/check_contract.py --base origin/main
   ```
   `--check` alone (no `--base`) verifies fixtures stay self-consistent
   with `shapes.json` without needing a base ref.

## Why this shape

- **JSON, not YAML**: Rust has no YAML dependency in this tree and the
  TypeScript SDK ships with zero runtime dependencies — JSON is the one
  format all three read with nothing extra installed.
- **Fixtures are generated, never hand-typed**: a fixture is exactly
  what the real server, seeded with a small deterministic corpus,
  answered — the same discipline `sdk/python/scripts/generate_sync.py`
  and `sdk/spec/check_versions.py --set` already use elsewhere in this
  tree (regenerate mechanically, review the diff, commit).
- **Committed fixtures already carry the placeholder volatile values**:
  normalization happens once, at generation time, so a plain
  `git diff`/`assert_eq!` across languages needs no special-casing.
