# 0005. Public wire contract inventory and compatibility policy

- **Status**: Accepted
- **Date**: 2026-07-30
- **Issue**: #299
- **Related**: #220, #216, #300, #301, #302, ADR 0002 §10, ADR 0003 §10
- **Supersedes**: ADR 0002 §10 (partially — see §6) / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What counts as a compatible or a breaking change to Taguru's public wire
contract — HTTP, MCP, and the two SDKs' decoding of both — and the minimal
version-dimension split and discovery surface #300 needs to enforce it before
#216's new evidence-assembly API (#302) can be added safely. This is the first
of #220's three v0.6.0 child issues; #300 (contract version discovery and the
SDK supported range) and #301 (golden wire fixtures and a breaking-change CI
guard) both build on the decisions here.

Out of scope, left to #220's v0.7.0+ follow-up as its own issue named it: a
general-purpose schema diff gate, a full cross-version compatibility matrix,
router/shard/replica version-skew enforcement, and any HTTP contract v2 /
path-negotiation scheme. This ADR also does not touch #216's own candidate,
budget, dedup, or reranker semantics — that is #302's job; this ADR only
states the wire-contract constraints #302's design must satisfy (§8).

No code changes ship with this ADR. It is inventory and decision only; the new
`GET /version` surface, the SDK supported-range check, and the
`llm-protocol.md` edit it implies are #300's implementation.

## 2. Context: the public surface as it exists today

### 2.1 The 0.4.0 precedent this ADR exists to prevent

`POST /contexts/{name}/sources/search` (and its cross-context form) changed
from a bare array of hits to a `PassagePage` object, `{plan, hits}`, in 0.4.0.
The hits were unchanged; they moved under a `hits` key and gained a sibling
`plan`. A 0.3.x SDK against a 0.4.x server failed with `expected an object for
PassagePage, got list`; a 0.4.x SDK against a 0.3.x server failed with no
`hits` key to find (`docs/troubleshooting.html` `#compatibility`). Nothing in
the tree today would have caught this before release — there was no
compatibility classification to check the change against, and there still
isn't.

### 2.2 Version dimensions already in the tree, and what each one covers

| Dimension | Where | Value / posture today |
|---|---|---|
| `server` | `Cargo.toml:3` | `0.5.0`. Locked to the four SDK packages via `sdk/spec/check_versions.py`, enforced in CI (`.github/workflows/sdk.yml:88`) |
| `GET /health`'s `version` | `src/metrics.rs:2080-2084` | `{"status":"ok","version":"…"}` (ADR 0002 §10, #244) — a bare JSON body, not the `ApiResponse` envelope |
| `route`'s own `/health` | `src/route.rs:2475-2481` | the same field plus `{"router":true,"shards":N}` |
| CLI skew warning | `src/remote.rs:400-467` | one `/health` read, compares `major.minor` only, never blocks; a bare `"ok"` body reads as a pre-0.5 server |
| stable HTTP error code | `src/api.rs:100-183` | `ErrorCode`, 25 variants; its own doc comment already declares a rename "a breaking change… like a response-shape change" |
| batch format | `src/ingest.rs:113,118` | `BATCH_VERSION`/`GROUP_VERSION` = 1, checked for **equality** |
| image format | `src/context/image.rs:51,219` | `IMAGE_VERSION` = 6, checked as a **range** `1..=6` |
| communities artifact | `src/api/communities.rs:106` | `COMMUNITIES_FORMAT` = 1 |
| MCP protocol version | `src/mcp/protocol.rs:11,114-124` | `["2024-11-05","2025-03-26","2025-06-18"]`; the client's requested version is echoed back only if supported, otherwise the newest supported version is substituted |
| SDK method surface | `sdk/spec/surface.yaml` | one entry per logical operation, CI-checked against both SDKs' actual signatures (`check_surface.py`, `check-surface.ts`) — covers method names/args, not response shape |
| range-vs-equality precedent | ADR 0003 §10 | user-authored artifact ⇒ equality; taguru-written-and-reread ⇒ range acceptance with additive-only growth; independent stamps kept apart on purpose ("either shape can rev without dragging the other along," `src/ingest.rs:115-117`) |

No two of these are the same number today, and this ADR does not change that —
it names each dimension explicitly instead of letting "the version" mean
whichever one the reader has in mind (#220's own stated goal).

### 2.3 HTTP surface

`fn routes()` (`src/main.rs:744-900`) is the server-mode route table: 47
routes, including the four probes (`/health`, `/live`, `/metrics`,
`/protocol`). `src/oauth_http.rs:37-57` adds six OAuth/discovery paths
(`.well-known/oauth-protected-resource[/mcp]`,
`.well-known/oauth-authorization-server`, `/oauth/register`,
`/oauth/authorize`, `/oauth/token`), and `POST /mcp` dispatches into a clone
of the same router (`src/main.rs:528-546`). Router (sharding proxy) mode
serves a deliberately smaller surface — `src/route.rs:531-556`, 17 routes —
since it proxies rather than owns most context-scoped operations.

Every JSON success is wrapped in `ApiResponse<T> {result, status:"ok", time}`
(`src/api.rs:76-91`) via `api::ok()`. Five things are not wrapped: `/protocol`
(`text/markdown`), `/live` (bare text `ok`), `/metrics` (Prometheus text),
`/health` (bare JSON, not `{result, ...}`), and the three NDJSON streams
(context export, group export, `GET .../communities`).

Errors answer one shape, `ApiError` (`src/api.rs:398-432`): `{status:"error",
code, error, time}` plus four additive, presence-conditional fields —
`issues`, `integrity`, `durable_batches`, `retryable_after_correction` — used
only by rejected `add_associations`/`store_passages`/`import` calls. The
code → HTTP status mapping lives in exactly one place,
`ErrorCode::status()` (`src/api.rs:192-217`); the code is the source of truth
the status is derived from, not the reverse.

Thirteen distinct pagination/result envelopes exist across `src/api.rs` and
`src/api/*.rs` (`MatchPage`, `CrossMatchPage`, `ContextPage`, `GroupPage`,
`AliasExport`, `LabelPage`, `SourcePage`, `ExplorePage`, `ActivationPage`,
`PassagePage`, `CrossPassagePage`, `CommunityPage`, `DriftAudit`). Most carry
`{total, …}` so a client can detect truncation against `limit`.
**`PassagePage`, `CrossPassagePage`, and `CommunityPage` do not** — they carry
`{plan, hits}` (or `{…, plan, hits}`) with no `total` field at all. This ADR
records the inconsistency as a known, frozen fact of the current contract's
history — §4 classifies adding `total` to any of the three, should someone
choose to, as an ordinary additive and therefore compatible change; this ADR
neither requires nor forbids making it.

Every request body's `Deserialize` is lenient about unknown fields today — no
struct used as an HTTP request body carries `#[serde(deny_unknown_fields)]`.
The only three types that do are the wire cursors a client copies verbatim
from a previous page and sends back unmodified: `MatchCursor`
(`src/api.rs:1240-1241`), `ExploreCursor` (`src/api.rs:1332-1333`), and
`CrossMatchCursor` (`src/api/recall.rs:199-200`).

No response header carries a version today. There is no `Taguru-Version` or
similar; the only per-response headers set anywhere are `Retry-After`,
`WWW-Authenticate`, and `Content-Type` overrides for the non-enveloped bodies.

### 2.4 MCP surface

46 tools, one source of truth: the `tools` vec inside `tool_definitions()`
(`src/mcp/schema.rs:56`, vec starting at line 70). No tool declares an
`outputSchema`. On success, a tool's result is `{"content":[{"type":"text",
"text": <verbatim HTTP response body>}]}` (`src/mcp/protocol.rs:160-177`) —
MCP does no re-serialization of its own; it passes through whatever `api.rs`
already serialized. `structuredContent` appears only on a tool **error**,
and only when the failing HTTP body parsed as a JSON object.

Because of that pass-through, 45 of the 46 tools inherit the HTTP contract
exactly — a shape change to an HTTP response is the identical shape change
seen through MCP, with no separate MCP-side classification needed. The one
exception is `retrieve`, whose composed output — `{resolved, outline,
associations, activations, citations, passage_hits, search_plan}`
(`src/mcp/retrieve.rs:321-329`) — exists only as an ad-hoc `json!` value, with
no HTTP endpoint and no Rust struct backing it (its typed counterparts live
only in the SDKs).

stdio and remote MCP serve the identical tool set (both compile
`src/mcp.rs`), differing only in transport concerns (framing, cancellation,
result-size caps, deadline handling) — never in which tools exist or what
they return. JSON-RPC errors (`-32700`/`-32600`/`-32601`, classified in
`protocol.rs:53-95`) abort the call; every `tools/call` failure instead comes
back as a successful JSON-RPC result with `isError: true`, so a tool error is
never itself a wire-shape break to classify — its shape is the same on every
call.

### 2.5 SDK decoding — the empirical basis for §4's rule table

The two SDKs are the actual audience for "compatible"; this ADR's compat/
break table is calibrated against what they really do, not against an
abstract ideal.

| | Python (`sdk/python/src/taguru/_decode.py`) | TypeScript (`sdk/typescript/src/models.ts`, `transport.ts`) |
|---|---|---|
| Unknown response field | dropped — walks `dataclasses.fields(cls)`, not the input's keys (`_decode.py:48-50`) | untyped but present; no runtime check exists at all |
| Required field absent | `ValueError: missing required field` (`_decode.py:55`) | **undetected** — the field silently decodes to `undefined` under an `as T` cast (34 cast sites in `client.ts`) |
| Container shape change (array ↔ object) | `ValueError: expected an object for {cls}, got {type}` (`_decode.py:45`) — the exact 0.4.0 symptom | **undetected**, same `as T` cast |
| Unknown enum-like value | passes through — every enum-shaped field in `_models.py`/`_types.py` is typed plain `str`; zero `Literal`/`Enum` types exist | passes the runtime cast, but seven fields are typed as **closed** string-literal unions that a new value silently violates: `models.ts:292,306,334,361,388,417,773` |
| Runtime schema validation | none beyond the `{status, result}` envelope (`_shared.py:110-126`) | none beyond the same envelope check (`transport.ts:116-137`) — no zod, no io-ts |

Both SDKs discard `GET /health`'s `version` field entirely today —
`health()`/`health()` return `None`/`Promise<void>`. The TypeScript SDK has no
runtime version constant anywhere in `src/` (`package.json`'s version is
build-time only); `docs/troubleshooting.html` tells a TS user to run `npm ls
taguru` instead. Python's three strict-decode failures above raise a bare
`ValueError`, outside the `TaguruError` hierarchy a caller would normally
catch. No golden wire-shape fixture exists in the repository today; the
closest thing, `DIRECTORY_ROW`, is a literal hand-duplicated once in each
SDK's unit-test suite.

### 2.6 The contract of record

`src/llm-protocol.md` — served verbatim by `GET /protocol` and handed to
every MCP client as its `initialize` `instructions` (`protocol_text`,
`src/api/contexts.rs:243-249`) — is already the one place that documents the
request/response shape of every endpoint (its `## API` table) and states a
compatibility posture (`## Compatibility`, lines 504-519): no `/v1` prefix,
additive fields, absent-optional-means-omitted-not-null, and — today — "a
minor version bump is allowed to change a response's shape."

This ADR does not create a second inventory file. §2.3-2.5 above is a frozen
snapshot for the record of what was true when this ADR was written; the
document that stays current as the surface grows is `src/llm-protocol.md`,
and this ADR names it as such. When #305 (opt-in HTTP integration for #216)
adds routes, it edits `llm-protocol.md`, not this file.

## 3. Version dimensions

Eight dimensions, kept independently numbered — the same reasoning ADR 0003
§10 already applied to `BATCH_VERSION`/`GROUP_VERSION`/`IMAGE_VERSION`/
`COMMUNITIES_FORMAT`: different owners, different consumers, and forcing one
shared number means an unrelated file's bump invalidates every artifact that
never changed.

1. **`server`** — SemVer (`Cargo.toml`). Stays lockstep with the four SDK
   packages via the existing `check_versions.py`; this ADR does not touch
   that mechanism.
2. **`http_contract`** — new, `u64`, starts at `1`. Owns every enveloped and
   non-enveloped HTTP response/request/error shape in §2.3.
3. **`mcp_contract`** — new, `u64`, starts at `1`, stamped independently of
   `http_contract`. Because 45 of 46 tools are verbatim HTTP pass-through
   (§2.4), this dimension covers only what MCP itself owns: the 46-tool
   name/`inputSchema` table, `retrieve`'s composed output shape, the
   `isError`/`structuredContent` convention, and the JSON-RPC error-code
   vocabulary. A pure HTTP-shape change is classified under `http_contract`
   even though it is also visible over MCP — there is no double bump for the
   same change.
4. **`mcp_protocol`** — the MCP specification date Taguru negotiates
   (`SUPPORTED_PROTOCOL_VERSIONS`). Taguru does not own this; it only tracks
   which upstream dates it speaks.
5. **`batch_formats`** — `taguru_batch`/`taguru_group`, unchanged, equality
   check.
6. **`image_formats`** — `IMAGE_VERSION`, unchanged, range-acceptance check.
7. **`communities_formats`** — `COMMUNITIES_FORMAT`
   (`src/api/communities.rs:106`), unchanged, equality check (the same
   `taguru_communities` header-stamp precedent ADR 0003 §10 already applies
   to it) — kept apart from `http_contract` because it is a taguru-written,
   taguru-reread artifact stamp like `image_formats`, not a request/response
   shape.
8. **SDK supported contract range** — independent of SDK package version;
   #300's responsibility to declare and check.

Responsibility boundary: **HTTP is the sole owner of wire shape.** MCP is a
transport over that same shape for all but one tool; it never redefines a
shape HTTP already defines. SDKs own method names/arguments (`surface.yaml`)
and their own error-class hierarchy; they own no wire shape of their own —
they decode HTTP's.

## 4. Compatible / breaking change table

Grounded in §2.5's actual decoder behavior, not a hypothetical strict
decoder.

| Change | Classification | Basis |
|---|---|---|
| Add an optional response field | compatible | Python drops unknown keys; TS is unaffected by an untyped extra key |
| Remove or rename a response field | **breaking** | Python: `missing required field`; TS: silently `undefined` — worse, since nothing signals the break |
| Stop emitting a field that was previously always present | **breaking** | same as above |
| Change a value's container shape (array ↔ object) anywhere in a response | **breaking** | `_decode.py:45`; the literal 0.4.0 incident (§2.1) |
| Add an optional request field | compatible | no request body denies unknown fields today (§2.3) |
| Add a required request field | **breaking** | old clients never send it |
| Add `deny_unknown_fields` to a request body that lacks it today | **breaking** | every current body is lenient; adding the attribute rejects previously-valid requests |
| Add a new `ErrorCode` variant that keeps an existing status | compatible | both SDKs select their error class from **HTTP status** (`error_for_status`/`errorForStatus`), never from `code` |
| Rename or repurpose an `ErrorCode` variant | **breaking** | `src/api.rs:98-99`'s own doc comment already says so |
| Change which status an existing `code` maps to | **breaking** | changes which SDK error class a caller catches |
| Add a new value to an enum-like field | compatible in Python; **currently breaking in TypeScript** | Python has no closed enum types; TS's seven closed literal unions (§2.5) silently lie once a new value arrives — §5 states the prerequisite that removes this asymmetry |
| Add a new HTTP endpoint or MCP tool | compatible | additive |
| Remove an HTTP endpoint or MCP tool | **breaking** | |
| Add an optional property to an MCP `inputSchema` | compatible | |
| Add a property to an MCP tool's `required` list | **breaking** | old callers don't send it |
| Change a pagination envelope's shape (e.g. `{total,…}` ↔ `{plan,hits}`) | **breaking** | |
| Change a field's meaning, unit, or ordering guarantee without changing its type | **breaking**, and the worst kind — nothing in either SDK detects it | |
| Change `retrieve`'s composed output keys | **breaking**, under `mcp_contract` | `retrieve` has no HTTP counterpart to inherit from |

`PassagePage`/`CrossPassagePage`/`CommunityPage` lacking `total` (§2.3) is
recorded here as a known, frozen inconsistency of the current
`http_contract: 1` — adding `total` to them now would itself be an additive,
compatible change; the frozen fact is only that no version of the contract to
date has had it.

## 5. Forward-compatibility policy

Server-side: within one `http_contract` (or `mcp_contract`) version, only
additive changes ship. Optional fields are omitted via `skip_serializing_if`,
never emitted as `null` — the pattern already used throughout `src/api.rs`
and `src/api/*.rs`; this ADR names it as the binding house style rather than
an incidental one.

Client-side, the contract requires: ignore unknown response fields, treat
every enum-like field as open (unrecognized values must not crash decoding),
and never depend on field order.

**TypeScript's seven closed string-literal unions (§2.5) currently violate
the "enum-like fields are open" requirement.** §4's "add an enum value is
compatible" row holds for Python today but not for TypeScript until those
unions are widened (to `| (string & {})` or equivalent) — this is filed as a
precondition for #300, not a new issue (§9).

Request-side leniency is preserved — no new `deny_unknown_fields` is added to
any request body. The three existing exceptions (`MatchCursor`,
`ExploreCursor`, `CrossMatchCursor`) stay exceptions: they are values the
server itself issued and the client returns unmodified, so an unrecognized
key there signals real client-side corruption, not a version gap — this is
also why the TypeScript SDK's `matchCursor()`/`crossMatchCursor()` narrowing
helpers exist (`models.ts:64,83`).

## 6. Contract version discovery — a partial supersession of ADR 0002 §10

ADR 0002 §10 decided, deliberately: no separate `/version` endpoint, no
capability-matrix response — "this repository's compatibility doctrine…
already makes the version number the capability descriptor; a matrix would
duplicate what the version already states." That reasoning held when there
was exactly one version number in play. #220's entire premise — separating
`server` from `http_contract` from `mcp_contract` (§3) — removes the
premise ADR 0002 §10 reasoned from: a single version number can no longer
describe all of what a caller needs to know. This ADR supersedes §10's
specific "no `/version` endpoint, no matrix" clause only. It does **not**
touch §10's addition of `version` to `GET /health`'s success body — that
field stays; removing it would itself be a breaking change, and
`src/remote.rs`'s CLI skew check already depends on it.

Decision: add `GET /version`.

- Exempt from auth like the existing probes (`PROBE_EXEMPT`,
  `src/auth.rs:43`), and — unlike `/health` — answers `200` even while the
  write path is degraded (`/health` returns `503` for `unhealthy` or
  `maintenance`, which makes it unusable as the base a compatibility check
  runs from).
- Bare JSON body, not the `ApiResponse` envelope (matching `/health`'s own
  precedent):
  ```json
  {
    "server": "0.6.0",
    "http_contract": {"current": 1, "supported": [1]},
    "mcp_contract": {"current": 1, "supported": [1]},
    "mcp_protocol": {"supported": ["2024-11-05", "2025-03-26", "2025-06-18"]},
    "batch_formats": [1],
    "image_formats": [1, 2, 3, 4, 5, 6],
    "communities_formats": [1]
  }
  ```
  Every dimension from §3 appears here except `server` (already the
  top-level field) and the SDK supported range (that one is declared inside
  each SDK, not by the server — §3-8). `mcp_protocol` restates
  `SUPPORTED_PROTOCOL_VERSIONS` (`src/mcp/protocol.rs:11`) so a caller can
  read it over plain HTTP, before ever opening an MCP connection to learn it
  from `initialize`.
- `supported` ships as an array from day one, even though `[1]` is a
  single-element array today: it gives #300's SDK-range check something to
  intersect against, and lets a future dual-serving window (not decided
  here) be expressed without a shape change to `/version` itself.
- No `/v1` URL prefix, no `Accept`-parameter negotiation, and no
  request-side version header — `llm-protocol.md:506-509`'s existing "one
  server serves one protocol version, its own" stays as-is. `GET /version`
  is discovery only, not negotiation.
- No global `Taguru-*` response header is added. Today zero custom headers
  exist on ordinary responses; one endpoint stating these facts once is
  simpler than every response carrying them, for no caller that doesn't
  already ask.
- Router mode answers its own `GET /version` the same way `route.rs`
  already answers its own `/health` (`src/route.rs:2475-2481`) — under the
  same "shards are homogeneous" assumption the existing `/health` and
  `/protocol` proxying already make. No shard-fanout proxy needed, unlike
  `/protocol`.
- MCP stays in sync for free: `protocol_trailer` (`src/api/contexts.rs:257`)
  already injects live server facts into both `GET /protocol`'s body and
  every MCP `initialize`'s `instructions` (the stdio bridge fetches this
  from the server at connect time, falling back to
  `include_str!("../llm-protocol.md")` only if that fails —
  `taguru-mcp.rs:36`). Folding the same JSON block into that trailer means
  no new MCP tool, and no `outputSchema`, is needed for an MCP client to
  learn the contract versions.

## 7. Deprecation, removal, and migration

Within one contract version, nothing is deleted and no field's meaning
changes. A field being retired is documented in `llm-protocol.md` and in
CHANGELOG's `Deprecated` section, and keeps being emitted. Removal happens
only alongside a `http_contract` (or `mcp_contract`) major bump, landing in
the same PR as: the bump itself, a CHANGELOG `Changed` entry, and a migration
note a caller can act on.

This raises the pre-1.0 minimum guarantee, **effective immediately on this
ADR's acceptance, not deferred to #300.** Today's stated posture
(`llm-protocol.md:512-513`) is "pre-1.0, shapes may also change between minor
versions." This ADR tightens that to: **within one contract version, nothing
breaks** — a minor `server` bump may still add things, but a break requires
the matching contract-version bump described above, regardless of whether
`server`'s own bump is major or minor. The guarantee binds what future PRs
are allowed to ship; it does not depend on `GET /version` existing yet. Until
#300's implementation PR edits `llm-protocol.md` (§10), that file's older,
weaker sentence is stale prose describing a posture this ADR has already
superseded — this ADR is authoritative over that unedited text for any PR
landing in between, not the reverse.

Support-window commitments (how many old contract versions stay served, for
how long) are explicitly not decided here — that is #220's v0.7.0+ scope,
which owns the cross-version compatibility matrix this would require testing
against.

## 8. Constraints this hands to #216 / #302

#302 designs the evidence-assembly API's candidate/budget/dedup/reranker
semantics; this ADR does not reach into that. It does fix the wire-contract
rules #302's design must satisfy, since #216 ships opt-in and therefore adds
without changing anything existing (`http_contract` stays `1`):

- Responses use the existing `ApiResponse<T>` envelope — no new top-level
  envelope shape.
- New failure modes are expressed as additional `ErrorCode` variants inside
  an existing HTTP status class (§4), never a new status-to-meaning mapping.
- Any selection-trace or provenance field with a closed set of values is
  designed open from the start (plain string on the wire, and the
  TypeScript model must not use a closed literal union — §5's rule applies
  to new fields, not only retrofits).
- If the response needs pagination, it reuses one of the thirteen existing
  envelope shapes (§2.3) rather than minting a fourteenth.
- The MCP surface for this feature is an ordinary routed tool, inheriting
  `http_contract` via pass-through (§2.4) — it does not grow a second
  `retrieve`-style ad-hoc composed shape unless a real composition need
  forces it.

## 9. Consequences and follow-up

Folded into #300's own scope, not filed as separate issues, because #300
cannot deliver a working SDK compatibility check without them:

1. Widen TypeScript's seven closed string-literal unions
   (`models.ts:292,306,334,361,388,417,773`) to open unions — the
   prerequisite §5 states for "add an enum value" to be compatible in
   TypeScript the way it already is in Python.
2. Add a TypeScript runtime version constant — none exists today, so a
   TS-side compatibility check has nothing local to compare `GET /version`
   against.
3. Bring Python's three strict-decode failures (`_decode.py:45,55`) into the
   `TaguruError` hierarchy — today they're bare `ValueError`, so the exact
   symptom `docs/troubleshooting.html` names as the canonical skew signal
   escapes an `except TaguruError` handler.

Filed separately, lower priority: adding the missing `total` field to
`PassagePage`/`CrossPassagePage`/`CommunityPage` (§2.3) — an ordinary
additive, compatible change per §4, just not one this ADR requires.

Not filed, and why: generating an OpenAPI document (`sdk/README.md:31-37`
already states "there is deliberately no OpenAPI spec — the contract is the
server's own protocol document"; #301's golden fixtures are this tree's
answer to machine-readable pinning, not a schema-generation pivot); a
support-window policy (§7, explicitly #220's later scope); router/shard/
replica version-skew enforcement (also #220's later scope, needs the
cross-version matrix this ADR does not build).

## 10. Documentation impact

`llm-protocol.md`'s `## Compatibility` section (lines 504-519) needs two
edits reflecting §6 and §7: documenting `GET /version` and restating the
minimum guarantee as "unchanged within one contract version" rather than
"may change between minor versions." Both edits land in #300's implementation
PR, not this one — writing them before `GET /version` exists would describe
an endpoint that isn't there yet, and §7 already establishes that this ADR,
not the stale sentence still on disk, governs in the interim. Likewise, `docs/troubleshooting.html`'s
`#compatibility` section linking out to a contract-version discovery
explanation (one of #220's acceptance criteria, alongside #193) is #300's
edit to make once there is something concrete to link to.
