# 0002. Remote server access from the taguru CLI

- **Status**: Accepted
- **Date**: 2026-07-25
- **Issue**: #190
- **Related**: ADR 0001
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

Whether, and how, the local `taguru` binary should be able to operate a
Taguru server that is not the local data directory — Docker, a separate
host, a VM, a cloud instance — and which of its subcommands that answer
applies to. Out of scope: any change to the SDKs or MCP surface, OAuth,
`route`'s internal proxying, confirmation UX for destructive operations
(nothing in this ADR's remote set is destructive — see §7), and the
CLI's broader flag/environment-variable consistency, which ADR 0001 §1
already flagged as this issue's territory and which this ADR further
narrows and defers to its own follow-up (§12.1, #248).

## 2. Context

### 2.1 The de facto rule already in the code

Three subcommands already reach a running server over HTTP:
`health`, `calibrate`, and `communities`. All three resolve their target
the same way, and the code says so explicitly — `default_base_url()`'s
doc comment (src/cli.rs:479-480) reads: *"`calibrate` resolves its
target the same way, for the same reason: one rule for 'which server
does a CLI verb mean'."* The rule (src/cli.rs:481-510): an explicit
positional URL argument, else `TAGURU_ADDR` (default
`127.0.0.1:8248`) with an unspecified bind address (`0.0.0.0`, `[::]`)
read as its own loopback — because the CLI process and the server it
probes share a network namespace exactly when the server bound
`0.0.0.0` for something else (containers, `serve --config`) to reach it
from. Port `0` (OS-assigned) is rejected with a message pointing at the
same fix: pass the real URL.

`calibrate` (src/calibrate.rs:664-761) and `communities`
(src/communities.rs:761-875) each carry their own hand-written `ureq`
client — a `struct Api` with `get`/`post`, a `finish()` that unwraps the
server's `{"result": …}` envelope or forwards its error text verbatim,
and a `bearer_token()` that reads `TAGURU_API_TOKEN`, falling back to
the first `name:token` entry of `TAGURU_API_TOKENS` — *the same
variables the server itself reads*, documented as such in
`communities`'s own `--help` text. The two implementations are
near-identical; nothing currently shares them.

### 2.2 Command taxonomy audit

Every `taguru` subcommand (src/cli.rs:290-321 `dispatch()`), classified
by what it touches:

| Command | Touches | Class |
|---|---|---|
| `serve` (default), `route` | binds a port, opens/proxies the data directory | server-mode |
| `version` | prints `CARGO_PKG_VERSION` | local, no target |
| `health`, `calibrate`, `communities` | HTTP only, no data directory | **remote-only** (already) |
| `import`, `export`, `compact` | `TAGURU_DATA_DIR` via `BootConfig::from_env().boot(...)`, exclusive directory lock | **offline, dual-mode candidate** |
| `inspect` | an explicit `PATH` argument, no data-directory boot, no HTTP | offline-only |
| `restore` | an object-storage bucket (s3/gs/az/file) to a local `--out` directory | offline-only |
| `estimate` | pure in-memory synthesis | offline-only |
| `extract` | an external OpenAI-compatible chat endpoint (`TAGURU_EXTRACT_URL`), never the taguru server or its data directory | offline-only |

`import`/`export`/`compact` are the only offline commands with a
data-directory lock that a running server *also* holds — and, as §2.3
shows, the only ones with an equivalent HTTP surface already exposing
the same operation on a running server.

### 2.3 Server API coverage for the offline three

- **import**: `POST /import` (src/api/import.rs:371) accepts the same
  NDJSON batch stream `taguru import` applies locally, with `?dry_run=true`
  giving the same preview `ingest.rs` supports. Its response already
  reports `durable_batches` and a `stream_integrity` verdict per line
  (src/api/import.rs:139-163) — the primitive a remote client needs to
  report partial application honestly.
- **export**: `GET /contexts/{name}/export` and `GET /groups/{name}/export`
  (src/api/import.rs:680, 726) each stream one context or group in the
  same NDJSON shape `taguru export` writes. There is no endpoint that
  exports every context and group in one call — `taguru export` alone
  produces that, by listing the data directory locally.
- **compact**: `POST /contexts/{name}/compact` (behind the heavy-ops
  limiter, src/main.rs:742-748) and `POST /maintenance/compact`
  (src/api/contexts.rs:159) cover both the per-context and the
  sweep form `taguru compact` runs offline.

`inspect` and `restore` have no server-side equivalent at all — the
former exists to validate a backup *without* a server, the latter
restores a *new* data directory from an object-storage bucket, never
from a running server's HTTP surface.

### 2.4 Auth, roles, and the replica refusal

`auth::required_role` (src/auth.rs:703-755) maps every route to
`Role::Read`, `Role::Write`, or `Role::Admin`, and its match falls
through to `Role::Admin` for anything unlisted (src/auth.rs:753, the
comment: *"Operator verbs — and everything unclassified"*) — a
deny-by-default posture, confirmed by its own test asserting
`required_role(&Method::POST, "/import") == Role::Admin`
(src/auth.rs:1477). `GET .../export` is `Role::Read`
(src/auth.rs:715, 709); `.../compact` and `/maintenance/compact` fall
into the `Role::Admin` catch-all alongside `/import`. `/health`,
`/live`, `/metrics` are the only credential-free routes
(`PROBE_EXEMPT`, src/auth.rs:43).

A read replica (`serve --replica`) refuses every non-`Role::Read`
request through `replica_gate` (src/api.rs:518-542) with a
`read_only_replica` error whose text names the operator's own
`TAGURU_WRITER_URL`, if configured, plus the bucket's fence holder
(src/replica.rs:93-113) — a named target, not an HTTP redirect. Nothing
in this repository issues an HTTP redirect anywhere in the write path.

## 3. Options considered

### A. Make `taguru` a general-purpose remote client (global `--url`)

Every verb — `recall`, `contexts list`, `import`, `export`, … — gains a
uniform `--url` and speaks HTTP.

- Evidence: §2.2's taxonomy — half the offline commands
  (`inspect`, `restore`, `estimate`, `extract`) have no server-side
  operation to point at, or point somewhere that is deliberately never
  the taguru server (`extract`'s LLM endpoint). Making all of them
  accept `--url` means either a per-verb reject matrix nobody asked for,
  or new server APIs invented solely to give a CLI flag something to
  call — `inspect`'s whole reason to exist is validating a backup
  *without* a server, and `restore`'s is rebuilding a data directory
  *before* a server exists to ask.
- Verdict: **Rejected.** A uniform surface promises more than the
  command set can honestly deliver, and completing that promise means
  growing the HTTP API to serve the CLI rather than the reverse.

### B. A dedicated `remote` subcommand namespace or separate binary

`taguru remote import`/`taguru remote export`/… (or a `taguructl`
binary) alongside the untouched offline verbs.

- Evidence: the three existing remote verbs — `health`, `calibrate`,
  `communities` — already live at the top level, not under any
  namespace, and are wired into deployment (`docker-compose.yml`,
  Kubernetes `HEALTHCHECK`) that way. A new namespace would put remote
  access in two incompatible places in the same binary. `dispatch()`
  (src/cli.rs:290-321) is a flat match over first-argument strings —
  hand-rolled on purpose (cli.rs:1-4) — and nesting a sub-parser under
  `remote` is exactly the structure that comment set out not to grow.
  A separate binary duplicates argument parsing, the `--config`
  loader, and `bearer_token()` for a distribution-size concern that
  does not exist here: the single `taguru` binary already ships
  `serve` and every offline tool together.
- Verdict: **Rejected.** Inconsistent with the precedent this repo
  already set, and adds a second CLI surface to document and test for
  no capability a flag does not already give.

### C. Add no CLI surface; treat HTTP/MCP as the sole remote entry point

Point operators at `curl`/the HTTP API/generated clients instead of
extending the CLI.

- Evidence: §2.3 — a full backup requires enumerating `GET /contexts`
  and `GET /groups` and looping per-item export calls by hand; a bulk
  load past `TAGURU_MAX_BODY_BYTES` (8 MiB default,
  src/env.rs:168,188-201) needs the caller to chunk the NDJSON stream
  and interpret `durable_batches`/`stream_integrity` on partial
  failure. Both are exactly the kind of bookkeeping this repository
  already automates for the *local* path (`taguru export` lists the
  data directory itself; `taguru import` already runs the same batches
  through the same parser). Asking every operator to hand-roll that in
  a shell script the day they move a server off `localhost` reproduces
  work this binary already does.
- Verdict: **Rejected as the sole answer**, but its discipline is kept:
  §4's set stays deliberately short, and every agent-facing verb
  (`recall`, `query`, `explore`, context/group CRUD, …) stays HTTP/MCP
  territory, never CLI. Nothing here proposes an OpenAPI spec or a
  generated client; if that gap is felt later, it is a separate issue.

### D. Hybrid — extend only the verbs with an existing server equivalent

Add `--url` to `import`, `export`, and `compact` only; leave every other
command exactly as it is.

- Evidence: §2.2's taxonomy makes the boundary fall out of the code
  already there, not a new design: these three are the only offline
  commands with (a) a data-directory lock that collides with a running
  server and (b) a documented HTTP endpoint doing the same thing.
  `health`/`calibrate`/`communities` already answer "yes" to remote
  access; `inspect`/`restore`/`estimate`/`extract` structurally cannot
  or should not.
- Verdict: **Adopted.**

## 4. Decision

**Option D. `import`, `export`, and `compact` each gain an `--url URL`
flag that, when given, sends the operation to a running server's HTTP
API instead of touching the local data directory. No other subcommand
changes. The three existing remote verbs (`health`, `calibrate`,
`communities`) are unchanged by this ADR — extending them to accept
`--url` as an alias for their positional-URL argument is deferred to
the follow-up in §12.1 (#248), which owns the CLI's flag/environment
consistency generally.**

The organizing principle, stated once because every following section
relies on it:

> **A verb becomes remote only when the command line names a server.
> No environment variable ever changes which mode a verb runs in, or
> which server it targets, on its own.**

This is deliberately narrower than "whichever server `TAGURU_ADDR` or
`TAGURU_URL` happens to point at right now" — a value that a shell
session accumulates across unrelated work. `import`/`export`/`compact`
without `--url` behave exactly as they do today: zero behavior change
for the default invocation, and no way for a stray environment variable
to turn a local dry run into a write against production, or a forgotten
flag in CI to silently write into the runner's own scratch directory
instead of the intended server.

## 5. The remote rule for the three dual-mode verbs

- `--url URL` (a value is required; `--url` alone is a usage error, exit
  2) is the **only** way to make `import`, `export`, or `compact`
  remote. No positional URL argument is added to these three — `import`
  already takes `FILE|DIR...` positionally and `compact` already takes
  `[CONTEXT...]`; a second positional meaning would be ambiguous.
- Without `--url`, behavior is pixel-for-pixel what it is today:
  `BootConfig::from_env().boot(...)` against `TAGURU_DATA_DIR`, subject
  to the same exclusive directory lock.
- No environment-variable fallback for `--url` on these three verbs.
  `TAGURU_URL` (today read only by the `taguru-mcp` bridge,
  src/bin/taguru-mcp.rs:39) is not read by `taguru` itself as a result
  of this ADR — see §11 for why, and where that question is deferred.
  The CI idiom for repeated remote calls is ordinary shell expansion —
  `taguru import --url "$TAGURU_URL" batch.jsonl` — which keeps the
  target visible in the invocation instead of ambient in the
  environment.
- A flag that only makes sense offline, combined with `--url`, is a
  usage error rather than a silent no-op — e.g. `import --no-embed
  --url …` (the server's own embedding configuration decides once the
  request lands there, so the local flag has nothing to control).
  Each follow-up issue in §12.1 owns its verb's exact incompatible-flag
  list; this ADR fixes only the principle.
- Every remote, mutating invocation prints its target to stderr before
  sending anything (`import → https://host/…`) — the one line that lets
  an operator notice a stale `--url` value pasted from history before
  the request lands, without adding a confirmation prompt (§7).
- `import`'s existing directory-lock refusal (when a local server is
  already running against the same data directory) gains one added
  sentence pointing at the way out: importing into a *running* server
  is `taguru import --url http://127.0.0.1:8248 FILE...`, not a second
  offline process racing the first.
- A replica's `read_only_replica` refusal (§2.4) is displayed verbatim
  and the command exits 1. The CLI does not parse
  `TAGURU_WRITER_URL` out of that message and retry there automatically
  — the fence is the server's statement of the current writer, and
  auto-following it on a write path client-side is exactly the kind of
  redirect-and-retry behavior §8 rules out generally.

## 6. Command boundary matrix

| Verb | Remote? | Mechanism |
|---|---|---|
| `serve`, `route` | n/a | these *are* the server |
| `version` | n/a | prints the local binary's own version; a running server's version is read from `/health` (§9) once the follow-up in #244 lands |
| `health`, `calibrate`, `communities` | already remote-only | unchanged: positional URL, else `default_base_url()` |
| `import` | **dual-mode** | `--url` sends to `POST /import`; absent, local `TAGURU_DATA_DIR` |
| `export` | **dual-mode** | `--url` enumerates and calls the per-context/group export endpoints; absent, local `TAGURU_DATA_DIR` |
| `compact` | **dual-mode** | `--url` calls per-context and/or `/maintenance/compact`; absent, local `TAGURU_DATA_DIR` |
| `inspect` | no | validates a backup *without* a server by design; no server endpoint performs the same check |
| `restore` | no | rebuilds a data directory from object storage *before* a server exists to ask; disaster recovery must not depend on the server it is recovering |
| `estimate` | no | pure in-memory sizing calculation |
| `extract` | no | its remote endpoint is an LLM provider, never the taguru server |

## 7. Authentication and why no confirmation gate is added

Authentication for the three dual-mode verbs reuses `bearer_token()`'s
existing rule unchanged: `TAGURU_API_TOKEN`, else the first
`name:token` entry of `TAGURU_API_TOKENS` — the same resolution
`calibrate` and `communities` already document as "the same variables
the server reads." Role enforcement stays entirely server-side
(§2.4's table); a 401/403 is displayed with the server's own message
and the command exits 1. No CLI-side role pre-check is added — the
server is always the authority on what a given key may do, and a
client-side approximation would drift from it.

`--token` is deliberately **not** added as a flag: a token passed on
the command line is readable from `ps` and shell history for the
lifetime of the terminal, a leak the existing `TAGURU_API_TOKEN`
environment variable already avoids. This ADR records that omission as
intentional so a later PR does not reintroduce it without revisiting
this reasoning. A URL carrying `userinfo`
(`https://user:token@host/...`) is rejected for the same reason.

**No `--yes`/confirmation prompt is added for `import` or `compact`.**
Three things already gate an accidental remote write, and stacking a
fourth was judged not worth the inconsistency of introducing this
codebase's first interactive TTY prompt (nothing in `taguru` today
blocks on stdin) for a CI-first tool:

1. `--url` itself is the explicit act of naming a target — there is no
   implicit remote mode to fall into (§4).
2. `import --url … --dry-run` (`?dry_run=true` on the wire,
   src/api/import.rs:361) previews exactly what would land, using the
   same preview machinery the local path already has.
3. The server's own `Role::Admin` requirement (§2.4) is the actual
   authorization boundary; a client-side prompt would not add a
   security property, only friction.

Nothing in the dual-mode set is a delete: `export` is read-only,
`compact` and `import` are additive/idempotent in the sense §8 states.
A future issue that extends remote access to genuinely destructive
verbs (context/group deletion) inherits the obligation to design that
verb's own confirmation UX — explicitly out of scope here (§1).

## 8. Failure semantics: timeout, retry, idempotency

- **Timeout**: the client budget follows `calibrate`'s existing
  precedent (`Duration::from_secs(35)`, src/calibrate.rs:673-677,
  documented as "above the server's default 30s request budget, so a
  server-side timeout answers as itself... instead of a client-side
  cut") — comfortably above the server's own
  `TAGURU_REQUEST_TIMEOUT_SECS` default of 30 (src/config.rs:71,
  src/env.rs:170-183). No new environment variable is introduced for
  this; a server operator who raises `TAGURU_REQUEST_TIMEOUT_SECS`
  raises it for every client uniformly, CLI included, once the
  follow-up implementation reads it the same way `calibrate` does.
- **No automatic retry, ever, on any of the three verbs.** A dropped
  connection after a request was sent leaves the caller unable to know
  whether the server applied it, and `import` in particular cannot
  safely be replayed blind (§9). Every failure is reported with the
  server's own error text (or the transport error) and the command
  exits non-zero; retrying is the operator's or CI's decision, informed
  by `--dry-run` re-verification where that applies.
- **No automatic HTTP redirect following.** Nothing in this server's
  write path issues a redirect (§2.4) — a 3xx would only come from an
  intermediary the operator placed (a reverse proxy), and following one
  silently on a write risks resending a POST body somewhere the
  operator did not expect, or dropping the `Authorization` header across
  a scheme change. `Retry-After` (sent on `503`/`429` shedding,
  src/limits.rs:78-101) is displayed, not acted on automatically.
- **Idempotency, stated per verb**: `export` is a pure read, safely
  repeatable. `compact` is safe to re-run — compaction of an
  already-compact context is a no-op cost, not a correctness risk.
  `import` has no idempotency guarantee — replaying an already-applied
  batch may create duplicate associations exactly as a second local
  `taguru import` of the same file would. On a mid-transfer disconnect,
  the CLI's message states the ambiguity plainly ("connection lost —
  the server may have applied some batches; re-run `--dry-run` before
  retrying") rather than implying either "nothing happened" or
  "everything happened."

## 9. Payload limits: the 8 MiB body cap and the missing bulk export

- **Remote `import`**: `POST /import`'s body limit defaults to 8 MiB
  (`DEFAULT_MAX_BODY_BYTES`, src/env.rs:168) and answers 413 above it.
  The remote client splits its NDJSON input on line boundaries into
  chunks under the cap (starting at 4 MiB, half the default, to leave
  headroom for a server configured lower) and issues one `POST /import`
  per chunk; a 413 on a chunk still oversized only from the cap being
  configured lower than assumed halves that chunk and resends — the
  server rejects an oversized body *before* applying anything
  (src/api/import.rs:371), so this adaptation is safe. This changes
  nothing about atomicity: batch application is already per-batch, not
  whole-file, on the local path too (`stream_integrity`,
  src/api/import.rs:139-163) — splitting the request stream does not
  introduce a transactionality guarantee that was never there.
  Server-side chunked/streaming ingestion (removing the cap entirely)
  is explicitly **not** pursued here — client-side splitting solves the
  case this ADR scopes without touching the server's request-body
  handling, and nothing in the audited evidence shows the cap binding
  outside bulk migration, which the client-side split already covers.
- **Remote `export`**: there is no endpoint that exports every context
  and group in a single call (§2.3). The remote client performs `GET
  /contexts` and `GET /groups` and calls each item's export endpoint in
  turn, writing each response to `--out` the same way the local path
  lays out its directory. This is **not** a point-in-time snapshot
  across contexts — each context/group's own export is internally
  consistent, but two contexts exported seconds apart may not reflect
  the same instant. The CLI's output states this limitation; operators
  needing a true point-in-time snapshot across a whole server already
  have one: the replication bucket plus `taguru restore`, which this
  ADR does not touch. A true bulk-export endpoint is deferred (§12.1,
  not currently filed — see §12.1's note on what is deliberately not
  filed).

## 10. Version discovery

`GET /health` currently returns bare `"ok"` text (200) or an `ApiError`
JSON body (503) — no version anywhere in either (src/metrics.rs:2054-2083).
There is no `/version` endpoint; the only places a server states its own
version today are `/metrics`'s `taguru_build_info` gauge and MCP's
`initialize` response `serverInfo.version`. `taguru health` itself only
checks the status code and prints whatever body came back
(src/cli.rs:449-465) — it does not require the literal string `"ok"` —
so changing `/health`'s success body to a small JSON object carrying a
`version` field is compatible with the one consumer that reads it today.

The concrete failure this closes: `auth::required_role`'s catch-all
(§2.4) means a route a newer CLI knows about but an older server does
not answers 401/403 (unclassified → `Role::Admin`, then the key's
actual role fails that bar), not 404. An operator debugging that sees
"a permissions problem" and starts adjusting scopes, when the real
issue is a version skew — the wrong diagnosis for the actual fault.

**Decision**: add a `version` field to `/health`'s success body (issue
B, §12.1) — no separate `/version` endpoint, no capability-matrix
response. This repository's compatibility doctrine (docs/troubleshooting.html
`#compatibility`: one release, one version number across every layer;
pre-1.0, minor bumps may break) already makes the version number the
capability descriptor — a matrix would duplicate what the version
already states. The three dual-mode verbs read this field once per
invocation once #244 lands and print a one-line stderr warning on a
minor mismatch; **the warning never blocks the operation** — a replica
mid-rollout legitimately runs a different minor than its writer for the
rollout's duration, and blocking would make the CLI unusable during
exactly the window an operator most needs it.

## 11. Configuration ruling: `TAGURU_URL` and the wider flag/env audit

`TAGURU_URL` is read today only by the `taguru-mcp` stdio bridge
(src/bin/taguru-mcp.rs:39) and is not in `KNOWN_KEYS`
(src/config.rs:51-106) — the 54-entry typo-lint list for the server's
own `--config` file. This ADR does **not** add `TAGURU_URL` support to
`taguru` itself, for two reasons that are about scope discipline more
than any objection to the variable itself: first, §4's principle (an
env var never silently retargets a verb) argues against adding *any*
new fallback source for these three verbs' `--url`, not just this
particular name; second, `KNOWN_KEYS` is a lint over the *server's*
configuration file, and admitting a client-side-only variable into it
would misrepresent what the list means.

Whether `TAGURU_URL` (or an equivalent) should ever back the three
*existing* remote verbs, whether `--url` should become an alias for
their positional argument, and the broader question this issue's own
body raised — flag/environment-variable consistency across the whole
CLI, including possible gaps and overlaps this audit was not scoped to
find — are real, and are deliberately **deferred, not decided, by this
ADR**. §12.1's #248 owns that audit as its own piece of work, so it
gets the scrutiny a cross-cutting consistency pass deserves rather than
being decided as a side effect of this ADR's narrower question.

## 12. Consequences

### 12.1 Follow-up issues

| Issue | Title | Depends on | Implements |
|---|---|---|---|
| [#243](https://github.com/t0k0sh1/taguru/issues/243) | remote: shared HTTP client module — extract calibrate's/communities' `Api`/`finish`/`bearer_token` into one `src/remote.rs` | none | §2.1, §7 |
| [#244](https://github.com/t0k0sh1/taguru/issues/244) | api: add a `version` field to `GET /health`'s success body; CLI minor-skew warning | none | §10 |
| [#245](https://github.com/t0k0sh1/taguru/issues/245) | export --url: remote export via context/group enumeration | #243 | §6, §9 |
| [#246](https://github.com/t0k0sh1/taguru/issues/246) | compact --url: remote per-context and maintenance compaction | #243 | §6 |
| [#247](https://github.com/t0k0sh1/taguru/issues/247) | import --url: chunked remote import with dry-run preview and 413 adaptation | #243 (#244 optional) | §6, §9 |
| [#248](https://github.com/t0k0sh1/taguru/issues/248) | cli: audit and reconcile flag/environment-variable consistency across the whole CLI (existing remote verbs' `--url` alias, `TAGURU_URL`'s scope, `KNOWN_KEYS`, `--config` coverage, other gaps) | none | §11 |

Explicitly and deliberately **not filed** as follow-ups, with the
reason recorded here so a future PR does not reopen these without first
reading why they were set aside: server-side streaming/chunked
`/import` (§9 — client-side splitting already covers the scoped case);
a bulk cross-context/group export endpoint (§9 — the replication+restore
path already serves the point-in-time need); automatic retry or an
import idempotency key (§8 — safety over convenience for a write path);
an interactive confirmation prompt (§7); `--token` (§7); a machine-readable
capability matrix beyond the version field (§10); remote `inspect` or
`restore` (§6 — structurally against what those commands are for).

### 12.2 Migration and API compatibility

- `import`, `export`, `compact` without `--url`: **zero behavior
  change.** This ADR adds a flag; it does not alter a single default.
- `health`, `calibrate`, `communities`: unchanged by this ADR outright
  (§4) — their positional-URL behavior, deployment usage
  (`HEALTHCHECK`), and `--help` text stay exactly as they are until
  #248 revisits them on its own terms.
- `/health`'s success body changing from bare `"ok"` to a small JSON
  object (#244) is compatible with `taguru health`'s own parsing
  (§10) but is a wire-shape change for anyone else scripted against the
  literal string; #244's implementation notes this in its own
  changelog entry, per this repository's changelog discipline.
- No existing environment variable's meaning changes. `TAGURU_URL`
  stays exactly what it is today (the MCP bridge's target) and gains no
  new reader as a result of this ADR.

### 12.3 Documentation impact

- docs/troubleshooting.html's `#compatibility` section is updated in
  this ADR's PR to mention the forthcoming `/health` version field
  (#244) alongside its existing `taguru version`/`taguru_build_info`/
  MCP `serverInfo.version` guidance, so the section does not go stale
  the moment #244 lands.
- README's remote-access material (the `taguru-mcp`/`TAGURU_URL`,
  `POST /mcp`, and `POST /import` sections) is checked against this
  ADR's decision in the same PR; any description that overstates or
  understates what today's CLI can do is corrected there.
- The `--url "$TAGURU_URL"` CI idiom (§5) is documented by whichever of
  issues C/D/E lands first, alongside that verb's own `--help` text —
  not written speculatively here before the flag exists.

## Appendix: requirement traceability

| Issue #190 completion criterion | Section |
|---|---|
| Chosen option recorded, with reasons the others were not chosen | §3, §4 |
| Local/remote boundary defined per command | §2.2, §6 |
| URL, profile, environment variable, and credential configuration and precedence defined | §5, §7, §11 |
| Policy defined for destructive operations, streaming, timeout/retry, and version skew | §7 (no destructive verb in scope), §8, §9, §10 |
| Minimal implementation scope decided, with follow-ups split out | §12.1 |
| If no CLI is added: HTTP/MCP guidance and missing docs/APIs named | N/A — a minimal CLI extension is adopted (§4); §2.3 and §9 name the missing bulk-export and streaming-import server APIs regardless, since remote `export`/`import` route around their absence rather than closing it |
