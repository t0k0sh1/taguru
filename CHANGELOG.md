# Changelog

Notable changes to taguru. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [SemVer](https://semver.org/) (pre-1.0: minor bumps may break).
Entries that change an on-disk format or a response shape say so.

## [Unreleased]

### Added
- Context groups: `/groups` bundles contexts (many-to-many) and may
  nest child groups — a shallow DAG, at most 3 groups tall, cycles
  refused — as the addressing unit cross-context retrieval will
  build on. `GET /groups` (keyset-paged directory), `PUT/GET/PATCH/
  DELETE /groups/{name}`; each row is `{name, description, contexts,
  groups}`, membership updates are deltas (`add_contexts`/
  `remove_contexts`, `add_groups`/`remove_groups`), and the same four
  operations ride MCP as `list_groups`/`create_group`/`update_group`/
  `delete_group`. A group bundles at most 1,000 member contexts and
  1,000 child groups: the delta lists were already capped per request,
  and the RESULT now is too (`over_limit`; removals apply first, so
  one request can trade members within the cap — past it, split into
  nested child groups). Referential integrity is strict: adding a
  member requires the context (`no_context`) or child group
  (`no_group`) to exist, deleting a context or a group drops it from
  every group, and boot reconciles any dangling member — or
  hand-written over-cap set, cycle, or over-deep chain — a crash or an
  edited data directory could leave behind. A group file that reads
  but does not parse keeps its name with empty content, its bytes set
  aside as `{name}.group.corrupt` and a fresh empty record written in
  their place; an UNREADABLE group file refuses the boot outright —
  registering it empty would let the next write silently overwrite
  membership that was never loaded. Nesting refusals answer
  `invalid_argument` (cycle) or
  `over_limit` (depth). Group reads/creates/updates need read/write;
  deletion is admin, like contexts. A context-scoped key sees every
  group row (child names included — labels, not content) but only the
  members its grant allows, and a group write touching any context
  beyond the grant — counted through nested children — is refused
  whole. Each group persists as one `{name}.group` JSON file beside
  the context files; one new error code, `no_group` (404). Groups
  ride export/import: a `taguru_group` record — one JSON line, the
  group's complete truth — travels the same stream batches do and
  restores AFTER every batch of a run as a create-or-replace of the
  whole record, so re-importing is idempotent and the files re-apply
  in any order. The set is validated whole (existence, caps, nesting;
  a child may be a name the same run brings) and a violation refuses
  every group record with the batches already durable; `POST /import`
  answers restored records under a new `groups: [...]` field (absent
  when the stream carried none — the old shape is untouched). A full
  `taguru export` writes each group as `{group}.group.jsonl`; a live
  server serves one at `GET /groups/{name}/export` (a context-scoped
  key exports its grant's slice, exactly the row it can read).
  `taguru inspect` verifies `.group` files too: unreadable or
  unparseable ones fail the check — a boot would refuse, or reset the
  record — and dangling references, over-cap sets, and ill-shaped
  nesting warn with exactly what boot's reconciliation would drop.
  Known limitations this iteration: `taguru compact` leaves group
  files alone (they hold nothing to compact), and a
  `DELETE /groups/{name}` whose unlink fails can resurface the group
  at the next restart (the error message says so).
- `taguru_groups_registered` gauge on `/metrics`.
- Cross-context search: `POST /recall`, `POST /query`, and
  `POST /sources/search` run one search across several contexts at
  once — `contexts: [full names]` beside the usual arguments. Every
  match carries the `context` it came from; recall/query merge on
  |weight| (weights share one scale — evidence mass) and passage hits
  interleave by per-context rank, since passage scores are
  corpus-local. The target list is vetted up front: a name beyond a
  scoped key's grant refuses the request whole (checked before
  existence, so grants cannot probe names), a missing name is
  `no_context`, an empty list is `invalid_argument`, and the list caps
  at the usual 1,000 items. The MCP search tools (`recall`, `query`,
  `search_passages`) take `contexts` as an alternative to `context`.
- The cross-context searches also take `groups: [group names]`, alone
  or beside `contexts`: each group searches every context it reaches,
  nested children included, and overlaps — with `contexts`, between
  groups, or through nesting — dedupe silently, so a context is
  searched once however many ways it was named. Directly named
  contexts lead the merge's tie order in request order; group-resolved
  members follow in name order. A name that is not a group is
  `no_group`, and the list shares the 1,000-item cap. For a scoped
  key, a group resolves to just the members the grant covers — the
  same slice group listings show it — rather than refusing, which
  would leak out-of-grant membership; directly named contexts keep the
  whole-request refusal. The MCP search tools take `groups` beside
  `contexts`.
- SDKs: groups and cross-context search on both clients. A `groups`
  resource (`client.groups`) mirrors `contexts` — `list`/`iter`/`get`/
  `exists`/`create`/`update` (deltas)/`delete`/`export` — and the
  top-level searches ride the client root: `client.recall(cue,
  contexts=…, groups=…)`, `client.query(…)`, `client.search_passages
  (…)` answer `CrossMatchPage`/`CrossPassageHit` rows, each match
  tagged with the `context` it came from. New models `GroupEntry`,
  `GroupPage`, `CrossAssociation`, `CrossMatchPage`, `CrossPassageHit`
  in both languages; surface parity is spec-checked as always.

## [0.2.0] - 2026-07-12

### Added
- Machine-readable error codes: every JSON error now carries a stable
  `code` beside the human `error` text —
  `{"status": "error", "code": "<kind>", "error": "...", "time": ...}`.
  The vocabulary (documented in `GET /protocol`): `malformed_request`,
  `invalid_argument`, `over_limit`, `unauthorized`, `forbidden`,
  `no_context`, `no_source`, `no_paragraph`, `unknown_path`,
  `method_not_allowed`, `timeout`, `already_exists`, `conflict`,
  `payload_too_large`, `rate_limited`, `internal`,
  `embeddings_unconfigured`, `embeddings_failed`, `overloaded`,
  `unhealthy`, `storage_full`. Branch on the code (or the status),
  never on message wording. The SDKs surface it as `.code` on every
  error.
- Client SDKs under `sdk/`: `taguru` for Python (sync + async, httpx)
  and TypeScript/JavaScript (fetch, zero dependencies) with one
  identical surface — typed models for every endpoint,
  idempotency-aware retry (`add_associations` never retries after an
  ambiguous transport failure; 429/503 always retry), keyset
  auto-pagination, chunked batch writes, export/import helpers, and a
  `retrieve()` implementation of the protocol's retrieval loop.
  Cross-language parity is machine-checked against
  `sdk/spec/surface.yaml` in CI; integration suites spawn the real
  server binary. Plus `langchain-taguru` for both ecosystems:
  `TaguruRetriever` (graph lane + text lane, RRF-merged, verbatim
  citations) and `TaguruIngester` (the LangChain twin of
  `taguru extract` — same prompt discipline, same merge validation,
  applied via `POST /import`'s per-source replace). The packages
  version in lockstep with the server: each `v*` release tag
  publishes all four to PyPI and npm alongside the crate.
- SDK use-case examples under `examples/langchain/` — RAG QA with
  citations, governed document ingestion (dry-run review → apply →
  per-source replace → retract), and conversational long-term memory
  with correction by negative weight — one directory per use case,
  each as a Python and a TypeScript program mirrored line for line.
  All run offline (a real server binary is spawned per run;
  deterministic fake chat models stand in for the LLM) and the SDK CI
  workflow executes every one of them.
- `taguru export` and `GET /contexts/{name}/export`: every context
  renders as the same JSONL batch stream `taguru import` and
  `POST /import` apply — the portable, version-independent backup.
  Both import entrances now read multi-batch streams (each
  `taguru_batch` header opens the next batch); a multi-batch
  `POST /import` answers `{batches: [...]}` per batch.
- `taguru compact` and `POST /contexts/{name}/compact` (admin):
  rebuild a context's image from live content alone, shedding
  retracted edges, unlinked attributions, and arena slack. Content,
  counts, and paragraph locators survive; the outcome reports the
  shrink.
- `TAGURU_KEY_SCOPES`: per-key authorization — roles
  read ⊂ write ⊂ admin plus optional per-context grants, enforced
  identically over raw HTTP and MCP tool dispatch. Keys the variable
  does not name keep the historical full grant.
- `TAGURU_MAX_CONCURRENT_REQUESTS` (default 256): an in-flight
  ceiling that sheds excess load with 503 + Retry-After before auth
  runs; probes stay exempt. New `taguru_inflight_requests` gauge and
  `taguru_requests_shed_total` counter.
- `GET /live`: unconditional liveness probe. `/health` keeps the
  readiness (write-path) signal; wire orchestrator probes
  accordingly.
- Audit trail: the access log now names the context each request
  addressed, and destructive operations (context delete, source
  retract, alias removal, import batches, compaction) each leave one
  self-contained `taguru::audit` event naming the key and the object.
- Embedding resilience: transient provider failures (transport, 429,
  5xx) retry twice with backoff; `TAGURU_EMBED_TIMEOUT_SECS` makes
  the per-attempt ceiling a knob; a new
  `taguru_embedding_duration_seconds` histogram times every round
  trip; boot warns when the request timeout sits under the provider
  ceiling.
- Load quarantine: a context (or passage store) whose load keeps
  failing answers its remembered refusal for 30s instead of
  re-reading the broken files on every request; restoring the files
  heals it on the next retry.
- Pinned contexts preload in parallel at boot.
- Deployment examples under `deploy/` (Kubernetes, docker-compose)
  matching the documented single-writer model.
- `examples/http_benchmark`: concurrent load against a running server
  — throughput and p50/p95/p99 per phase (seed writes, reads, a 90/10
  mix), the capacity-planning companion to the library benchmark.
- The protocol doc states the compatibility policy: no `/v1` (the
  protocol travels with the server), additive responses parsed
  tolerantly, pre-1.0 shape changes announced here.
- Documentation site at <https://t0k0sh1.github.io/taguru/>: getting
  started, concepts, the import/extract references, per-platform
  deployment guides (Docker Compose, Kubernetes, Amazon Bedrock), the
  internal architecture, and a captured MCP retrieval walkthrough. The
  README slims down to a user-facing overview that points there.

### Fixed
- The protocol document and README now list `/live` among the
  auth-exempt probes — the code always exempted it alongside
  `/health` and `/metrics`; only the docs omitted it.
- A failed `DELETE /contexts/{name}` unlink could leak the context's
  sidecar files forever — or, if `.ctx` itself survived, resurrect
  the context at the next boot. Deletion now writes a durable
  `.deleted` marker first, boot resumes any deletion it finds a marker
  for, and recreating a context clears a stale marker so a failed
  delete followed by a same-name create cannot be undone at the next
  boot.
- Export is now a true fixed point: a context with sourceless weight
  exports a reserved `export:unsourced` batch, and re-exporting the
  restored context (which carries a real attribution to that id) folds
  it back instead of refusing — the round trip export exists for no
  longer breaks on its own output.
- `taguru export` writes each stream atomically (stage + fsync +
  rename), so a crash while refreshing a backup no longer truncates
  the previous good copy.
- `/flush` refuses a context-scoped key (it is server-wide and names
  every flushed context); authorization now wraps the `/mcp` and OAuth
  routes it previously missed; and `@` in a key name is refused at
  boot (it collided with the OAuth-delegation scope fallback).
- A context compaction racing a background flush's stage-then-publish
  window could have the flush win the race and republish
  pre-compaction bytes over the compacted image, silently reverting
  the associations the compaction had just discarded. A per-entry
  generation counter now detects a Hot-to-Hot swap mid-flush and backs
  the stale republish off instead.
- `POST /import`'s multi-batch apply loop, `create`/`update` (a pin
  toggle can also load a context from disk) and `delete` on
  `/contexts/{name}`, and passage lookup/citation/listing on a
  context's first cold load all ran their synchronous, fsync-bearing
  I/O directly on the async runtime with no `block_in_place` — a large
  import alone could stall it for seconds, delaying every other
  in-flight request. All now wrap their blocking calls, matching the
  rest of the write and passage-search paths.
- A non-numeric component in an embedding provider's response silently
  became `0.0` instead of refusing the response like every other
  malformed shape (missing vector, wrong width, bad index) — a
  corrupted vector could then rank as a plausible neighbor in
  similarity search. It now refuses and names the offending index.
- `activate`'s decay and every `dice_floor` entry point
  (`resolve_with_floor`, `set_dice_floor`,
  `similar_concepts`/`similar_labels`) clamped into `[0, 1]` with a
  bare `.clamp`, which passes a NaN input straight through instead of
  clamping it — flipping some fail-closed filters open. Each now maps
  NaN onto the safe extreme instead.

### Changed
- **Response shape** (pre-1.0 break): `POST /import` now answers
  `{batches: [...]}` for a single-batch body too (was: that batch's
  bare outcome) — one shape for every import, no client-side
  branching on stream length.
- A request body over `TAGURU_MAX_BODY_BYTES` now answers 413 in the
  same JSON error shape as every other axis (was: axum's plain-text
  rejection).
- `add_associations`' partial-write arm keeps the capacity/conflict
  status split (507 vs 409) every other batch write reports —
  previously it answered 507 unconditionally. Unobservable today
  (association writes only fail on capacity), pinned for uniformity.
- **Response shapes** (pre-1.0 break): `GET /contexts/{name}/labels`,
  `.../aliases`, and `.../sources` now page like the directory —
  `?limit=1000&after=...` in, `{total, ...}` out. The alias cursor
  spans both namespaces (`after=concept:<alias>` or `label:<alias>`).
  MCP tool schemas carry the same parameters.
- Embedding failures no longer embed the provider URL in client-facing
  502 bodies; messages name the status code or transport error kind.
- Boot warns when listening beyond loopback with the per-key rate
  limit off.

### Security
- The OAuth grant store (`oauth.json`) is created owner-only (0600) at
  open time — born with the mode, not chmod'd after, so no readable
  window exists between create and the secret write.
- The OAuth consent page carries `X-Frame-Options: DENY`, a
  locked-down `Content-Security-Policy`, and
  `Referrer-Policy: no-referrer`.
- Dynamic client registration accepted a `redirect_uri` by
  string-prefix-matching `"https://"`, so
  `https://trusted-app.example.com@evil.attacker.com/callback`
  registered without error — the host an approved code actually
  reaches is the attacker's domain after the `@`, not the
  trusted-looking name before it. Registration now parses the URI
  structurally and refuses any userinfo component outright.

## [0.1.0] - 2026-07-05

Initial release: the association-graph library (flat-buffer images,
WAL-backed durability), the HTTP server (auth, rate limits, metrics,
OTLP tracing, OAuth for remote MCP), the MCP stdio bridge, and the
offline tooling (`import`, `extract`, `inspect`, `estimate`).
Published to crates.io and GHCR.

[Unreleased]: https://github.com/t0k0sh1/taguru/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/t0k0sh1/taguru/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/t0k0sh1/taguru/releases/tag/v0.1.0
