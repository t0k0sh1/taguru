# Changelog

Notable changes to taguru. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [SemVer](https://semver.org/) (pre-1.0: minor bumps may break).
Entries that change an on-disk format or a response shape say so.

## [Unreleased]

### Added
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

### Fixed
- A failed `DELETE /contexts/{name}` unlink could leak the context's
  sidecar files forever — or, if `.ctx` itself survived, resurrect
  the context at the next boot. Deletion now writes a durable
  `.deleted` marker first, and boot resumes any deletion it finds a
  marker for.

### Changed
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
- The OAuth grant store (`oauth.json`) is written owner-only (0600),
  mode set before content lands.
- The OAuth consent page carries `X-Frame-Options: DENY`, a
  locked-down `Content-Security-Policy`, and
  `Referrer-Policy: no-referrer`.

## [0.1.0] - 2026-07-05

Initial release: the association-graph library (flat-buffer images,
WAL-backed durability), the HTTP server (auth, rate limits, metrics,
OTLP tracing, OAuth for remote MCP), the MCP stdio bridge, and the
offline tooling (`import`, `extract`, `inspect`, `estimate`).
Published to crates.io and GHCR.

[Unreleased]: https://github.com/t0k0sh1/taguru/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/t0k0sh1/taguru/releases/tag/v0.1.0
