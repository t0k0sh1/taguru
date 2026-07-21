# Changelog

Notable changes to taguru. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [SemVer](https://semver.org/) (pre-1.0: minor bumps may break).
Entries that change an on-disk format or a response shape say so.

## [Unreleased]

### Added
- `TaguruIngester` gains an optional `on_event` progress callback (#177) —
  synchronous, typed events (`document_started`, `chunk_started`,
  `attempt_started`, `attempt_failed`, `chunk_completed`, `import_started`,
  `import_completed`, `embedding_refresh_started`/`completed`/`warning`)
  fire from both `ingest_text` and `aingest_text`, so a caller can show
  live progress and see *why* a corrective attempt fired (parse error,
  provider finish reason, token usage when the model reports them)
  without copying the private extraction helpers. Callback exceptions are
  caught and reported via `warnings.warn` rather than failing the ingest.

## [0.4.0] - 2026-07-20

### Added
- Source metadata and pre-lane search filters (#167) — the two B-1
  entries #148 deferred as a data-model change. Every stored source
  now carries a server-stamped `stored_at` (epoch seconds, stamped
  once as the WAL op is built so replay never re-stamps), an optional
  user-supplied document `date`, and `tags` — accepted by
  `POST /contexts/{name}/sources` (`tags`/`dates` per-source maps;
  MCP: `store_passages`) and the import batch format (riding the
  `passage` line; export writes them back, and import preserves an
  exported `stored_at` so restores don't re-date the corpus), durable
  through the passage WAL, a new S5 snapshot generation (S1–S4 read
  forever), compaction, WAL shipping, and restore-on-start. Passage
  search (`POST /contexts/{name}/sources/search`, `POST
  /sources/search`; MCP: `search_passages`; SDKs:
  `search_passages`/`searchPassages`) takes `tags` (any-of) and a
  half-open `[since, until)` window over each source's `date ??
  stored_at`, resolved to an eligibility set BEFORE the BM25/vector
  lanes run — BM25 statistics stay corpus-global (the filter gates,
  never re-weights), the ANN probe widens until its oversample target
  is met among eligible rows, and absent metadata never matches a
  filter (pinned by tests). The plan gains a per-context
  `filter: {eligible_sources, total_sources}` block,
  `search/explain` gains the same filter params and a `filtered_out`
  verdict, `GET /contexts/{name}/sources` lists metadata back under
  `entries`, and retrieval/semantic cache keys carry the filter on
  both search variants (same query, different filter: never one
  entry).
- Community analysis as an offline derived index (#166) — the
  corpus-overview surface GraphRAG's global search answers and taguru
  had no verb for. `GET /contexts/{name}/communities` detects
  communities on the concept graph server-side (hand-rolled
  deterministic Louvain with a component-split pass, `louvain-cc/1`;
  hierarchical via aggregation; heavy-ops gated) and streams them with
  the revision snapshot the analysis was cut at. `taguru communities`
  — an HTTP client to a running server, like calibrate — turns that
  into an artifact: an ordinary context (default `{name}::communities`)
  holding one summary passage per community, membership and hierarchy
  as associations, and a manifest recording the source revision.
  Summaries come from the extract provider and are incremental by
  content fingerprint: an unchanged graph re-runs with zero LLM calls,
  and only changed communities re-summarize.
  `POST /contexts/{name}/communities/search` (MCP: `search_communities`;
  SDKs: `search_communities`/`searchCommunities`) ranks the summaries
  with the same two-lane passage search — plan, floors and all — and
  answers with an honest staleness verdict (`stale: true` = the source
  graph moved since derivation); a missing artifact is a refusal
  naming the build command, never an empty result. Because the
  artifact is an ordinary context, quotas, retrieval caching (a new
  cache op keyed on the artifact AND the source's current graph
  revision), export/import, groups, and router-mode routing all apply
  unchanged.
- Image supply chain, made verifiable end to end (#138). The release
  pipeline already signed the multi-arch manifest (Sigstore keyless)
  and attached BuildKit SBOM + SLSA provenance since 0.2.0 — but the
  SBOM was vacuous: a scratch image gives the scanner nothing, so two
  releases shipped an SPDX document cataloguing zero crates. The
  binary is now built with `cargo auditable`, which embeds the
  Cargo.lock crate list in the executable — the same attached SBOM
  now carries every crate with its version, and a bare binary is
  auditable outside any container (`cargo audit bin`). A new `verify`
  job gates every release from a consumer's seat — blank runner, no
  registry login, no OIDC — on three checks: the signature binds this
  repository's exact workflow identity (unanchored identity patterns
  match look-alike repos; the docs pin it), the SBOM actually lists
  crates (a ≥100 floor, against the vacuous-SBOM failure that shipped
  silently), and the provenance names this repository's CI run.
  Verification is documented for consumers in SECURITY.md ("Verifying
  a release"), and deploy/README.md now spells out digest pinning —
  a tag pin is a convention, `tag@sha256:…` is a guarantee — with the
  kustomize `digest:` override alongside.
- Hot-reload for the auth table (#134): `TAGURU_API_TOKEN`,
  `TAGURU_API_TOKENS`, and `TAGURU_KEY_SCOPES` — and nothing else —
  now reload on a running server, so key rotation no longer costs the
  restart-outage a single-writer boot implies. Two triggers, one
  swap: SIGHUP (unix; previously the unhandled default disposition
  made SIGHUP *kill* the server), and a ~5s watch on the `--config`
  file when one was given — which is what makes the Kubernetes
  secret-volume flow hands-free (mount the Secret as a file, `kubectl
  apply` a rotation, the kubelet's atomic symlink swap reaches the
  watch; no exec, no signal) and gives non-unix platforms, which have
  no SIGHUP, the same rotation. Reload sources mirror boot precedence
  exactly — a shell-set variable keeps winning over the file at every
  reload (which is also why an env-only deployment reloads as an
  explicit no-op: a live process's environment cannot change).
  Fail closed on every path: an unreadable or malformed source keeps
  the previous table armed with a loud error line, and the one
  transition a reload must never perform — "tokens configured" → "no
  tokens", which would silently reopen the server unauthenticated —
  is refused outright (arming keys on an open dev server stays
  allowed; that direction closes). Every gate reads the ring through
  one `SharedKeyring` handle and resolves authentication AND scope
  from a single per-request snapshot (the bearer gate now stamps the
  resolved `KeyScope` onto the request, `enforce_authorization`
  judges from it keyring-free, and `/mcp` stamps it through to
  dispatched tool calls), so in-flight requests see the old table or
  the new one — never a torn one, and never a removed key falling
  through `scope_of` to the unscoped admin default mid-request.
  OAuth delegations minted from a removed key die with it (the
  per-request `recognizes` check reads the swapped ring); per-key
  rate-limit buckets need no hook (new names start at full capacity,
  removed names' idle buckets fall to the existing prune). Each
  reload leaves one `taguru::audit` line — keys added / removed /
  rotated (same name, new bytes — the k8s case a name diff can't
  see) / rescoped, names only, never token bytes, with an explicit
  "no change" line so a SIGHUP is never silent — plus two counters:
  `taguru_keyring_reloads_total` and the alertable
  `taguru_keyring_reload_refusals_total` ("the rotation you think
  you performed didn't take"). `taguru route` is untouched: the
  router holds no keyring by design.
- Per-context quotas (#136), declared as one JSON env in the
  `TAGURU_KEY_SCOPES` mold — `TAGURU_CONTEXT_QUOTAS='{"name":
  {"storage_bytes": …, "cache_bytes": …}}'`, each field optional but
  never both absent; a broken declaration refuses boot, like broken
  credentials. `storage_bytes` caps the context's whole on-disk family
  (image, both WAL lanes, passages snapshot, sidecars — the same sum
  `taguru_context_disk_bytes` serves, read from the live WAL
  bookkeeping plus the flush-refreshed #137 snapshot, which now stays
  on whenever a storage quota is declared even with the gauges off):
  at or over the ceiling, growth writes refuse with the
  already-documented 507 `storage_full` across every entrance — graph
  batches (associations/aliases, via the one `logged_write`
  chokepoint, gated only when a batch carries growth ops),
  `store_passages`, and `/import`, which stops before the first capped
  batch as a resumable prefix exactly like a spent deadline (dry runs
  stay advisory). Retract, unalias, `DELETE`, and compaction stay open
  at the ceiling — shrinking is how a tenant gets back under, the line
  the passage store's own cap already draws. `cache_bytes` is the
  ceiling side of the pinning floor: no reservation while there is
  slack, but under cache pressure a context past its declared share is
  evicted before any compliant one, so the eviction damage one
  saturating context can inflict on the rest is bounded by its ceiling
  (pinning still wins — a pinned context never enters the sweep).
  Declared ceilings surface as
  `taguru_context_quota_bytes{context,resource="storage"|"cache"}`
  beside the #137 usage families (same knob, same top-N cut), and
  refusals count on `taguru_storage_quota_refusals_total`. Offline
  commands (`taguru import`/`compact`/…) run as the operator, outside
  the policy; a replica refuses writes before any gate and honors only
  the eviction ordering.
- Ratio-triggered auto-compaction for contexts (#135), default on:
  each flusher tick rebuilds at most the one worst context whose dead
  ratio (dead edges / total edges — the bookkeeping the maintenance
  sweep already reads, live for hot contexts, sidecar stats for cold)
  strictly exceeds `TAGURU_AUTO_COMPACT_RATIO` (default 0.5, i.e. dead
  weight outgrew live content — the graph-side restatement of the
  passages store's own self-compaction ratio). The compaction takes a
  permit from the same `TAGURU_MAX_CONCURRENT_HEAVY_OPS` pool manual
  calls contend on (no free slot: the candidate waits for a later
  tick), runs under a 60-second budget so one oversized rebuild cannot
  stall the loop that persists every other context (a context that
  blows the budget is set aside for the process's lifetime with a
  pointer at `POST /maintenance/compact` / offline `taguru compact` —
  retrying a rebuild that cannot finish would burn the budget every
  tick), reuses `compact_context` verbatim (same crash guarantee: the
  fresh image carries the old WAL watermark), leaves the same
  `taguru::audit` "context compacted" line with `trigger="auto"` and
  the measured ratio, and shows on `/metrics` as
  `taguru_auto_compactions_total{outcome}`,
  `taguru_auto_compact_reclaimed_bytes_total`, and
  `taguru_auto_compact_last_success_timestamp_seconds`.
  `TAGURU_AUTO_COMPACT=0` restores manual-only compaction for
  scheduled quiet-window sweeps; replicas never auto-compact (they run
  no flusher — their images belong to the primary).
- Per-context capacity gauges on `/metrics` (#137), behind
  `TAGURU_METRICS_PER_CONTEXT` (default off; `1`/`all` = every
  context, `N ≥ 2` = the top-N by total disk bytes): the
  `taguru_context_*` families — on-disk bytes by file family (image,
  graph WAL, passages snapshot, passages WAL, sidecars), modeled
  resident bytes, the pinned flag, and concept/association/label/
  source counts. Sizes come from a stat sweep at each flush tick,
  `POST /flush`, and boot — never from the scrape, which reads only
  registry state; counts and residency are live for hot contexts and
  the last-saved snapshot for cold ones, the same semantics
  `GET /contexts` serves. The WAL series reuse the existing live
  bookkeeping, so they sum exactly to `taguru_wal_bytes` and friends.
- `taguru calibrate` (#131): measures the semantic-floor bands of a
  running server's embedding model instead of prescribing the manual
  ritual. `--context NAME --probes FILE` (TSV `cue<TAB>expected`
  pairs) drives the resolve/explain machinery: the expected name's own
  gloss cosine feeds the upper band (measured floor-independently),
  the best other semantic candidate a 0.05-floor resolve surfaces
  feeds the lower, and the report prints both distributions, the gap,
  and a suggested `TAGURU_SEMANTIC_FLOOR` mid-gap — `--json` for
  automation, one provider embed per probe (the cue cache covers the
  second call). Probes whose cue lexically resolves — the step humans
  get wrong; the semantic tier never scores them — are excluded
  loudly, each with its own diagnosis; overlapping bands earn a
  warning verdict, never a fabricated number. Auth and URL resolution
  ride the same variables the server reads (`TAGURU_API_TOKEN`/
  `TAGURU_API_TOKENS`, `TAGURU_ADDR`, `--config`), and the run is
  read-only, so a replica serves it.
- `GET /contexts/{name}/embeddings` (#131): the embedding identity in
  one read — the configured provider model beside the (model, width)
  each vector sidecar was actually built with (#133's recorded
  identity, now exposed), plus row counts. What `taguru calibrate`
  stamps its report with, and the state to check after a model switch
  without provoking a search. Read role, replica-safe.
- Execution plans on every search response (#151): recall/query
  (single-context and cross) gain a `plan` object beside
  `total`/`matches` — additive — carrying `contexts`, the list of
  contexts actually consulted in effective order. For the cross
  variants that is the resolved target list (groups expanded, the
  key's grants applied), which the tagged matches alone cannot
  reconstruct when a target comes back empty; it names nothing a
  caller cannot already see through `GET /groups`. The passage-search
  half of #151 is a response-shape change and lives under **Changed**
  below. The MCP `retrieve` tool and both SDKs' composed `retrieve()`
  forward the fallback search's plan as `search_plan` (null/absent
  when no fallback ran).
- Semantic retrieval cache for passage search (#153): with
  `TAGURU_SEMANTIC_CACHE_THRESHOLD` set (off by default), a
  paraphrased `sources/search` (single-context or cross, MCP
  included) can answer from an equivalent earlier query's exact-cache
  entry. Equivalence requires the query-vs-query embedding cosine
  (through the same cue cache the search uses — no extra provider
  calls on the fresh path) to clear the threshold AND a
  negation/number/entity guard to find no mismatch between the two
  query texts, so "does it X" never serves "does it not X", a changed
  number, or a swapped name at high cosine. The tier stores only
  equivalence claims — no payloads, no invalidation machinery: a
  match rewrites the request's exact key to the canonical query's
  params under current revision fingerprints, so every #150 freshness
  guarantee (revision lanes, identity nonce, delete-recreate, replica
  lineage) applies unchanged, and a post-write serve becomes a
  `stale` fall-through that re-canonicalizes the cluster. Outcomes in
  `taguru_semantic_cache_total{outcome=hit|stale|guarded|miss}` plus
  a `taguru_semantic_cache_entries` gauge; semantic serves add
  `similarity` and `matched` to the opt-in search log line. Guard
  blind spots (spelled-out and kanji numerals, sentence-initial
  English entities, entities in unsegmented scripts) are documented
  in `src/registry/semantic_cache.rs` — the threshold is the primary
  filter and the default posture is off. No request or response shape
  changes.
- Exact-match retrieval cache (#150): an identical
  recall/query/passage-search request (single-context and cross
  variants, MCP tool calls included) against an unchanged corpus now
  answers from the stored response bytes without re-running the
  search. Invalidation is the key itself: each entry is keyed on the
  resolved target list plus, per target, the #149 revision lanes that
  surface depends on (recall/query: graph+passages; passage search:
  passages+config) and a per-incarnation identity — read before the
  search runs — so a write simply makes stale entries unreachable
  (delete-recreate and replica lineage switches included), with no
  TTL and no purge hooks. Scoped keys share an entry exactly when
  their grants resolve a request identically. Byte-budgeted LRU:
  `TAGURU_RETRIEVAL_CACHE_BYTES` (default 32 MiB, `0` disables).
  Hit/miss per op lands in `taguru_retrieval_cache_total` with
  entry/byte gauges beside it, while `taguru_searches_total`, the
  passage lane contributions, and per-context usage counters keep
  counting served responses — dashboards read continuously. Cache
  hits emit the opt-in search log line with `cached=true`. No
  request or response shape changes.
- Context revision counters (#149): every directory row (`GET
  /contexts`, `GET /contexts/{name}`, and the MCP
  `list_contexts`/`get_context` pass-throughs) now carries
  `revision: {graph, passages, config}` — applied graph writes, the
  passage log watermark, and config/embedding changes respectively —
  the change token the upcoming retrieval caches key on. Group rows
  gain a `fingerprint` hashing the scope-visible transitive members'
  counters, so a group-level cache invalidates exactly when a
  relevant member changed. Compare for equality only: within one
  process the counters are live and strictly monotonic; across a
  crash a cold context can lag until its first load, and a cache
  that outlives the process must treat a restart (or a
  delete-recreate) as invalidation. Response-shape addition only;
  the `.meta.json` sidecar gains a `revision` field older sidecars
  simply lack (they read as zeros until the first flush).
- Passage search takes a per-request `semantic_floor` (#148): a
  one-call override of the vector lane's cosine floor — request beats
  the context setting beats the server default, the same chain
  resolve's override walks — on `/contexts/{name}/sources/search`,
  cross-context `/sources/search`, and `…/sources/search/explain`
  (the explanation reports the floor it actually ran under). Exposed
  through the MCP tools and both SDKs. It floors only the vector
  lane: BM25-only hits still return, and the fused score stays rank
  arithmetic.
- Embedding tier resilience (#132): the provider now sits behind a
  small circuit breaker — three consecutive failed attempts open it,
  every embedding call then fails fast (the lanes behind it degrade
  exactly as they do today: `sources/search` serves its lexical lane,
  resolve serves lexical candidates or answers the existing
  `embeddings_failed`) instead of each paying the provider timeout,
  and after a 30s cooldown a single probe decides whether to close.
  Breaker state, opens, and short-circuit counts land on `/metrics`
  as `taguru_embedding_breaker_state` /
  `_consecutive_failures` / `_opened_total` / `_short_circuits_total`
  (present only when a provider is configured).

### Changed
- **On-disk format**: the gloss vector sidecar (`{stem}.vectors.bin`)
  now records its vector width beside the model — header `TAGURUV2` →
  `TAGURUV3` (#133). Existing V2 sidecars load exactly as before (the
  width is taken from the rows, which loads now verify are uniform)
  and are stamped to V3 by their next save, so the upgrade costs no
  provider spend; a binary older than this release, however, reads a
  V3 file as corrupt and re-embeds on its next refresh. Loads of
  either header refuse rows that mix widths, and a V3 header
  disagreeing with its rows, the same way they refuse other
  corruption: discard, warn, re-embed — never serve. The passage
  sidecar (`TAGURUP2`) already recorded (model, dim) and is unchanged.
- **Response shape**: `POST /contexts/{name}/sources/search` and cross
  `POST /sources/search` now answer `{plan, hits}` instead of a bare
  hit array (#151) — the hits themselves are unchanged, moved under
  `hits`; `plan.contexts` carries, per context actually searched,
  whether each lane ran (`bm25`/`vector`, mirroring the per-hit
  `lanes` shape), the reason when one was skipped — embeddings off,
  nothing embedded yet, model changed, provider refused, in the same
  prose `sources/search/explain` uses — and the vector lane's
  effective cosine floor when it swept (the override → context
  setting → server default chain, resolved per context). This is what
  makes "did the semantic lane actually run" visible without a
  diagnostic call; a zero-hit page under a skipped lane no longer
  reads as "nothing matched". Breaking for clients that parse the
  search result as an array: both SDKs (`search_passages` /
  `searchPassages` now return the page object with `.hits` and
  `.plan`), the langchain retrievers, the MCP `retrieve` tool, and
  `taguru route`'s shard merge move in lockstep in this release —
  upgrade router and shards together, and pin older SDKs to older
  servers (the pre-1.0 posture `GET /protocol` and this file already
  declare; older Python SDKs fail loudly on the new shape rather than
  mis-reading it). The plan rides inside the cached result bytes, so
  #150 exact hits and #153 semantic serves replay it unchanged — every
  event that could alter a plan (corpus write, vector publish, floor
  change) already moves the cache key.
- The approximate passage index now activates at 10 000 vector rows
  instead of 50 000 (#148). The old threshold sat above the default
  `TAGURU_PASSAGE_VECTOR_LIMIT` (20 000), so no default-config
  deployment could ever reach the index — every semantic sweep was
  the linear scan. Calibrated by benchmark (in the PR): at 10k rows
  the exact sweep costs 6–14 ms of read-fenced CPU per query against
  <1 ms via the index at ~100% measured recall@10, and the lazy
  one-time build (0.6–1.3 s) fits inside a request budget, which the
  7–15 s build at 50k did not reliably do. A compile-time assertion
  pins the threshold at or below the default vector limit, and boot
  says so once when an operator-lowered limit puts the index out of
  reach.
- Embedding provider calls are deadline-aware and stop-signal-aware:
  each attempt's HTTP timeout is the smaller of
  `TAGURU_EMBED_TIMEOUT_SECS` and the request's remaining budget (a
  slow provider is cut at the budget and that request's lane
  degrades, instead of holding the request past its own timeout and
  answering 408 after the work was done), and SIGTERM/SIGINT abandons
  in-flight provider waits, so a graceful drain no longer waits out
  the timeout ladder (up to 180s at the defaults — measured down to
  under a second). The deploy manifests resize accordingly:
  `terminationGracePeriodSeconds` / `stop_grace_period` drop from
  200(s) to 60(s), now sized by the request budget plus the final
  flush.

### Fixed
- A provider changing vector width behind an unchanged model name (a
  `dimensions` setting is a request-time parameter on Titan V2 and
  Matryoshka-style models) no longer produces silently empty semantic
  results in the window before the next refresh (#133). The refresh
  side already detected the change and re-embedded both stores
  wholesale; the serve side now refuses to score across the mismatch
  — every cosine would be `similarity`'s 0.0 width-mismatch sentinel
  — and names it instead: search plans and `sources/search/explain`
  report stored vs current width as a `ran: false` reason (the plan
  previously claimed `ran: true` over that all-zero sweep), and
  `resolve/explain`'s semantic report does the same where it
  previously presented the 0.0 sentinel as a measured cosine and
  prescribed lowering a floor no value could satisfy. `resolve`
  itself keeps folding to empty, exactly like a model change. The
  width-triggered wipe is now counted
  (`taguru_embedding_width_rebuilds_total{store="gloss"|"passages"}`)
  beside its existing warn line, and the Bedrock page's "pick a
  dimension once and never change it" instruction became "detected
  and rebuilt — still pick one width; rebuilds cost provider spend".
- A passage search whose query embedding was refused by the provider
  (a transient failure — the one vector-lane state that recovers with
  no revision bump) is no longer filled into the retrieval caches
  (#151): previously the degraded BM25-only page was cached like any
  other result and kept serving — and could canonicalize semantic-tier
  paraphrases onto itself — until the next unrelated corpus or config
  write, silently outliving the provider's recovery. The degraded page
  is still served (and its plan now confesses the failure); it is just
  never pinned. Stable skip states (embeddings off, nothing embedded,
  model changed) stay cacheable — config changes do move the key.
- Hydration against a LIVE lineage no longer mistakes the writer's own
  progress for rot: a replica (or stateless writer) booting while the
  writer keeps shipping could fetch an object the writer had just
  replaced — newer bytes than the manifest snapshot — and refuse to
  start with "downloaded bytes do not match the manifest". A
  verification mismatch now re-reads the generation's manifest and
  retries against whatever it currently says (a few paced rounds,
  every fetched shape: published files, sidecar meta, log lanes with
  reset series); true rot — bytes disagreeing with a manifest that is
  NOT moving — still refuses exactly as before. Missing bucket objects
  also keep their NotFound kind through the download wrapper, so a
  lane whose old series aged out heals the same way.

### Added
- Kustomize packaging for the Kubernetes manifests (#139):
  `deploy/kustomize/` serves `kubectl apply -k` over the reference
  manifests — a base (the single-writer PVC model) and overlays for
  the stateless, writer+replicas, and sharded-router variants; the
  router overlay is the two-shard fleet worked out (writer shards via
  nameSuffix + selector labels, the route-map as a content-hashed
  generated ConfigMap so a map edit rolls the routers on apply, and
  the front-door Deployment/Service). Kustomize over a Helm chart,
  with the rationale recorded in `deploy/kustomize/README.md`: the
  reference manifests stay the documentation (comments intact,
  consumed verbatim), `apply -k` needs no extra tool, and the retuned
  knobs are patches, not templates. `deploy/kustomize/verify.sh` —
  run by a new CI workflow on every PR touching `deploy/` — keeps the
  in-tree manifest copies byte-identical to the reference files,
  schema-validates every rendered configuration (kubeconform), and
  asserts the base renders equivalent to
  `kubectl apply -f kubernetes.yaml`.
- Router mode (#130): `taguru route` is a stateless scatter-gather
  router over sharded instances — `TAGURU_ROUTE_MAP` names a
  `context = shard-url` map file (optional `* = shard-url` fallback),
  context verbs proxy byte-for-byte to the owning shard, and the
  cross-context searches (`POST /recall`, `/query`,
  `/sources/search`), the directories, and groups span every shard
  with the exact single-instance merge semantics — `after` cursors
  forward verbatim (they anchor on the last match, not on a
  per-instance position), and equivalence with a single instance
  holding the same contexts is pinned by an integration test. Groups
  exist on every shard with member lists projected by the map;
  `/import` splits its batch stream by context and dry-run-preflights
  batch chunks and projected group records alike, so a stream one
  instance would refuse with nothing applied is refused the same way; `POST /mcp` works
  unchanged (bearer re-attached to each dispatched call). Auth is
  pass-through — the router holds no key store (setting
  `TAGURU_API_TOKEN(S)`/`TAGURU_KEY_SCOPES`/`TAGURU_PUBLIC_URL` on it
  refuses to boot), shards keep enforcing keys, scopes, and rate
  limits. Failure honesty: a shard that answers an error fails the
  request whole; an unREACHABLE shard degrades fan-out reads to
  labeled partials — the envelope gains an `unreached` array
  (response-shape note: new optional top-level field, absent when
  every shard answered) — and routed verbs answer the new
  `502 shard_unreachable` error code. Router-shaped `/metrics`
  (`taguru_router_*`). Moving a context is a documented runbook:
  quiesce → export → delete from the old shard through the router →
  map edit + rolling router restart → re-import through the router.
- Read replicas (#129): `serve --replica` / `TAGURU_REPLICA=1` serves
  the replication bucket's lineage read-only and keeps tailing it —
  issue #128's hydration running continuously. Every retrieval verb
  (resolve, describe, recall, query, activate, explore,
  sources/search, the listings and exports) serves from the replica's
  own hydrated copy, so reads scale horizontally with the pool; every
  mutating verb — raw HTTP and the MCP write tools alike — answers
  `403 read_only_replica` naming the writer (`TAGURU_WRITER_URL`,
  plus the bucket's fence holder), and neither SDK retries it. A
  replica never claims a generation and never ships; deletions and
  new contexts propagate; a new writer's generation is followed live,
  no restart. Consistency, stated honestly: per context at that
  context's applied watermark, cross-context skew possible, staleness
  bounded by shipping lag + `TAGURU_REPLICATE_INTERVAL_MS`; a bucket
  outage freezes the replica at its last watermark and it keeps
  serving. `deploy/kubernetes-replicas.yaml` is the read-pool
  manifest.
- Replica lag on `/metrics` (#129): `taguru_replica_applied_seq` vs
  `taguru_replica_shipped_seq` per context and lane,
  `taguru_replica_behind_seconds`, the followed generation, and the
  manifest/poll freshness timestamps — the promotion-time RPO on
  display, which is what the manual promotion runbook (now in the
  architecture docs, rehearsed end-to-end by an integration test)
  reads before flipping a replica's directory into the next writer.
  Availability with a replica pool is promotion time, not restore
  time; what the bucket never received — the deposed writer's
  un-shipped tail — remains the async-replication RPO.
- Boot from the bucket (#128): with `TAGURU_REPLICATE_URL` set, a
  server started on an **empty** data directory materializes itself
  from the bucket's newest complete generation instead of starting
  blank — the volume demotes to a cache of the bucket lineage, and
  recovery becomes "start anywhere". Hydration is lazy and
  priority-ordered: shared files (groups, the grant store, every
  context's sidecar meta) land before boot, pinned contexts hydrate in
  parallel before the port opens, and everything else hydrates on
  first touch or via a background fill; local files whose bytes
  already match the manifest are reused without a download, so warm
  restarts of a cache-mode volume stay cheap. The successor's own
  generation is not marked `complete` until every family has settled
  locally, so a restore can never land on a hollow lineage.
  `deploy/kubernetes-stateless.yaml` is the emptyDir variant this
  enables.
- The takeover guard (#128): starting a writer against a bucket IS the
  takeover/promotion act, so while the bucket's newest generation
  still looks alive — a heartbeat object refreshed every minute, no
  clean-shutdown marker, within a 300s grace — booting a different
  writer against it refuses unless the operator states the intent with
  `serve --take-over` / `TAGURU_TAKEOVER=1`. A cleanly stopped writer
  retires its generation on the way out and never trips the guard; a
  crashed one ages out of it. Ergonomics only: epoch fencing (#127)
  remains the sole arbiter, and a writer past the guard still deposes
  its predecessor cleanly and loudly. The deposed writer's un-shipped
  tail exists only on its own volume — a successor hydrating elsewhere
  serves the lineage without it.
- The `complete` marker now carries a manifest — every shipped file's
  and lane's exact extent (length + CRC-32C), refreshed after each
  batch of uploads. `taguru restore` verifies every downloaded object
  against it (a swapped or rotted object is a refusal, not a quiet
  divergence); pre-manifest buckets keep restoring through the listing
  fallback, unverified as before.
- Continuous replication to object storage (#127): set
  `TAGURU_REPLICATE_URL` (`s3://` / `gs://` / `az://` / `file://`,
  credentials via each cloud's default chain) and a background shipper
  continuously copies every context's complete file family to the
  bucket — both log lanes (the graph WAL and the passage log) tailed
  record-by-record with the same CRC-32C verification replay runs,
  published files (images, meta, sources, passage snapshots, derived
  sidecars, groups, the OAuth grant store, crash markers) whole on
  change. Durability becomes two honest tiers: a local crash still
  loses nothing (unchanged), and losing the machine or volume now
  costs at most the shipping lag — seconds, exported per lane as
  `taguru_replication_lag_records` / `_lag_seconds` in `/metrics`,
  beside upload/error counters and a last-success timestamp. Shipping
  polls; the acknowledge path gains no work, no latency, and no new
  failure modes (a dead bucket degrades replication only).
  `TAGURU_REPLICATE_INTERVAL_MS` (default 1000) is the cadence.
- `taguru restore --out DIR [URL]`: materializes a data directory from
  the bucket's newest complete generation — published files verbatim,
  each log lane reassembled from its shipped segments and re-verified
  record-by-record — refusing gapped segment runs and non-empty
  targets. Verify the result with `taguru inspect`; the derived
  sidecars ride along but remain rebuildable, so a restore tolerates
  their absence.
- Epoch fencing on the replication bucket (#127): each writer claims a
  monotonic generation with a conditional create and ships only into
  its own `gen-N/` namespace, so two live writers behind one URL (a
  botched restore, a doubled deployment) can never interleave one
  lineage. A deposed writer's shipper fail-stops permanently and
  loudly (`taguru_replication_fenced` latches, plus a `taguru::audit`
  line) while its serve path keeps answering from local truth.
  Deliberately no TTL, heartbeats, or automatic failover — the fence
  is lease-compatible (a permanent lease with TTL 0) for any future
  automation layer.

## [0.3.0] - 2026-07-18

### Added
- `TAGURU_EMBED_PARALLEL` (default 1, the prior sequential behavior):
  gloss refresh and passage refresh now dispatch each 128-item
  embedding chunk to the provider on up to `N` worker threads instead
  of one chunk at a time (#65). Both lanes persist whatever subset
  landed even when a later chunk fails, and under parallelism that
  subset is no longer necessarily a prefix of the original order — it's
  whatever completed before the first failure was recorded; the refresh
  still returns the error, so the rows it skipped stay stale for the
  next refresh to retry. Raise to match the provider's rate limit, not
  the machine's core count.
- `TAGURU_MAX_CONCURRENT_HEAVY_OPS` (default 2; `0` disables): one
  shared, non-queuing semaphore around `audit_vocabulary` and
  `compact_context`, over both raw HTTP and MCP dispatch. Once full,
  another heavy call is shed immediately as 503 `overloaded` with
  `Retry-After: 1`, leaving worker capacity available for ordinary
  requests while admitted sweeps run to their individual deadlines.
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
- `TaguruRetriever` (both LangChain packages) now addresses `contexts`
  and `groups` beside the single `context` (at least one required —
  the field is no longer mandatory alone). Across several contexts the
  graph lane runs per context (concurrently, in the async clients) and
  interleaves by per-context rank, the
  text lane rides the server's cross-context search, and every
  Document's metadata gains a `context` key naming where it came from
  (single-context retrievers too — additive).

- Single-association retraction:
  `POST /contexts/{name}/associations/retract` `{subject, label,
  object}` withdraws ONE association outright — every source's
  contribution to that edge, unsourced weight included — where
  `sources/retract` withdraws a whole document's. The surgical
  correction for a fact that should never have been asserted; a fact
  that is merely contested still wants a negative-weight assertion,
  which preserves the dispute. Names resolve through aliases; the
  answer is `{retracted, attributions_removed}` with
  `retracted: false` for a triple naming no live edge (nothing
  changed, found-nothing honesty like `sources/retract`). The edge row
  stays visible at weight 0 / count 0 until compaction sheds it
  (`activate` already skips it), re-asserting the triple later just
  works, and the write is WAL-staged like every other (write role;
  one `taguru::audit` line, since the triple lives in the body). Rides
  MCP as `retract_association` and the SDKs as
  `Context.retract_association(subject, label, object)` →
  `RetractAssociationOutcome`. On-disk note: the WAL grew a
  `retract_association` op — a binary predating it refuses a log
  holding one as corruption (the documented forward-only WAL posture).
- `GET /protocol` documents the correction split: retract what should
  never have been asserted, contest with negative weight what the
  world disputes.
- MCP parity for the backup verbs: `flush` (admin, answers the flushed
  names), `export_context` (the import batch stream as one text
  block), and `export_group` (the group's `taguru_group` line) ride
  the tool surface, mapping onto `POST /flush` and the export routes
  with their roles intact — an agent can run the documented
  flush-then-export discipline without leaving MCP. Very large
  contexts should still export over plain HTTP or `taguru export`
  offline; the tool descriptions say so.

- Integrity checksums in every on-disk format that holds acknowledged
  data (#59). The context image gains a whole-file CRC-32C footer
  (format v5 → v6), verified before anything else is trusted on load;
  the passage snapshot does the same (`TAGURUS3` → `TAGURUS4`); and
  every WAL record — graph and passages — now carries a `crc` field
  verified on replay. Structural validation alone accepts silent
  corruption that happens to keep the invariants (a flipped byte
  inside a stored name loads, serves, and flushes back as truth);
  the checksums close exactly that gap, and `taguru inspect` now says
  what was *verified* versus merely parsed (image/snapshot generation
  in each ok-line, a NOTE counting pre-checksum WAL records).
  On-disk notes: older images and snapshots keep loading forever,
  unverified, and writing always produces the checksummed formats — so
  after the first flush a DOWNGRADED binary refuses the image as an
  unsupported version (roll back onto a pre-upgrade backup, or through
  export/import). The WAL change is additive in both directions: a
  pre-checksum binary ignores the field, and pre-checksum records
  replay unchecked.
- Torn-import detection (#59): one import batch applies as four
  separately durable steps (retract the source → store the passage →
  add associations → add aliases), each store individually consistent,
  so a crash — or an unrepaired mid-batch refusal — used to leave the
  source half-applied with nothing able to say so. Now a per-source
  batch-open marker (`{stem}.{source-hash}.importing`, the pair named
  in its content) is written before the first step and removed only
  after the last: the server's next boot warns for every surviving
  marker whose context still exists (and removes moot ones), and
  `taguru inspect` reports the same tear with its repair. Both
  documented repairs clear it — re-importing the batch file (offline
  or `POST /import`; retract-then-apply keeps the retry exact) or
  retracting the source. Deleting or recreating the context sweeps its
  markers. Cross-store atomicity is deliberately not attempted:
  per-source idempotency already makes the repair exact, so detection
  was the whole remaining gap.
- Search explainability (#75): every retrieval lane can now say why an
  expected result did not appear, in one read-only call instead of
  orchestrating four endpoints with varied thresholds by hand.
  `POST /contexts/{name}/sources/search/explain` takes `{query, source,
  paragraph?, limit?}` and answers the first verdict that applies —
  `not_stored` (never stored here, or retracted; the store keeps no
  tombstone history to tell which), `paragraph_out_of_range`,
  `no_query_terms`, `no_term_overlap` (both sides' terms rendered AS
  STRINGS, so a 酒蔵-vs-酒造 spelling fork is visible on the table),
  `below_cutoff` (the actual rank, the cutoff score at the requested
  limit, and a `limit_to_reach` VERIFIED by rerunning the real serve
  computation, pool caps included), or `served` — with per-term BM25
  evidence (tf, df, idf, contribution: bit-for-bit the addends search
  summed) and the vector lane's cosine, or the named reason that lane
  never ran (off, no provider, query embedding failed, nothing
  embedded, model changed). `POST /contexts/{name}/resolve/explain`
  and `resolve_label/explain` take `{cue, expected}` plus the same
  one-call overrides resolve honors, and answer `not_in_vocabulary`
  (nearest stored spellings attached, lexical and semantic — the
  register-an-alias repair is one step away), `cue_resolved_exactly`
  (the cue IS another stored spelling; the exact tier answers alone,
  which no floor tweak can fix), `below_floor` (the actual Dice score
  vs the floor in effect — only the fuzzy tier is floor-gated, and
  the verdict honors that), `below_cutoff`, `semantic_not_run` /
  `semantic_below_floor` (whether the fallback tier joined, its gloss
  cosine vs the semantic floor, or which precondition failed), or
  `served`. All three ride MCP as `explain_search` /
  `explain_resolve` / `explain_resolve_label`. No new persistence, no
  new counters; explain shares the live scoring code paths (one term
  walker, one BM25 addend, one fusion/trim), so it cannot disagree
  with the search it explains.
- Match pagination past the 1,000-row cap (#60): `recall`/`query`/
  `unreachable_from` (single- and cross-context) and `explore` used to
  hard-truncate at `limit` (max 1000) with no way to reach whatever
  sat past it — a corpus with 5,000 matches for a cue permanently hid
  4,000 of them from every response. Each now accepts `after`, a
  keyset cursor copied verbatim from the last row of the previous
  page: `{weight, subject, label, object}` for `recall`/`query`/
  `unreachable_from`, the same plus `context` for their cross-context
  forms `POST /recall`/`POST /query` (two different target contexts
  can independently hold an edge at the identical triple, so `context`
  is the tiebreak they can't share on their own), and `{distance,
  subject, label, object}` for `explore`. `total` stays constant
  across pages — it's the population before the cursor and before
  truncation, the same convention `aliases`/`labels`/`/contexts`
  already use — so a client pages until `matches` comes back empty,
  never until `total` changes. The server never mints an opaque
  cursor; the client always derives the next `after` from the last
  item of the page it just received. Rides MCP (`recall`/`query`/
  `explore`/`audit_coverage`) and both SDKs as `MatchCursor`/
  `CrossMatchCursor`/`ExploreCursor`. Wire-visible ordering note:
  these endpoints now always sort their results (by weight or hop
  distance, then lexicographically on `(subject, label, object)` as
  the tiebreak) instead of only sorting when truncation kicked in —
  keyset pagination requires one deterministic order on every page
  whether or not a cursor is present, so a caller relying on the old
  insertion-order tiebreak under the limit will see a different order
  now.
- List filters (#62): `GET /contexts` takes `pinned: bool`;
  `GET /contexts/{name}/sources`, `/labels`, and `/aliases` take
  `prefix: string` — all filtered before `total` is computed,
  consistent with the search endpoints' "total describes the filtered
  population" contract. Chasing this down uncovered a real MCP bug:
  `query_string` silently dropped any boolean argument (`Value::Bool`
  had no case), so `pinned` on `create_context`/`update_context` was
  unreachable over MCP until now.
- MCP passthrough (#62): `compact` (admin), `get_context`, and
  `get_group` — mirroring their HTTP routes exactly — plus a new
  `import` tool (admin) that takes a `stream` (NDJSON, capped at
  32MiB by a constant local to `mcp.rs`, since `taguru-mcp` does not
  link `ingest.rs`) and an optional `dry_run`. Wiring up `stream` also
  fixed a latent MCP transport bug: a tool argument that is already a
  plain string was still being JSON-string-encoded before it reached
  the HTTP body, which escaped the NDJSON's newlines and broke the
  parser.
- `POST /import?dry_run=true` (#62): previews an import batch —
  creates, retractions, dropped passages/questions/associations —
  without writing anything, reusing the real apply path's counting
  logic wherever that is already read-only. Association/alias counts
  are the one place dry-run is necessarily optimistic: a capacity or
  version conflict cannot surface without actually writing, which the
  response and the tool description both say plainly. `taguru_group`
  records go through a separate restore path and are skipped in
  dry-run mode (omitted from the response's `groups` field).
- `retrieve` MCP tool (#62): runs the SDKs' resolve → describe →
  query/activate → cite_passage → search_passages walk server-side in
  one call, so an MCP-only client gets the same one-shot retrieval the
  Python/TypeScript SDKs already had over HTTP. Citations come back as
  `[{source, paragraph, citation}]`, since a JSON object cannot key on
  a `(source, paragraph)` tuple the way the SDKs' in-memory dicts do.
- Rename (#62): `POST /contexts/{name}/rename` and
  `POST /groups/{name}/rename` (admin role, body `{"to": "..."}`),
  plus `rename_context`/`rename_group` on both SDKs and as MCP tools.
  A context rename moves its whole file family and rewrites every
  group naming it; a group rename moves its one file and rewrites
  every OTHER group naming it as a child. Both are crash-safe the same
  way delete is: a durable marker (`.renaming` / `.grouprenaming`) is
  written before anything moves, and a boot that finds one resumes the
  file move AND the group-membership rewrite before the usual
  dangling-reference reconciliation runs — the ordering matters, since
  reconciliation has no notion of a rename in flight and would
  otherwise prune the old name as dangling instead of carrying it to
  the new one. A context rename's `from`/`to` are reserved against a
  concurrent create or another rename; a group rename runs entirely
  under the group table's single write lock instead, so no extra
  reservation is needed there.
- Compaction dead-weight visibility (#60): three counters — live dead
  edges (count fallen to zero), attributions unlinked from every chain
  but not yet reclaimed, and arena slack (bytes behind removed aliases)
  — answer "how much would `compact` reclaim right now" without
  actually running it. Tracked incrementally (retraction, alias
  removal) and seeded once at load by piggybacking on the existing
  attribution/name-table walks, so there is no extra full walk and a
  freshly compacted context starts all three back at zero. Exposed via
  `GET /contexts/{name}` and the directory (new `ContextStats` fields),
  `/metrics` (`taguru_dead_edges`, `taguru_dead_attributions`,
  `taguru_arena_slack_bytes` — server-wide sums; a context name is
  unbounded user data, so no per-context label), `taguru inspect`'s
  stats line (the same three plus the dead ratio), and `taguru
  estimate` (a new "maintenance window" line pricing a compaction's
  transient double footprint).
- `POST /maintenance/compact`: the operational counterpart to the
  visibility above — closes the server to ordinary traffic just long
  enough to drain in-flight requests and rebuild every context whose
  live dead ratio clears an optional `min_dead_ratio` (default: any
  dead weight at all), worst ratio first, then reopens. `/health`
  answers 503 `maintenance` for the duration (distinct from an actual
  fault) and new work is shed early; only one sweep runs at a time; a
  second call while one is running answers 409 rather than queuing.
  Server-wide like `/flush`: a context-scoped key is refused outright.
  Admin-only, via the existing catch-all default rather than a new
  authorization rule.
- `taguru compact --parallel N`: compacts up to `N` contexts at once
  (default 1, the prior sequential behavior), reusing the worker-pool
  pattern boot-time preload already uses. Output is reordered to the
  original argument order before printing, so stdout is byte-for-byte
  identical to `--parallel 1` regardless of `N` or thread scheduling.
- `taguru estimate` now prices passage-related memory — the passage
  store, the BM25 index, and (with `--embedding-dims`) passage vectors
  — into its `TAGURU_CACHE_BYTES` budget; before, `--passage-bytes`
  only ever showed up in the disk section. The paragraph count the
  vector estimate multiplies is capped at the same
  `DEFAULT_PASSAGE_VECTOR_LIMIT` the server enforces per context.
- `taguru extract --parallel N` (or `TAGURU_EXTRACT_PARALLEL`, the flag
  wins when both are set): runs up to `N` of one document's chunk
  completions concurrently instead of one at a time (default 1, the
  prior sequential behavior) (#64). Chunks still merge in their
  original index order, so output is byte-for-byte identical to
  `--parallel 1` regardless of `N` or thread scheduling — only
  wall-clock changes. The first chunk to fail, by index rather than by
  which thread finishes first, still fails the whole document: no
  worker claims a new chunk past the failure once it is recorded, but
  a chunk already claimed and in flight at that moment still runs to
  completion — its result is simply discarded. Parallelism never
  crosses documents — each document's
  relation-label vocabulary feeds the next document's prompt, so
  documents themselves keep extracting one at a time.
- Drift audit (#63): `unsourced weight` — an association's weight left
  over once every named source's contribution is subtracted, the same
  bucket export/re-import round trips already tag with the reserved
  source id `export:unsourced` — now surfaces at three layers instead
  of only inside exported batches. `GET /contexts/{name}` and `taguru
  inspect`'s stats line add `unsourced_edges`/`unsourced_weight`;
  `/metrics` adds `taguru_unsourced_edges`/`taguru_unsourced_weight`
  gauges (server-wide sums, same reasoning as `taguru_dead_edges`).
  `POST /contexts/{name}/drift/audit` (MCP `audit_drift`, Role::Read)
  is the new read-only verb, answering three things at once: edges
  whose unsourced weight clears an optional `unsourced_floor` (default:
  any at all), worst-first and cursor-paginated like
  `unreachable_from`; aliases whose canonical concept or label has
  gone dead (zero live edges); and, opt-in via `include_twins`, the
  same lexical/semantic fork candidates `vocabulary/audit` already
  finds, at the same `dice_floor`/`cosine_floor` defaults (0.6/0.6) —
  one shared implementation, not a second copy. `taguru estimate` is
  unaffected: synthesized associations always carry a generated
  source, so unsourced weight cannot arise there.
- Registering an alias now leaves its own `"aliases registered"`
  `taguru::audit` line (context, concept/label counts, applied count —
  never the spellings themselves), symmetric with the existing one for
  removal. Reconstructing a bad alias's live window — see the new
  "Recovering from a bad alias" note under Running in production — no
  longer depends on the removal side alone.

### Changed
- doc2query `questions` now index into their paragraph's BM25 postings
  (terms and length both — the doc2query move itself), so a
  question-shaped search lands lexically on every server; before, a
  deployment without `TAGURU_EMBED_PASSAGES` stored questions and
  ignored them for retrieval. Passage scores shift only where
  questions are attached. On-disk note: the BM25 sidecar format bumped
  (`TAGURUB1` → `TAGURUB2`, slots carry a question fold for the drift
  digest) — a derived structure, so an old sidecar rebuilds itself on
  the residency's first search, in either upgrade direction; no
  action needed.
- `aliases`/`labels`/`/contexts`'s directory paging no longer
  re-collects and re-sorts the entire namespace on every page request
  (O(n·log n) per page, O(n²·log n) to walk the whole thing) (#60):
  each context now keeps a `BTreeMap`/`BTreeSet` index alongside its
  existing storage, and the server registry does the same for
  `/contexts`, so a page is a true keyset seek — O(log n + k),
  independent of table size, for an unscoped key. A context-scoped
  key's allow-list has no relation to name order, so `/contexts`
  still sorts that (typically small, operator-configured) allow-list
  per request rather than seeking the registry — `aliases`/`labels`
  are unaffected, since they page one context's own namespace, not
  the registry. Cross-context search (`POST /recall`/
  `POST /query`, and cross-context `sources/search`) also no longer
  fans out to its target contexts with a sequential `for` loop: every
  target is now fetched concurrently, bounded by
  `TAGURU_CROSS_SEARCH_CONCURRENCY` (default 4), so one slow or cold
  context no longer blocks every context listed after it. Results and
  `total` are unchanged in both cases — only the cost/wall-clock
  improves.
- Passage vector search grows an approximate nearest-neighbor index
  past 50,000 rows in one context (#60): a hand-rolled IVF index
  (deterministic farthest-point clustering — no RNG, no external ANN
  crate, matching every other binary index in this codebase), built
  lazily on the first search past the threshold and cached for the
  store's lifetime. Below the threshold, and for any call asking for
  every row (`explain_passage_search`'s exact-ranking contract, and any
  deadline too tight to build the index), the full linear sweep still
  runs unchanged — approximation is strictly an optimization here,
  never a behavior callers need to account for.
- Three more scalability improvements (#60): the in-memory cue-
  embedding cache now evicts least-recently-used instead of FIFO;
  boot-time directory scanning parallelizes each context's disk I/O
  (sidecar read plus WAL stats) across a worker pool instead of a
  sequential loop, with results merged into the same sorted map
  regardless of arrival order; and the passage log size `/metrics`
  reports for a cold context is now cached at eviction time instead of
  re-`stat`ed on every scrape.
- `taguru extract`'s retry policy replaces the previous fixed-2-second
  sleep, 2-attempt retry with exponential backoff and full jitter (1s
  base, doubling toward a 30s ceiling; 4 attempts total — 1 initial
  plus 3 retries) (#64). A 429 response's `Retry-After` header, when
  present as delta-seconds (HTTP-date values are not recognized), is
  honored verbatim (clamped to the same 30s ceiling) instead of the
  computed backoff, since the server's own instruction beats a guess;
  other statuses are unaffected. A non-retryable 4xx
  still fails immediately, spending none of the retry budget. The
  final error message now reports how many attempts were made.

### Fixed
- `estimate`'s synthesis walked labels by round (`round % labels`), so
  the number of distinct labels actually materialized in the measured
  context was capped at the round count (`associations / concepts`),
  not at the requested `--labels`. The default shape
  (`concepts = associations / 2`) has exactly 2 rounds, so every
  default-shape run measured a context holding only 2 labels while the
  header printed the planned 50 — and an explicit `--labels N` for a
  label-rich workload was silently capped the same way, with no
  warning. The label index is now offset by the subject index
  (`(round + subject_index) % labels`), so every requested label
  appears from round 0 whenever `concepts >= labels` (every realistic
  shape, including the default); a new warning also fires on the
  residual case where `labels > concepts` and rounds are too few to
  cover them all.
- Gloss embedding refresh now prunes the vectors of concepts and labels
  the graph no longer holds. `refresh_embeddings` extended the loaded
  vector store in place, so a name dropped by compaction kept its row
  in the `.vectors.bin` sidecar forever, and `resolve`, the vocabulary
  audit's twin suggestions, and `explain` kept surfacing it. The
  sidecar is now rebuilt against the live concept and label names each
  refresh, the way the passage refresh already was.

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

[Unreleased]: https://github.com/t0k0sh1/taguru/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/t0k0sh1/taguru/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/t0k0sh1/taguru/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/t0k0sh1/taguru/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/t0k0sh1/taguru/releases/tag/v0.1.0
