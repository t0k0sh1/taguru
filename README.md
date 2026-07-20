<p align="center">
  <img src="docs/logo.png" alt="Taguru — an association graph with threads being pulled from it" width="220">
</p>

# Taguru

Long-term semantic memory for LLMs that recalls the way a thread is
pulled in hand over hand (手繰る, *taguru*). Knowledge accumulates as
(subject, relation label, object, signed weight, source) associations,
and retrieval is **structural** first: the cue is not what a question
looks like but what it is *about* — the 糸口, the end of the thread —
and the graph is walked from there to draw the knowledge out. Original
passages ride alongside as the text lane, searched per paragraph by
BM25 fused with optional paragraph embeddings — similarity serves as
one lane of evidence there, never as the primary retrieval mechanism,
and every hit says which lane found it.

The intended client is an LLM. Everything that needs language
understanding — decomposing documents into facts, choosing a context,
recomposing results into prose — is the client's job; this server only
stores and walks structure. The server distributes the complete
playbook for clients itself: `GET /protocol` (the content of
[src/llm-protocol.md](src/llm-protocol.md)).

**Documentation: <https://t0k0sh1.github.io/taguru/>** — getting
started, concepts, the import/extract references, deployment guides
(Docker Compose, Kubernetes, Amazon Bedrock), the internal
architecture, and a walkthrough of an LLM retrieving over MCP.

## Install

```sh
cargo install taguru        # the server (`taguru`) and the MCP bridge (`taguru-mcp`)

# from source
git clone https://github.com/t0k0sh1/taguru && cd taguru
cargo run --release

# Docker (linux/amd64 + linux/arm64)
docker run -d --name taguru \
  -p 127.0.0.1:8248:8248 \
  -v taguru-data:/data \
  ghcr.io/t0k0sh1/taguru:latest
```

## Try it in 30 seconds

```sh
taguru   # listens on 127.0.0.1:8248, data in ./data

# create a context
curl -X PUT localhost:8248/contexts/sake -H 'Content-Type: application/json' \
  -d '{"description":"青嶺酒造という架空の酒蔵の知識"}'

# store one association
curl -X POST localhost:8248/contexts/sake/associations -H 'Content-Type: application/json' \
  -d '[{"subject":"青嶺酒造","label":"代表銘柄","object":"青嶺","weight":1.0,"source":"第1段落"}]'

# pull the thread
curl -X POST localhost:8248/contexts/sake/activate -H 'Content-Type: application/json' \
  -d '{"origins":["青嶺酒造"]}'
```

Contexts can be bundled into **groups** (`PUT /groups/{name}`, nesting
allowed), and the searches — `POST /recall`, `POST /query`,
`POST /sources/search` — take `contexts` / `groups` lists to run one
search across several contexts at once, every match tagged with the
context it came from. Every search response also carries a `plan`:
which contexts were actually searched and — for passage search — which
lanes ran there and why not when one was skipped, with the effective
cosine floor. Deep dives (`activate`, `explore`) stay per-context:
search across, then pull the thread where it answered.

For corpus-overview questions ("what are the main themes here?") there
is a third lane: `taguru communities` detects communities on the
association graph server-side and derives an artifact of LLM summaries
— an ordinary context, incremental by content fingerprint, so an
unchanged graph re-runs without a single LLM call — and
`POST /contexts/{name}/communities/search` (MCP: `search_communities`)
ranks those summaries with an honest staleness verdict when the graph
has moved on since.

For the endpoint list and the ingest/retrieval discipline, ask the
running server: `GET /protocol`. A guided tour is
[Getting started](https://t0k0sh1.github.io/taguru/getting-started.html).

## Using it from an LLM agent (MCP)

`taguru-mcp` is an MCP stdio bridge to a running HTTP server. Agents
(Claude Code / Claude Desktop, and so on) ingest and retrieve through
it — decomposing documents into facts and composing answers out of
results is the agent's job, and the discipline rides along
automatically as the tool definitions and the MCP instructions (the
content of `/protocol`).

```sh
claude mcp add taguru -e TAGURU_URL=http://127.0.0.1:8248 -- taguru-mcp
```

With that in place, requests like "ingest the documents in this folder
into the sake context" or "tell me what you know about 青嶺酒造, with
sources" just work, as the loop directory pick → resolve →
describe/query/activate → passage lookup → cited answer. A real
round trip, request by request, is traced in the
[walkthrough](https://t0k0sh1.github.io/taguru/mcp-rag-walkthrough.html).

`taguru-mcp` also honors `TAGURU_MCP_TIMEOUT_SECS` (per-request budget
against the server, default 75 — raise it for a slow local model) and
`TAGURU_MCP_MAX_LINE_BYTES` (stdio frame cap).

The same tools are also served remotely: `POST /mcp` speaks the MCP
Streamable HTTP transport (stateless profile — plain JSON responses,
no session to manage), behind the same bearer token as the rest of the
API. Any remote-capable MCP client connects with just a URL:

```sh
claude mcp add --transport http taguru https://your-host/mcp \
  --header "Authorization: Bearer $TAGURU_API_TOKEN"
# Claude API: mcp_servers = [{type: "url", url: "https://your-host/mcp",
#                             name: "taguru", authorization_token: "…"}]
```

claude.ai custom connectors (web and mobile) authenticate with OAuth
instead of a pasted header: set `TAGURU_PUBLIC_URL`, point the
connector at `https://your-host/mcp`, and approve the consent page by
pasting one of your API keys — the connection then acts as that key.
Discovery, dynamic registration, and PKCE are built in; no external
identity provider is involved.

Expose it beyond localhost only behind TLS (a reverse proxy, as with
the rest of the API), and remember a token is the whole credential —
whoever holds one holds the memory. Mint one named key per client
(`TAGURU_API_TOKENS`) so a leak is one revocation, not a rotation of
everything.

Running the agent or the embeddings on Amazon Bedrock? See the
[Bedrock guide](https://t0k0sh1.github.io/taguru/bedrock.html): the
Converse hosting pattern for taguru-mcp, an InvokeModel embedding
proxy, floor calibration, and the access-gate diagnosis commands.

## Loading knowledge in bulk

Initial loads and migrations skip HTTP entirely: `taguru import
FILE|DIR...` applies JSONL batch files straight to `TAGURU_DATA_DIR`
through the same WAL-staged write path the server uses. One batch
states one **source**'s complete truth — import retracts the source,
then applies the batch — so re-importing is idempotent and a revised
file replaces cleanly instead of double-counting weights.

```jsonl
{"taguru_batch": 1, "context": "sake", "source": "docs/aomine.md", "create": {"description": "酒蔵の知識"}}
{"passage": "青嶺酒造は1907年創業。杜氏は高瀬。"}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0}
{"alias": "Aomine Brewery", "canonical": "青嶺酒造", "kind": "concept"}
```

A **running** server takes the same contract at `POST /import` (one
request = one batch file or stream, same validation, same
replace-a-source semantics), so live systems bulk-load without a
downtime window:

```sh
curl -X POST localhost:8248/import -H 'Authorization: Bearer <key>' \
  --data-binary @docs-aomine.jsonl   # --data-binary: -d strips the newlines
```

Where do batch files come from? Any pipeline that speaks the format —
or the packaged producer: `taguru extract` reads `.md`/`.txt`
documents, has any OpenAI-compatible chat model decompose each into
associations under the /protocol discipline, and writes one batch
file per document, ready for either import entrance:

```sh
TAGURU_EXTRACT_URL=https://api.openai.com/v1/chat/completions \
TAGURU_EXTRACT_MODEL=gpt-4.1 TAGURU_EXTRACT_API_KEY=$KEY \
taguru extract --context sake --description "酒蔵の知識" --out batches/ docs/
taguru import batches/
```

The server never holds model credentials; `TAGURU_EXTRACT_*` lives in
the offline producer's environment only, and local or bridged models
(Ollama, LiteLLM in front of Bedrock) work the same way. Full
contracts:
[batch import](https://t0k0sh1.github.io/taguru/import.html) ·
[document extraction](https://t0k0sh1.github.io/taguru/extract.html).

## Configuration

Everything is environment variables. `taguru --help` lists every one;
`taguru --config taguru.env` reads the same `KEY=VALUE` file
`docker run --env-file` accepts, so one file drives both. The
load-bearing ones:

| Variable | Default | Role |
|---|---|---|
| `TAGURU_ADDR` | `127.0.0.1:8248` | Bind address ("TAGU" on a phone keypad) |
| `TAGURU_DATA_DIR` | `./data` | Data directory |
| `TAGURU_API_TOKEN` | — | Bearer token (everything but `/health`, `/live`, `/metrics`). Unset = unauthenticated, localhost only |
| `TAGURU_API_TOKENS` | — | Named keys (`"ci:tokA,laptop:tokB"`): the access log says which key, a leak costs one revocation |
| `TAGURU_KEY_SCOPES` | — | Per-key grants as one JSON object: roles `read` ⊂ `write` ⊂ `admin`, optionally restricted to named contexts. These three are the auth table, and it hot-reloads — SIGHUP, or just editing the `--config` file — so rotation never costs a restart (see production notes) |
| `TAGURU_WAL` | on | fsync every acknowledged write before applying it — a crash loses nothing |
| `TAGURU_REPLICATE_URL` | — | Continuous replication to object storage (`s3://` / `gs://` / `az://` / `file://`), epoch-fenced; restore with `taguru restore`, or boot an empty directory straight from the bucket. Unset = off |
| `TAGURU_TAKEOVER` | off | `1` (or `serve --take-over`) acknowledges deposing the bucket's newest writer while it still looks alive — starting a writer against a bucket IS the promotion act |
| `TAGURU_REPLICA` | off | `1` (or `serve --replica`) serves the bucket lineage read-only, tailing it continuously: reads scale across replicas, writes answer 403 `read_only_replica` naming the writer, per-context lag on `/metrics` |
| `TAGURU_WRITER_URL` | — | Where a replica's write-refusal points clients (the writer's base URL / LB name); unset = the refusal names only the bucket's fence holder |
| `TAGURU_ROUTE_MAP` | — | `taguru route` only: the context→shard map file (`context = shard-url` per line, optional `* = shard-url` fallback); edits take a router restart |
| `TAGURU_CACHE_BYTES` | 512 MiB | Resident budget for unpinned contexts (LRU eviction) |
| `TAGURU_RETRIEVAL_CACHE_BYTES` | 32 MiB | Exact-match result cache for recall/query/passage search — an identical request against an unchanged corpus answers without re-running the search; invalidated by the revision counters (`0` = off) |
| `TAGURU_SEMANTIC_CACHE_THRESHOLD` | unset (off) | Semantic tier over the exact cache, passage search only: a paraphrased query whose embedding cosine clears this floor (`[0,1]`; start at `0.94`) AND passes a negation/number/entity guard serves the equivalent earlier query's cached result. Needs the exact cache and `TAGURU_EMBED_PASSAGES` |
| `TAGURU_EMBED_URL` / `_MODEL` / `_API_KEY` | — | Semantic entry tier (OpenAI-compatible `/embeddings`); unset keeps the entrance purely lexical |
| `TAGURU_EMBED_AUTO` | off | Re-embed changes with each flush — recommended whenever agents drive the ingest |
| `TAGURU_EMBED_PASSAGES` | off | Also embed paragraphs (the semantic lane of passage search); a corpus is much larger than its glosses, so the spend is opt-in |
| `TAGURU_SEMANTIC_FLOOR` | 0.35 | Floor for the semantic tier — a property of the embedding model; `taguru calibrate --context NAME --probes FILE` measures it against a running server whenever you switch |
| `TAGURU_PUBLIC_URL` | — | Public base URL; enables OAuth on `/mcp` for claude.ai custom connectors |
| `TAGURU_RATE_LIMIT_PER_MIN` | 0 (off) | Per-key request budget — turn on whenever the server leaves localhost |
| `TAGURU_REQUEST_TIMEOUT_SECS` | 30 | Per-request budget; raise it when an embedding provider is configured |
| `TAGURU_MAX_CONCURRENT_HEAVY_OPS` | 2 | Shared ceiling for vocabulary audits and context compactions; excess calls get 503 + `Retry-After` (`0` disables) |
| `TAGURU_AUTO_COMPACT` | on | Ratio-triggered auto-compaction: each flush tick rebuilds at most the one worst context whose dead ratio exceeds `TAGURU_AUTO_COMPACT_RATIO` (0.5 — dead weight outgrew live content), behind the heavy-ops ceiling; `0` keeps compaction manual-only |
| `TAGURU_CONTEXT_QUOTAS` | — | Per-context ceilings as one JSON object (`{"sake": {"storage_bytes": …, "cache_bytes": …}}`): `storage_bytes` refuses growth writes at the ceiling with 507 `storage_full` (retract/compact/delete stay open), `cache_bytes` bounds the context's resident share — under cache pressure the over-share context is evicted first, so one hot tenant cannot evict the rest beyond its ceiling. A broken declaration refuses boot, like broken credentials |

The full table — durability ceilings, observability (`RUST_LOG`,
`TAGURU_LOG_FORMAT=json`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`TAGURU_LOG_SEARCHES`, per-context capacity gauges via
`TAGURU_METRICS_PER_CONTEXT`), body/concurrency caps — is in
[Getting started](https://t0k0sh1.github.io/taguru/getting-started.html)
and [Internal architecture](https://t0k0sh1.github.io/taguru/architecture.html).

## Running in production

- **The image is the server alone on `scratch`** — a ~13 MB static
  binary, no shell, no libc, nothing to patch. It runs as uid 65532,
  works with `--read-only` (/data is the only write target), stops
  gracefully (flush + usage sweep), and health-checks itself
  (`taguru health` — there is no curl on scratch).
  `docker build -t taguru .` builds the same image locally.
- **Verify what you pull.** Releases are signed (Sigstore keyless)
  and carry a real SBOM — the binary is built with `cargo auditable`,
  so every crate version rides in the executable itself — plus SLSA
  provenance naming the exact CI run; CI re-verifies all three from a
  clean runner on every release. The commands, and why the signing
  identity must be pinned exactly, are in
  [SECURITY.md](SECURITY.md#verifying-a-release); prefer a digest pin
  in deploys ([deploy/README.md](deploy/README.md)).
- **Linux is what's tested.** CI runs on `ubuntu-latest` and
  `ubuntu-24.04-arm` only, matching what's released — crates.io source
  and the Linux container image above, no Windows binaries. A few call
  sites are unix-specific (owner-only file creation, a durability
  fsync, SIGTERM-triggered graceful shutdown) and fall back to
  weaker-but-working behavior elsewhere (`#[cfg(not(unix))]`); macOS
  takes the unix code path but isn't covered by CI. Details:
  [CONTRIBUTING.md](CONTRIBUTING.md#platform-support).
- **One writer per data directory.** The directory admits one taguru
  process at a time (serve or import) via an advisory lock — dependable
  on local disks, *not* on NFS/EFS. Deploys are stop-then-start. Scale
  reads with replicas (below); scale writes by giving independent
  shards disjoint sets of contexts and putting `taguru route` in front
  (below) — contexts no longer need to cohabit to be searched or
  grouped together.
- **Key rotation never costs a restart.** The auth table —
  `TAGURU_API_TOKEN(S)`, `TAGURU_KEY_SCOPES` — hot-reloads on the
  running server: send SIGHUP, or (with `--config FILE`) just rewrite
  the file and the ~5s watch picks it up. Fail closed: a malformed
  edit keeps the previous table armed (loud log line +
  `taguru_keyring_reload_refusals_total` — alert on it, or a rotation
  you believe you performed silently never took), and a reload can
  never transition "tokens configured" → "no tokens"; disarming auth
  stays a restart-only decision. Each reload leaves one
  `taguru::audit` line naming the keys added / removed / rotated /
  rescoped — names only, never token bytes. OAuth delegations minted
  from a removed key die with it, per-key rate-limit budgets carry
  over untouched, and in-flight requests judge against the old table
  or the new one, never a mix. On Kubernetes, mount the keys Secret
  as a volume and pass `--config`: `kubectl apply` a rotated Secret
  and the kubelet's atomic symlink swap reaches the watch by itself —
  no exec into the pod, no signal, no restart
  ([deploy/kubernetes.yaml](deploy/kubernetes.yaml) shows the
  wiring). Everything else — bind address, data dir, budgets — stays
  boot-time on purpose.
- **Read replicas; availability is promotion time.** `serve --replica`
  (or `TAGURU_REPLICA=1`) against the same `TAGURU_REPLICATE_URL`
  serves the bucket lineage read-only and keeps tailing it: every
  retrieval verb works from the replica's own copy (reads scale
  ~linearly with the pool), every write answers 403
  `read_only_replica` naming the writer (`TAGURU_WRITER_URL`), and
  each context is consistent at its applied watermark — staleness ≤
  shipping lag + poll interval, cross-context skew possible, all of it
  on `/metrics` (`taguru_replica_applied_seq` vs `_shipped_seq`,
  `_behind_seconds`). A replica doubles as the warm standby: losing
  the writer costs the **manual promotion** (drain the lag metric,
  start a writer against the bucket with `--take-over` if the old one
  crashed, flip the name — replicas re-aim at the new generation by
  themselves) plus the dead writer's un-shipped tail, which those
  metrics had on display. No leases, no auto-failover — by design; the
  runbook lives in the
  [architecture page](https://t0k0sh1.github.io/taguru/architecture.html#replicas)
  and is rehearsed by an integration test.
- **Sharding, with one front door.** `taguru route` is a stateless
  scatter-gather router over sharded instances: `TAGURU_ROUTE_MAP`
  names a file of `context = shard-url` lines (plus an optional
  `* = shard-url` fallback), context verbs proxy to the owning shard,
  and cross-context `recall`/`query`/`sources/search` and groups span
  every shard with the exact single-instance merge semantics — the
  equivalence is pinned by an integration test, cursors included
  (`after` anchors on the last match itself, so it forwards to every
  shard verbatim). Groups exist on every shard with members projected
  by the map; `/import` splits its batch stream by context and
  dry-run-preflights batch chunks and projected group records alike,
  so a stream one instance would refuse with nothing applied is
  refused the same way here; `/mcp` works unchanged. No data directory, no state — run any number of routers
  behind one LB. Auth passes through: shards enforce keys and scopes
  (keep their keyrings identical), the router holds none. A shard that
  answers an error fails the request whole; a shard that cannot be
  *reached* degrades fan-out reads to labeled partials (`unreached`
  in the envelope) and refuses routed verbs with 502
  `shard_unreachable`. Moving a context, in order: quiesce its
  writes → `taguru export` → DELETE it through the router (the old
  shard also sweeps its group projections) → edit the map + rolling
  router restart → re-import through the router, which now routes it
  to the new shard. Delete before re-import — a leftover copy keeps
  answering the old shard's slice of every group fan-out.
- **Health and metrics.** `GET /health` is readiness (503 while the
  write path is degraded — route away, don't restart), `GET /live` is
  liveness, `GET /metrics` is Prometheus text.
  `TAGURU_METRICS_PER_CONTEXT=all` (or top-`N` by disk size) adds
  per-context capacity gauges — disk bytes by file family, resident
  bytes, pinned, counts — measured at flush time, never by the scrape.
  Every request lands in the access log with its key, context, and
  latency; destructive operations additionally leave one
  self-contained `taguru::audit` line. Distributed tracing is opt-in
  via `OTEL_EXPORTER_OTLP_ENDPOINT`.
- **Backups.** Set `TAGURU_REPLICATE_URL` and the server continuously
  ships every file family — both log lanes tailed record-by-record,
  published files whole — to object storage (S3/GCS/Azure, credentials
  via each cloud's default chain): RPO becomes seconds of shipping lag
  (per-lane at `/metrics`) instead of a snapshot interval, DR degrades
  to bucket cross-region replication, and the bucket is epoch-fenced so
  a second accidental writer fail-stops loudly instead of corrupting
  the lineage. Recover with `taguru restore --out DIR` — or start a
  server on an **empty** directory with the same URL and it boots from
  the bucket directly: shared files and pinned contexts hydrate before
  the port opens, everything else on first touch, and local files that
  already match are reused without a download. That makes the volume a
  cache and recovery "start anywhere"; while the previous writer still
  looks alive, the boot demands `--take-over` / `TAGURU_TAKEOVER=1`
  first. Point-in-time alternatives, unchanged: `POST /flush` then
  snapshot the data directory (every writer is fsync+rename, so
  filesystem snapshots are safe at any instant; back up each context's
  file family as a set) — or take the portable JSONL stream with
  `taguru export` and restore anywhere through `taguru import` /
  `POST /import`. Verify any of them with `taguru inspect` — images,
  passage snapshots, and WAL records carry CRC-32C checksums, so "ok"
  means the bytes were proven intact, not just parseable.
  Revision-heavy contexts reclaim themselves: once a context's dead
  ratio passes `TAGURU_AUTO_COMPACT_RATIO`, the flusher rebuilds it on
  an upcoming tick — worst ratio first, one context per tick, as the
  heavy-ops ceiling allows (audit line +
  `taguru_auto_compactions_total` on `/metrics`). `taguru compact` and `POST /contexts/{name}/compact`
  remain for opted-out deployments (`TAGURU_AUTO_COMPACT=0`) and
  scheduled quiet-window sweeps; size targets with `taguru estimate`.
- **Recovering from a bad alias.** Alias registration takes effect
  immediately and resolves that spelling on every subsequent write — a
  wrong alias silently pulls all matching ingestion onto the wrong
  canonical from that moment on. `DELETE /contexts/{name}/aliases`
  only stops new contamination; it does not revisit associations
  already interned under the bad spelling, and there is no alias-node
  concept to enumerate them after the fact. To recover: delete the
  alias first so nothing more lands on it, then bound its live window
  by cross-referencing the WAL's `AliasConcept`/`AliasLabel` …
  `UnaliasConcept`/`UnaliasLabel` op order, the `aliases registered`/
  `aliases removed` `taguru::audit` lines (who, when, which
  namespace), and the general access log (every request against the
  context, by key and timestamp) — together they say precisely when
  the alias existed. Cross-reference that window against your own
  ingestion record (or the `source retracted`/`import batch applied`
  audit lines, if the affected sources were re-imported rather than
  freshly asserted) to find which sources landed under the bad
  spelling, then retract each (`POST /contexts/{name}/sources/
  retract`) and re-import it under the correct one. There is no merge
  for two canonicals that already diverged before the alias was
  caught — unify them the way `compact` itself rebuilds a context:
  export both sides and re-import everything under the one you keep.

Worked deployments, probe wiring, and the reasoning behind them:
[Docker Compose](https://t0k0sh1.github.io/taguru/docker-compose.html) ·
[Kubernetes](https://t0k0sh1.github.io/taguru/kubernetes.html) —
the manifests themselves are in [deploy/](deploy/).

## SDKs and examples

Official clients live in [sdk/](sdk/): `taguru` for Python and for
TypeScript (one identical surface — typed models, retries, pagination,
batched writes, the `retrieve()` loop), and `langchain-taguru` for
Python and JS/TS (a retriever that merges both lanes, and an ingester
that decomposes LangChain Documents into the graph).

```sh
pip install taguru        # or: pip install langchain-taguru
npm install taguru        # or: npm install langchain-taguru
```

Library-level examples live in [examples/](examples/) — one directory
per example, each with its own README. Runnable LangChain use cases
(RAG QA with citations, governed ingestion, conversational long-term
memory) live in [examples/langchain/](examples/langchain/), each as a
Python and a TypeScript program mirrored line for line.

## License

MIT — see [LICENSE](LICENSE).
