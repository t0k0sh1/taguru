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

## Architecture

- **Library** (`src/context.rs`) — one `Context` = one 文脈 (one
  context of meaning). All state fits in a flat buffer — a UTF-8
  string arena plus seven tables of fixed-width `#[repr(C)]` records —
  and the adjacency lists are intrusive chains threaded through the
  edge records. Every mutation is an append or a field update, so the
  whole state round-trips as one image through `to_bytes` /
  `from_bytes` (little-endian, fully validated on load). Capacity is
  the u32 space (~4.29 billion records per table, 4 GiB of interned
  text); a write past it is a `ContextFull` error, not a panic.
  - Reads: `recall` / `query` / `query_any` / `describe` / `explore` /
    `activate` / `resolve` / `resolve_label` / `unreachable_from`
  - Writes: `associate` / `associate_from` / `add_concept_alias` /
    `add_label_alias`
  - The retrieval entry is normalization (NFKC, case folding,
    katakana → hiragana) plus a bigram inverted index, absorbing
    spelling variation and light typos. Aliases are entry-only
    alternative spellings (results always carry the canonical
    spelling).
- **Server** (`src/main.rs`, `src/registry.rs`, `src/api.rs`) — disk
  is the source of truth; memory is a cache managed at whole-context
  granularity. Each context is persisted as `{name}.ctx` (the image) +
  `{name}.meta.json` (description, pinning, stats) +
  `{name}.passages.bin` / `{name}.passages.wal.jsonl` (original
  passages: a compacted snapshot plus an append-log, fsynced per batch
  — a pre-migration `{name}.sources.json` is read once and retired).
  Boot registers every context cold (pinned ones preload), the first
  access loads transparently, least-recently-used contexts are evicted
  past the cache budget, and writes mark a context dirty — persisted by
  the periodic flusher, on eviction, and on shutdown.

## Running

```sh
cargo run --release   # or `cargo install taguru`, which installs the
                      # server (`taguru`) and the MCP bridge (`taguru-mcp`)
# Same knobs from a file: `taguru --config taguru.env` (or
# TAGURU_CONFIG=taguru.env) reads KEY=VALUE lines — the exact dialect
# `docker run --env-file` accepts, so ONE file drives both. Real
# environment variables win over the file, and unknown TAGURU_* keys
# in it are flagged as probable typos. `taguru --help` lists every
# variable; `taguru version` prints the version.
# Environment:
#   TAGURU_ADDR         bind address (default 127.0.0.1:8248 — "TAGU" on a
#                     phone keypad, chosen to avoid the defaults of likely
#                     neighbours: FastAPI/Chroma 8000, Next.js/Grafana 3000,
#                     Ollama 11434, Qdrant 6333, Weaviate 8080)
#   TAGURU_DATA_DIR     data directory (default ./data)
#   TAGURU_CACHE_BYTES  resident budget for unpinned contexts (default 512 MiB)
#   TAGURU_FLUSH_SECS   image flush interval (default 5). With the WAL on
#                     this is freshness cadence, not a loss window.
#   TAGURU_WAL          per-context write-ahead log: every acknowledged
#                     graph write is fsynced before it applies, so a
#                     crash loses nothing (default on; 0/false restores
#                     the flush-interval loss window)
#   TAGURU_WAL_MAX_BYTES  per-context WAL ceiling (default 256 MiB,
#                     0 = unlimited). The log truncates after every
#                     successful image flush, so it only nears the cap
#                     when flushes keep failing — past it, writes are
#                     refused (500) instead of growing the log forever.
#                     Watch taguru_wal_bytes.
#   TAGURU_PASSAGES_WAL_MAX_BYTES  backstop for each context's passage
#                     log (default 1 GiB, 0 = unlimited). Passages ride
#                     their own always-on log + compacted snapshot; the
#                     log legitimately grows to about the snapshot's
#                     size before compaction, so this refuses stores
#                     only when the log is ALSO past 2× the snapshot —
#                     i.e. compaction itself is failing. Watch
#                     taguru_passages_wal_bytes.
#   TAGURU_EMBED_URL / TAGURU_EMBED_MODEL / TAGURU_EMBED_API_KEY
#                     semantic entry tier (OpenAI-compatible /embeddings).
#                     Unset keeps the entry purely lexical.
#                     e.g. URL=https://api.openai.com/v1/embeddings
#                          MODEL=text-embedding-3-large  (3-small separates
#                          short Japanese names too poorly; the default floor
#                          0.35 is calibrated for 3-large + glosses)
#                          API_KEY=$OPENAI_API_KEY
#                     Every request carries X-Taguru-Embed-Purpose:
#                     "index" (gloss refresh) or "query" (cue resolve), so
#                     a proxy in front of an asymmetric model (Cohere,
#                     Voyage: input_type / prefixes) can encode each side
#                     correctly. Plain OpenAI servers ignore the header.
#   TAGURU_EMBED_AUTO=1 refresh embeddings incrementally with each flush
#                     (opt-in; unset means manual POST /embeddings/refresh only).
#                     Recommended whenever agents drive the ingest: an agent
#                     cannot be counted on to call refresh_embeddings, and
#                     GET /protocol tells connecting agents which mode this
#                     server runs.
#   TAGURU_EMBED_PASSAGES=1 also embed stored paragraphs — the semantic
#                     lane of passage search, for question-shaped queries
#                     landing on answer-shaped text. Opt-in on top of the
#                     provider: a corpus is orders of magnitude more text
#                     than its glosses, so the spend must be chosen.
#                     Budget it as paragraphs × dimensions × 4 bytes of
#                     resident vectors per context, and raise
#                     TAGURU_CACHE_BYTES to keep the whole working set in.
#   TAGURU_PASSAGE_VECTOR_LIMIT  max embedded rows per context (default
#                     20000 ≈ 120 MiB at 1536 dims). Paragraphs and
#                     doc2query questions (`taguru extract --questions`)
#                     each spend one row. Past it the lexical lane still
#                     serves every paragraph; only the semantic lane goes
#                     partial (the refresh response reports the skips).
#   TAGURU_SEMANTIC_FLOOR  server default for the semantic entry floor
#                     (default 0.35, calibrated for text-embedding-3-large).
#                     The right value is a property of the EMBEDDING MODEL —
#                     each puts true matches and noise on different cosine
#                     bands. Calibrate by probing resolve with
#                     {"semantic_floor": 0.05} and setting the floor between
#                     the noise band and the true matches:
#                       text-embedding-3-large            0.35 (the default)
#                       amazon.titan-embed-text-v2 (512d) ~0.2 (Japanese:
#                         true matches 0.2–0.3, noise ~0.15)
#                     Per-context settings and per-call overrides still win.
#   RUST_LOG            log filter (default info), EnvFilter syntax
#   TAGURU_LOG_FORMAT   json for one JSON object per log line (default: pretty).
#                     Logs go to stderr.
#   TAGURU_LOG_SEARCHES=1 one taguru::search event line per retrieval
#                     (context, op, cue, hits) — the raw material for
#                     keyword analysis in your log pipeline. Off by
#                     default: cues are memory CONTENT, and the log
#                     stream carries none unless you opt in.
#   OTEL_EXPORTER_OTLP_ENDPOINT
#                     turns on OTLP/HTTP span export (e.g. the collector
#                     sidecar http://localhost:4318). The other standard
#                     OTEL_* variables apply — OTEL_SERVICE_NAME (default
#                     taguru), OTEL_EXPORTER_OTLP_HEADERS, batch cadence.
#                     Unset = no tracing, byte-identical logs.
#   TAGURU_API_TOKEN    bearer token required on everything but /health,
#                     /live, and /metrics. Unset = UNAUTHENTICATED
#                     (localhost only).
#                     The MCP bridge reads the same variable.
#   TAGURU_API_TOKENS   named bearer keys ("ci:tokA,laptop:tokB"), accepted
#                     alongside TAGURU_API_TOKEN (whose key name is
#                     "default"). The access log then says WHICH key made
#                     each request (key=...), one leaked key is one
#                     revocation instead of a total rotation, and rotation
#                     itself is an overlap: add the new key, move callers,
#                     drop the old. Malformed entries refuse to boot.
#   TAGURU_KEY_SCOPES   per-key grants as one JSON object, by key name:
#                     {"ci": "read", "laptop": "admin",
#                      "bot": {"role": "write", "contexts": ["sake"]}}
#                     Roles nest: read (the retrieval loop) ⊂ write
#                     (+ the ingest loop: create contexts, assert, store
#                     passages, aliases, retract-and-resync sources,
#                     refresh embeddings) ⊂ admin (+ delete contexts,
#                     /import, /flush). "contexts" restricts a key to
#                     those contexts (GET /contexts shows it only its
#                     grant; /import bodies are checked batch by batch);
#                     omit it for every context. A key the object does
#                     not name keeps the historical full grant — admin,
#                     everywhere — so setting the variable changes only
#                     the keys it names. Grants hold identically over
#                     raw HTTP and MCP tool calls (each dispatched tool
#                     call is judged as the route it lands on), and an
#                     OAuth connection is capped by the key it
#                     delegates. Out of range: 403 in the error shape.
#                     Malformed JSON or unknown key names refuse to boot.
#   TAGURU_PUBLIC_URL   the server's public base URL (e.g.
#                     https://memory.example.com). Setting it enables OAuth
#                     on the remote MCP endpoint for clients that insist on
#                     it (claude.ai custom connectors): they discover the
#                     flow (RFC 9728), register themselves, and land on a
#                     consent page where you paste an EXISTING API key —
#                     the connection then acts as that key, "key@client" in
#                     the access log and the rate limits. OAuth tokens open
#                     /mcp only; the raw API stays key-only. Revoke by
#                     removing the grant from data_dir/oauth.json (restart);
#                     idle grants expire after 30 days. Requires
#                     TAGURU_API_TOKEN(S) — consent delegates a real key.
#   TAGURU_MAX_BODY_BYTES      request body cap (default 8 MiB)
#   TAGURU_REQUEST_TIMEOUT_SECS  per-request budget (default 30; raise
#                     above TAGURU_EMBED_TIMEOUT_SECS when TAGURU_EMBED_URL
#                     is configured — the provider round trip cannot be
#                     preempted mid-call, and the boot log warns when the
#                     two are misordered)
#   TAGURU_EMBED_TIMEOUT_SECS  the provider's per-attempt budget (default
#                     60). Transient failures — dropped connections,
#                     timeouts, 429s, 5xx — retry twice with a short
#                     backoff before the error surfaces; 4xx refusals and
#                     malformed responses fail immediately. Watch
#                     taguru_embedding_duration_seconds (histogram,
#                     retries included) next to the ok/failed counters:
#                     the counters say THAT the provider misbehaves, the
#                     histogram says how slowly.
#   TAGURU_RATE_LIMIT_PER_MIN  per-key request budget (default 0 = off):
#                     each named key may burst a full minute's allowance,
#                     then settles to the sustained rate; unauthenticated
#                     callers (and OAuth discovery/grant traffic) are
#                     bucketed per source IP. Past it: 429 in the error
#                     shape with Retry-After (seconds); /health and
#                     /metrics stay exempt. Turn this on whenever the
#                     server leaves localhost.
#   TAGURU_AUTH_FAIL_LIMIT_PER_MIN  failed bearer attempts per source IP
#                     before a 429 (default 10; 0 = off). Bounds
#                     brute-force and the CPU it would burn; a valid
#                     token is never throttled. Behind a reverse proxy
#                     every caller shares the proxy's IP, so this (and
#                     the anonymous rate bucket above) collapse to one
#                     bucket — throttle at the proxy there. taguru does
#                     not trust X-Forwarded-For.
#   TAGURU_MAX_CONCURRENT_REQUESTS  in-flight ceiling (default 256; 0 = off).
#                     Past it new requests are SHED — an immediate 503
#                     with Retry-After, before auth even runs — instead
#                     of queueing until everything times out. /health
#                     and /metrics stay exempt so probes see the
#                     overload rather than joining it. Watch
#                     taguru_inflight_requests (gauge) and
#                     taguru_requests_shed_total (counter).
```

### Docker

The published image is the server alone on `scratch`: a ~13 MB static
binary — no shell, no libc, no CA bundle (TLS roots are compiled in),
nothing to patch or scan.

```sh
docker run -d --name taguru \
  -p 127.0.0.1:8248:8248 \
  -v taguru-data:/data \
  ghcr.io/t0k0sh1/taguru:latest
```

Inside the container the server binds 0.0.0.0 (loopback would be
unreachable through `-p`), so UNAUTHENTICATED mode reaches as far as
the port is published — keep `-p` scoped to `127.0.0.1`, or set
`TAGURU_API_TOKEN`. Configuration is the same environment variables
(`-e`, or `--env-file taguru.env` — the very file `--config` reads).
The process runs as uid 65532: a named volume inherits that ownership
automatically, a bind mount needs `chown -R 65532` first (or run with
`--user "$(id -u)"`). `--read-only` works (/data is the only write
target), `docker stop` is a graceful shutdown (flush + usage sweep),
and the built-in HEALTHCHECK is `taguru health` — the binary probing
itself. Backups verify without a local toolchain:

```sh
docker run --rm -v taguru-data:/data ghcr.io/t0k0sh1/taguru inspect /data
```

`docker build -t taguru .` builds the same image locally; releases
publish linux/amd64 + linux/arm64 to GHCR on version tags.

Observability: every request lands in the access log — method, route
template, the CONTEXT the request addressed, status, key, latency —
and each destructive operation (context delete, source retract, alias
removal, import batches) additionally leaves one self-contained
`taguru::audit` line naming who did what to which object, body-carried
objects included; "which contexts did this key delete" is one grep.
`GET /metrics` serves Prometheus text — per-route request counts and
latency histograms, cache/flush/WAL/embedding outcomes, a 500-cause
breakdown (`taguru_errors_total{kind=...}`), retrieval hit/empty
counts per operation (`taguru_searches_total`) with a resolve-tier
split (`taguru_resolves_total` — a rising `semantic` share means cues
are drifting from the stored vocabulary), residency and log-size
gauges (the graph WAL and the passage log each have their own), and
the last-successful-flush timestamp. Per-context numbers
deliberately stay OUT of the metric labels (context names are
client-minted, so they would mint unbounded series): the routing
directory (`GET /contexts`) carries them instead — usage counters
beside the graph stats (reads, empty reads, writes, last-read/write
times), separating contexts nobody chooses from contexts that get
chosen but answer nothing. Usage counters are advisory: they persist
with each image flush and once at graceful shutdown, so a crash loses
the tail; reads never cause disk writes.
Distributed tracing is opt-in: with `OTEL_EXPORTER_OTLP_ENDPOINT` set,
every request becomes an OTLP span that joins the trace an inbound
W3C `traceparent` or AWS `X-Amzn-Trace-Id` (ALB / API Gateway)
started, the embedding-provider round trip shows up as a child span,
and the access log carries the `trace_id` so logs and traces
cross-reference. Any OTLP backend works — an OpenTelemetry Collector,
the Datadog Agent, ADOT for X-Ray, Jaeger, Tempo — and switching is
collector configuration, not a Taguru change.
`GET /health` answers `200 ok` while the write path is healthy and
`503` (JSON error shape) while the most recent image flush has failed
— it recovers by itself one flush interval after the disk does. That
is a READINESS signal: stop routing traffic while the disk is bad,
resume when it heals. Liveness is `GET /live` — `200` for as long as
the process answers at all — because restarting on a disk stall fixes
no disk and re-pays the pinned preload. On Kubernetes:
`livenessProbe: /live`, `readinessProbe: /health`, and a
`startupProbe` on `/live` sized for the preload (the port opens only
after pinned contexts load — now in parallel — so "connection
refused" simply means "still starting").

Backups: one context is the whole file family — `{stem}.ctx`,
`.meta.json`, `.passages.bin`, `.passages.wal.jsonl`, `.bm25.bin`,
`.vectors.bin`, `.wal.jsonl` (plus a legacy `.sources.json` until the
first passage compaction retires it) — back them up together, never
partially. Every writer is fsync + rename, so
a filesystem-level point-in-time snapshot (ZFS/Btrfs/LVM) of the data
directory is safe at any moment; a file-by-file copy of a *running*
server (plain `rsync`/`cp`) is not guaranteed consistent across files
— stop the server or use a real snapshot. `POST /flush` persists every
dirty context on demand (and answers with their names), so images are
current before the snapshot is taken. Verify a backup offline with
`taguru inspect /path/to/data`: every image goes through the same
fully validating load the server boots with, every WAL through the
same replay parser, with per-context stats — nonzero exit means
something holding acknowledged data is corrupt.

The *portable* backup is `taguru export --out DIR` (offline; no
CONTEXT arguments means every context), or `GET /contexts/{name}/export`
on a running server: each context comes back as the very JSONL batch
stream `taguru import` and `POST /import` apply — one batch per
source, the create block on the first, aliases on the last. Unlike
the file family it is version-independent plain text, restores
without downtime through `POST /import`, migrates between servers,
and diffs cleanly (the rendering is deterministic). Re-importing a
stream is idempotent — each batch replaces its own source — so a
restore over live data is a per-source sync; delete the context
first when the result must equal the snapshot exactly. Weight from
sourceless writes rides a reserved `export:unsourced` batch, and
fully retracted (weight-zero) edges are shed on the way out, so an
export → import round trip also compacts the graph.

The image is append-only by design — retraction unlinks records but
never reclaims them — so a context with heavy revision traffic grows
monotonically. `taguru compact` (offline) and
`POST /contexts/{name}/compact` (live; admin role, that context's
requests wait out the rebuild) rewrite each image from its live
content alone and report what was shed; every fact, count, and
paragraph locator survives.

The data directory admits one taguru process at a time — a serve or
an import — via an advisory lock (`.taguru.lock`); the second comer
is refused with a message naming the conflict, instead of two live
registries silently overwriting each other's flushes. Two notes for
operators. The lock is `flock`-style: dependable on local disks, NOT
on network filesystems (NFS/EFS/FUSE mounts may grant it to both
sides) — shared-storage deployments must enforce single-attachment
themselves. And a rolling deploy that starts the new server while
the old still holds the volume will see the new one exit — correct
(it is what prevents a split-brain), but it means stop-then-start
(Kubernetes `strategy: Recreate` on a shared volume), not overlap.

### Deployment and availability

Taguru is deliberately a **single-node, single-writer** system: the
durability story (one fsync-then-apply owner per data directory) is
what makes "200 = durable" simple enough to trust, and the advisory
lock exists to enforce it, not to work around. Plan deployments on
that model rather than against it:

- **Deploys are stop-then-start** (`strategy: Recreate` on a shared
  volume). The downtime is the boot: cold registration is cheap
  however many contexts exist, and pinned contexts preload in
  parallel — the port opens when they finish. Size a `startupProbe`
  on `/live` accordingly; wire `livenessProbe: /live`,
  `readinessProbe: /health` (503 = write path degraded; routing off
  is the remedy, restarting is not).
- **Availability is restore time, not failover.** There is no
  replication or leader election; the recovery unit is the data
  directory. Two rungs: a filesystem snapshot restores the whole
  directory byte-exactly (fastest, same-version), and `taguru export`
  streams restore anywhere through `taguru import` /  `POST /import`
  (portable across versions and machines). Rehearse whichever rung is
  the plan — `taguru inspect` verifies either result offline.
- **Scale horizontally by partitioning, not clustering.** A context
  is the unit of routing (clients pick one by the directory), so
  independent taguru instances each owning a disjoint set of contexts
  scale reads and writes without any coordination protocol — point
  each client (or each MCP bridge) at the instance holding its
  contexts. What does NOT work is two instances sharing one data
  directory: on local disks the lock refuses it, and on NFS/EFS the
  lock may fail to — which is why shared network volumes need
  single-attachment enforced by the platform (e.g. EBS/PD
  ReadWriteOnce, never ReadWriteMany).
- **A rollback is a restore.** Image formats migrate forward on load
  and never write the old version back out, so rolling the BINARY
  back past a format bump needs the data directory rolled back with
  it (snapshot) or re-imported from an export stream. Check the
  release notes for format bumps before downgrading.

```sh
curl -X PUT localhost:8248/contexts/sake -H 'Content-Type: application/json' \
  -d '{"description":"青嶺酒造という架空の酒蔵の知識"}'
curl -X POST localhost:8248/contexts/sake/associations -H 'Content-Type: application/json' \
  -d '[{"subject":"青嶺酒造","label":"代表銘柄","object":"青嶺","weight":1.0,"source":"第1段落"}]'
curl -X POST localhost:8248/contexts/sake/activate -H 'Content-Type: application/json' \
  -d '{"origins":["青嶺酒造"]}'
```

For the endpoint list and the ingest/retrieval discipline, see
`GET /protocol`.

### Bulk loads offline (`taguru import`)

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

Validation is a separate pass: a malformed line refuses the whole run
(with its line number) before anything is written, and `--dry-run`
stops there on purpose. The data directory lock makes import and a
running server mutually exclusive — no torn state, just a refusal.
A file may carry one batch or a whole **stream** of them — every
`taguru_batch` header line starts the next batch — which is exactly
what `taguru export` writes, so a backup restores through the same
two entrances as any bulk load.

A **running** server takes the same contract at `POST /import` (one
request = one batch file or stream, same validation, same
replace-a-source semantics), so live systems bulk-load without a
downtime window:

```sh
curl -X POST localhost:8248/import -H 'Authorization: Bearer <key>' \
  --data-binary @docs-aomine.jsonl   # --data-binary: -d strips the newlines
```

Full contract for both entrances: [docs/import.html](docs/import.html).

### Documents in, batch files out (`taguru extract`)

Where do batch files come from? Any pipeline that speaks the format —
or the packaged producer: `taguru extract` reads `.md`/`.txt`
documents, has any OpenAI-compatible chat model decompose each into
associations under the /protocol discipline, and writes one batch
file per document (the document's path as the source id), ready for
either import entrance:

```sh
TAGURU_EXTRACT_URL=https://api.openai.com/v1/chat/completions \
TAGURU_EXTRACT_MODEL=gpt-4.1 TAGURU_EXTRACT_API_KEY=$KEY \
taguru extract --context sake --description "酒蔵の知識" --out batches/ docs/
taguru import batches/
```

The server-side rule is unchanged — it never holds model
credentials; `TAGURU_EXTRACT_*` lives in the offline producer's
environment only. A manifest in `--out` skips unchanged documents on
re-runs (`--force` overrides), model output is validated against the
batch contract before anything is written, and local or bridged
models (Ollama, LiteLLM in front of Bedrock) work the same way the
embedding side does. Full contract: [docs/extract.html](docs/extract.html).

## Using it from an LLM agent (MCP)

`taguru-mcp` is an MCP stdio bridge to a running HTTP server. Agents
(Claude Code / Claude Desktop, and so on) ingest and retrieve through
it — decomposing documents into facts and composing answers out of
results is the agent's job, and the discipline rides along
automatically as the tool definitions and the MCP instructions (the
content of `/protocol`).

```sh
cargo build --release   # or `cargo install taguru`, which puts taguru-mcp on PATH
claude mcp add taguru -e TAGURU_URL=http://127.0.0.1:8248 -- /path/to/target/release/taguru-mcp
```

With that in place, requests like "ingest the documents in this folder
into the sake context" or "tell me what you know about 青嶺酒造, with
sources" just work, as the loop directory pick → resolve →
describe/query/activate → passage lookup → cited answer. Chunking,
fact extraction, and the post-ingest coverage audit (audit_coverage)
are likewise driven by the agent through the tools.

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

Running the agent or the embeddings on Amazon Bedrock? See
[docs/bedrock.html](docs/bedrock.html): the Converse hosting pattern for
taguru-mcp, an InvokeModel embedding proxy, floor calibration, and the
access-gate diagnosis commands.

## Verification

```sh
cargo test                                    # library + registry + QA goldens
cargo test --test qa_recall -- --nocapture    # per-question recall table
cargo run --release --example benchmark       # per-operation latency (100k/1M associations)
taguru estimate --associations 1_000_000      # memory/disk sizing for a target corpus:
                                              # builds a context of that shape and
                                              # MEASURES it (latency: see benchmark)
```

Library-level usage examples live in [examples/](examples/) — one
directory per example, each with its own README saying what it
demonstrates and what to look for in the output. Client-side SDK
examples (Python/TypeScript with LangChain, one use case per
directory) live in [examples/langchain/](examples/langchain/).

`tests/qa_recall.rs` is the retrieval-quality regression floor: 11
questions against a fictional corpus (covering typo entry, full-width
entry, aliases, two-hop composition, negation, and corroboration) must
keep coming back fully answered by mechanically running the documented
retrieval loop.

`tests/code_recall.rs` is the same floor for source-code knowledge: 9
questions against a fictional Rust crate — camelCase and typo cues
landing on snake_case identifiers, qualified `Type::fn` and
path-fragment cues, case twins (`Frame` the struct vs `frame` the
accessor, both surfaced rather than merged), and natural-language
aliases bridging onto identifiers.

## License

MIT — see [LICENSE](LICENSE).
