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
context it came from. Deep dives (`activate`, `explore`) stay
per-context: search across, then pull the thread where it answered.

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
| `TAGURU_KEY_SCOPES` | — | Per-key grants as one JSON object: roles `read` ⊂ `write` ⊂ `admin`, optionally restricted to named contexts |
| `TAGURU_WAL` | on | fsync every acknowledged write before applying it — a crash loses nothing |
| `TAGURU_CACHE_BYTES` | 512 MiB | Resident budget for unpinned contexts (LRU eviction) |
| `TAGURU_EMBED_URL` / `_MODEL` / `_API_KEY` | — | Semantic entry tier (OpenAI-compatible `/embeddings`); unset keeps the entrance purely lexical |
| `TAGURU_EMBED_AUTO` | off | Re-embed changes with each flush — recommended whenever agents drive the ingest |
| `TAGURU_EMBED_PASSAGES` | off | Also embed paragraphs (the semantic lane of passage search); a corpus is much larger than its glosses, so the spend is opt-in |
| `TAGURU_SEMANTIC_FLOOR` | 0.35 | Floor for the semantic tier — a property of the embedding model; recalibrate when you switch |
| `TAGURU_PUBLIC_URL` | — | Public base URL; enables OAuth on `/mcp` for claude.ai custom connectors |
| `TAGURU_RATE_LIMIT_PER_MIN` | 0 (off) | Per-key request budget — turn on whenever the server leaves localhost |
| `TAGURU_REQUEST_TIMEOUT_SECS` | 30 | Per-request budget; raise it when an embedding provider is configured |

The full table — durability ceilings, observability (`RUST_LOG`,
`TAGURU_LOG_FORMAT=json`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`TAGURU_LOG_SEARCHES`), body/concurrency caps — is in
[Getting started](https://t0k0sh1.github.io/taguru/getting-started.html)
and [Internal architecture](https://t0k0sh1.github.io/taguru/architecture.html).

## Running in production

- **The image is the server alone on `scratch`** — a ~13 MB static
  binary, no shell, no libc, nothing to patch. It runs as uid 65532,
  works with `--read-only` (/data is the only write target), stops
  gracefully (flush + usage sweep), and health-checks itself
  (`taguru health` — there is no curl on scratch).
  `docker build -t taguru .` builds the same image locally.
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
  on local disks, *not* on NFS/EFS. Deploys are stop-then-start;
  availability is restore time, not failover; scale horizontally by
  giving independent instances disjoint sets of contexts. Note that
  groups and cross-context search reach only the contexts of their own
  instance — keep contexts that must be searched together on the same
  one when sharding.
- **Health and metrics.** `GET /health` is readiness (503 while the
  write path is degraded — route away, don't restart), `GET /live` is
  liveness, `GET /metrics` is Prometheus text. Every request lands in
  the access log with its key, context, and latency; destructive
  operations additionally leave one self-contained `taguru::audit`
  line. Distributed tracing is opt-in via
  `OTEL_EXPORTER_OTLP_ENDPOINT`.
- **Backups.** `POST /flush`, then snapshot the data directory (every
  writer is fsync+rename, so filesystem snapshots are safe at any
  instant; back up each context's file family as a set) — or take the
  portable JSONL stream with `taguru export` and restore anywhere
  through `taguru import` / `POST /import`. Verify either with
  `taguru inspect` — images, passage snapshots, and WAL records carry
  CRC-32C checksums, so "ok" means the bytes were proven intact, not
  just parseable. Reclaim revision-heavy contexts with
  `taguru compact`; size targets with `taguru estimate`.

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
