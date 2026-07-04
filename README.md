# Taguru

Long-term semantic memory for LLMs that recalls the way a thread is
pulled in hand over hand (手繰る, *taguru*). Knowledge accumulates as
(subject, relation label, object, signed weight, source) associations,
and retrieval is **structural** rather than embedding-similarity: the
cue is not what a question looks like but what it is *about* — the
糸口, the end of the thread — and the graph is walked from there to
draw the knowledge out.

The intended client is an LLM. Everything that needs language
understanding — decomposing documents into facts, choosing a context,
recomposing results into prose — is the client's job; this server only
stores and walks structure. The server distributes the complete
playbook for clients itself: `GET /protocol` (the content of
[docs/llm-protocol.md](docs/llm-protocol.md)).

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
  `{name}.sources.json` (original passages). Boot registers every
  context cold (pinned ones preload), the first access loads
  transparently, least-recently-used contexts are evicted past the
  cache budget, and writes mark a context dirty — persisted by the
  periodic flusher, on eviction, and on shutdown.

## Running

```sh
cargo run --release
# Environment:
#   TAGURU_ADDR         bind address (default 127.0.0.1:3000)
#   TAGURU_DATA_DIR     data directory (default ./data)
#   TAGURU_CACHE_BYTES  resident budget for unpinned contexts (default 512 MiB)
#   TAGURU_FLUSH_SECS   image flush interval (default 5). With the WAL on
#                     this is freshness cadence, not a loss window.
#   TAGURU_WAL          per-context write-ahead log: every acknowledged
#                     graph write is fsynced before it applies, so a
#                     crash loses nothing (default on; 0/false restores
#                     the flush-interval loss window)
#   TAGURU_EMBED_URL / TAGURU_EMBED_MODEL / TAGURU_EMBED_API_KEY
#                     semantic entry tier (OpenAI-compatible /embeddings).
#                     Unset keeps the entry purely lexical.
#                     e.g. URL=https://api.openai.com/v1/embeddings
#                          MODEL=text-embedding-3-large  (3-small separates
#                          short Japanese names too poorly; the default floor
#                          0.35 is calibrated for 3-large + glosses)
#                          API_KEY=$OPENAI_API_KEY
#   TAGURU_EMBED_AUTO=1 refresh embeddings incrementally with each flush
#                     (opt-in; unset means manual POST /embeddings/refresh only)
#   RUST_LOG            log filter (default info), EnvFilter syntax
#   TAGURU_LOG_FORMAT   json for one JSON object per log line (default: pretty).
#                     Logs go to stderr.
#   TAGURU_API_TOKEN    bearer token required on everything but /health and
#                     /metrics. Unset = UNAUTHENTICATED (localhost only).
#                     The MCP bridge reads the same variable.
#   TAGURU_MAX_BODY_BYTES      request body cap (default 8 MiB)
#   TAGURU_REQUEST_TIMEOUT_SECS  per-request budget (default 30; raise
#                     above 60 when TAGURU_EMBED_URL is configured — the
#                     provider round trip cannot be preempted mid-call)
```

Observability: every request lands in the access log, and
`GET /metrics` serves Prometheus text — per-route request counts and
latency histograms, cache/flush/embedding outcomes, and residency
gauges.

Backups: one context is the whole file family — `{stem}.ctx`,
`.meta.json`, `.sources.json`, `.vectors.bin`, `.wal.jsonl` — back
them up together, never partially. Every writer is fsync + rename, so
a filesystem-level point-in-time snapshot (ZFS/Btrfs/LVM) of the data
directory is safe at any moment; a file-by-file copy of a *running*
server (plain `rsync`/`cp`) is not guaranteed consistent across files
— stop the server or use a real snapshot.

```sh
curl -X PUT localhost:3000/contexts/sake -H 'Content-Type: application/json' \
  -d '{"description":"青嶺酒造という架空の酒蔵の知識"}'
curl -X POST localhost:3000/contexts/sake/associations -H 'Content-Type: application/json' \
  -d '[{"subject":"青嶺酒造","label":"代表銘柄","object":"青嶺","weight":1.0,"source":"第1段落"}]'
curl -X POST localhost:3000/contexts/sake/activate -H 'Content-Type: application/json' \
  -d '{"origins":["青嶺酒造"]}'
```

For the endpoint list and the ingest/retrieval discipline, see
`GET /protocol`.

## Using it from an LLM agent (MCP)

`taguru-mcp` is an MCP stdio bridge to a running HTTP server. Agents
(Claude Code / Claude Desktop, and so on) ingest and retrieve through
it — decomposing documents into facts and composing answers out of
results is the agent's job, and the discipline rides along
automatically as the tool definitions and the MCP instructions (the
content of `/protocol`).

```sh
cargo build --release                       # builds target/release/taguru-mcp
claude mcp add taguru -e TAGURU_URL=http://127.0.0.1:3000 -- /path/to/target/release/taguru-mcp
```

With that in place, requests like "ingest the documents in this folder
into the sake context" or "tell me what you know about 青嶺酒造, with
sources" just work, as the loop directory pick → resolve →
describe/query/activate → passage lookup → cited answer. Chunking,
fact extraction, and the post-ingest coverage audit (audit_coverage)
are likewise driven by the agent through the tools.

## Verification

```sh
cargo test                                    # library + registry + QA goldens
cargo test --test qa_recall -- --nocapture    # per-question recall table
cargo run --release --example benchmark       # per-operation latency (100k/1M associations)
```

`tests/qa_recall.rs` is the retrieval-quality regression floor: 11
questions against a fictional corpus (covering typo entry, full-width
entry, aliases, two-hop composition, negation, and corroboration) must
keep coming back fully answered by mechanically running the documented
retrieval loop.

## License

MIT — see [LICENSE](LICENSE).
