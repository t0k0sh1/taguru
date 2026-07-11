# Examples

Mostly library-level examples: each drives
`taguru::context::Context` directly — the embedded association graph
the server wraps — with no running server, no network, no
credentials; `http_benchmark` is the one exception, a load generator
that needs a server to aim at. Every example is one
`cargo run --example <name>` and prints what it wants you to notice.
For the server-and-agent side (HTTP API, MCP, `import` / `extract`),
see the docs under [docs/](../docs/) or `GET /protocol` on a running
server.

| Example | One run teaches |
|---|---|
| [network_retrieval](network_retrieval/) | The mental model: `explore` hop by hop, the hard ceiling on reconstruction, one spelling = one referent |
| [accurate_retrieval](accurate_retrieval/) | Provenance and corroboration, ranked retrieval, and the `resolve` → `activate` entry pattern |
| [paragraph_corpus](paragraph_corpus/) | A five-paragraph corpus ingested under the full discipline, then audited for coverage |
| [benchmark](benchmark/) | Per-operation latency at 100k and 1M associations (release mode only) |
| [http_benchmark](http_benchmark/) | The whole SERVER under concurrent load: throughput and p50/p95/p99 per phase (needs a running server) |

Suggested order: `network_retrieval` → `accurate_retrieval` →
`paragraph_corpus`. Reach for `benchmark` and `http_benchmark`
(together with `taguru estimate`) when sizing a deployment.

## SDK + LangChain examples

[langchain/](langchain/) holds the client-side counterpart: use-case
programs built on the [SDK packages](../sdk/) (`taguru`,
`langchain-taguru`), one directory per use case with a Python and a
TypeScript version side by side — RAG QA with citations, governed
document ingestion, and conversational long-term memory. Unlike the
Rust examples above, each drives the whole server over HTTP (a real
binary is spawned per run) and still runs offline, with deterministic
fake LLMs standing in until you export an API key.
