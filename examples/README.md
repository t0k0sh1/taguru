# Examples

Library-level examples: each one drives `taguru::context::Context`
directly — the embedded association graph the server wraps. No running
server, no network, no credentials; every example is one
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

Suggested order: `network_retrieval` → `accurate_retrieval` →
`paragraph_corpus`. Reach for `benchmark` (together with
`taguru estimate`) when sizing a deployment.
