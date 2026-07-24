# langchain-taguru (Python)

Official LangChain integration for the [Taguru](https://github.com/t0k0sh1/taguru)
long-term semantic memory server. The TypeScript twin (`langchain-taguru` on
npm) exposes the identical surface.

```sh
pip install langchain-taguru
```

```python
from langchain_openai import ChatOpenAI
from taguru_langchain import TaguruIngester, TaguruRetriever

# Write: an LLM decomposes documents into the association graph
# (the LangChain twin of `taguru extract`; per-source replace, idempotent).
ingester = TaguruIngester(
    context="sake",
    llm=ChatOpenAI(model="gpt-4.1", temperature=0),
    create_context=True,
    context_description="青嶺酒造という架空の酒蔵の知識",
)
ingester.ingest_documents(docs)          # docs[*].metadata["source"] required

# Read: graph lane (resolve → activate → citations) + text lane
# (search_passages), merged by Reciprocal Rank Fusion.
retriever = TaguruRetriever(context="sake", k=8)
documents = retriever.invoke("青嶺酒造")
```

Runnable use-case examples (RAG QA with citations, governed ingestion,
conversational long-term memory — each mirrored in TypeScript) live in
[examples/langchain](https://github.com/t0k0sh1/taguru/tree/main/examples/langchain);
they work offline, no API key needed.

`TaguruIngester` takes an optional `on_event` callback for live progress —
document/chunk/attempt/import/embedding-refresh events, including *why* a
corrective attempt fired. Useful with slow local models, where a single
`ingest_text()` call can otherwise look like one long silent block:

```python
ingester = TaguruIngester(..., on_event=lambda event: print(event.kind))
```

Three more constructor arguments bound how a chunk's structured-output
retry behaves, all optional and all unchanged by default: `fact_budget`
asks the model to keep a chunk's answer to at most N associations;
`max_attempts` (default 2, 1-10) raises or lowers the total attempts at
valid JSON per chunk before the document fails; and
`corrective_context_bytes` caps how much of a malformed answer gets
replayed back on the next attempt (`0` omits it behind a placeholder;
left unset, the default, replays it in full). Worth raising
`max_attempts` or setting `fact_budget`/`corrective_context_bytes` on slow
local models, where a large malformed answer near the output cap can
otherwise stall a chunk for minutes.

`TaguruIngester` also takes an optional `structured_output` flag (default
`False`) that asks the chat model for JSON-schema-constrained generation —
`llm.with_structured_output(MODEL_OUTPUT_JSON_SCHEMA, include_raw=True)` —
instead of parsing a free-text answer. Strictly opt-in and provider/model
dependent: a chat model that cannot bind tools raises out of the
constructor immediately, before any document is ingested, rather than
surfacing later as a per-attempt failure. Either way the answer still goes
through the same lenient validation walk and business-rule checks a
free-text answer gets — a schema only narrows what shape a well-behaved
provider can return.

By default, a business-rule-invalid item (a bad weight, a dangling alias,
an out-of-range question, ...) never gets silently dropped and reported as
a success: it earns one targeted, path-addressed corrective turn naming
exactly which fields are wrong, and the source fails outright (no
`/import` call) if it's still invalid afterward. Pass `lossy=True` to
restore the old drop-and-proceed behavior instead — the source still
imports, and `IngestOutcome.invalid_dropped` counts what got silently
discarded.

Not provided, deliberately: a VectorStore facade (Taguru's retrieval is
structural-first — `similarity_search` would misrepresent it), a Memory class
(deprecated upstream in favor of LangGraph state), and agent Tools (the MCP
bridge `taguru-mcp` already serves the identical tools; pair it with
`langchain-mcp-adapters`).

The behavioral contract is the server's protocol document (`GET /protocol`);
the ingestion prompt/validation mirror `taguru extract` (PROMPT_VERSION is
kept in sync with `src/extract.rs`).
