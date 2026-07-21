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

Not provided, deliberately: a VectorStore facade (Taguru's retrieval is
structural-first — `similarity_search` would misrepresent it), a Memory class
(deprecated upstream in favor of LangGraph state), and agent Tools (the MCP
bridge `taguru-mcp` already serves the identical tools; pair it with
`langchain-mcp-adapters`).

The behavioral contract is the server's protocol document (`GET /protocol`);
the ingestion prompt/validation mirror `taguru extract` (PROMPT_VERSION is
kept in sync with `src/extract.rs`).
