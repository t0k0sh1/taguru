# langchain-taguru (TypeScript/JavaScript)

Official LangChain.js integration for the [Taguru](https://github.com/t0k0sh1/taguru)
long-term semantic memory server. The Python twin (`langchain-taguru` on PyPI)
exposes the identical surface — method names differ only by casing convention;
configuration fields are snake_case in both.

```sh
npm install langchain-taguru @langchain/core
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { TaguruIngester, TaguruRetriever } from "langchain-taguru";

// Write: an LLM decomposes documents into the association graph
// (the LangChain twin of `taguru extract`; per-source replace, idempotent).
const ingester = new TaguruIngester({
  context: "sake",
  llm: new ChatOpenAI({ model: "gpt-4.1", temperature: 0 }),
  create_context: true,
  context_description: "青嶺酒造という架空の酒蔵の知識",
});
await ingester.ingestDocuments(docs); // docs[*].metadata.source required

// Read: graph lane (resolve → activate → citations) + text lane
// (searchPassages), merged by Reciprocal Rank Fusion.
const retriever = new TaguruRetriever({ context: "sake", k: 8 });
const documents = await retriever.invoke("青嶺酒造");
```

Runnable use-case examples (RAG QA with citations, governed ingestion,
conversational long-term memory — each mirrored in Python) live in
[examples/langchain](https://github.com/t0k0sh1/taguru/tree/main/examples/langchain);
they work offline, no API key needed.

Not provided, deliberately: a VectorStore facade (Taguru's retrieval is
structural-first — `similaritySearch` would misrepresent it), a Memory class
(deprecated upstream in favor of LangGraph state), and agent Tools (the MCP
bridge `taguru-mcp` already serves the identical tools).

The behavioral contract is the server's protocol document (`GET /protocol`);
the ingestion prompt/validation mirror `taguru extract` (PROMPT_VERSION kept
in sync with `src/extract.rs`).
