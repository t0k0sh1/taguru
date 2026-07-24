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

`TaguruIngester` also takes an optional `structured_output` flag (default
`false`) that asks the chat model for JSON-schema-constrained generation —
`llm.withStructuredOutput(MODEL_OUTPUT_JSON_SCHEMA, { includeRaw: true })`
— instead of parsing a free-text answer. Strictly opt-in and provider/model
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
`/import` call) if it's still invalid afterward. Pass `lossy: true` to
restore the old drop-and-proceed behavior instead — the source still
imports, and `IngestOutcome.invalid_dropped` counts what got silently
discarded.

`TaguruIngester` also takes the same bounded structured-output controls as
`taguru extract` and the Python SDK, all defaulting to today's unbounded,
2-attempt, full-replay behavior:

- `fact_budget` — ask the model to keep each chunk's answer to at most N
  associations, folded into the system prompt (default: unbounded).
- `max_attempts` — total attempts (1 initial + corrections) at getting the
  model to answer with the JSON object asked for, `1..=10` (default `2`;
  `1` skips the corrective turn entirely).
- `corrective_context_bytes` — how much of the model's own prior bad
  answer is replayed back to it in the next attempt's corrective turn:
  unset replays it in full (default), `0` omits it behind a placeholder,
  any other value truncates it to that many bytes.

When the provider's `AIMessage.response_metadata` says a malformed answer
was cut off at its output-length cap (`finish_reason`/`done_reason`
`"length"`, or Anthropic's `stop_reason: "max_tokens"`), the corrective
ask switches from "try again" to "try again shorter," naming
`fact_budget` when one is set.

Not provided, deliberately: a VectorStore facade (Taguru's retrieval is
structural-first — `similaritySearch` would misrepresent it), a Memory class
(deprecated upstream in favor of LangGraph state), and agent Tools (the MCP
bridge `taguru-mcp` already serves the identical tools).

The behavioral contract is the server's protocol document (`GET /protocol`);
the ingestion prompt/validation mirror `taguru extract` (PROMPT_VERSION kept
in sync with `src/extract.rs`).
