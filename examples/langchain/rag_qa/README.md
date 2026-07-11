# rag_qa — RAG question answering with citations

Three short documents about a fictional sake brewery go in through
`TaguruIngester` (an LLM decomposes each into graph facts, aliases, and
doc2query questions; one idempotent `POST /import` batch per source). A
plain LCEL chain answers questions:

```
{context: TaguruRetriever | format_docs, question: passthrough}
  | ChatPromptTemplate | chat model | string parser
```

## Run

```sh
# Python                                  # TypeScript
cd examples/langchain                     cd examples/langchain
.venv/bin/python rag_qa/python/main.py    npm start --workspace=rag_qa/typescript
```

(Setup for both is in [../README.md](../README.md). No `TAGURU_URL` → a
real server is spawned; no `OPENAI_API_KEY` → deterministic fake models.)

## What to look for

- Each question first prints what the retriever returned: the lane it came
  through (`graph` = resolve → activate → citations, `text` = BM25 passage
  search, `graph+text` = corroborated by both) and its locator
  (`source ¶paragraph`). The answer's bracketed citations point back at
  exactly those locators.
- 「青嶺酒造はお酒を大量生産していますか?」 lands on a NEGATIVELY
  weighted fact: the extraction stored 青嶺酒造–行う–大量生産 with weight
  -1.0 — the graph keeps what a document denies as first-class evidence
  (look for `weight: -1` in the Document metadata), instead of hoping
  prose retrieval implies the refusal.
- Retriever knobs to try: `k`, `include_graph` / `include_text` (run one
  lane alone), `text_limit`, `resolve_limit`.
