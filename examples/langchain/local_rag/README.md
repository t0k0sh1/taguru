# local_rag — a fixed, fully local RAG corpus built from PDFs

Two short fictional papers (`papers/tanaka2024.pdf`, `papers/sato2023.pdf`) go in
through a PDF loader, a numbered-heading splitter, and `TaguruIngester` — one
context per section, one group per paper, `TaguruIngester(context=...)` switched
on every section. A plain LCEL chain then answers across both papers' groups at
once:

```
{context: TaguruRetriever(groups=[paper/tanaka2024, paper/sato2023]) | format_docs,
 question: passthrough} | ChatPromptTemplate | chat model | string parser
```

## Run

```sh
# Python                                       # TypeScript
cd examples/langchain                          cd examples/langchain
.venv/bin/python local_rag/python/main.py      npm start --workspace=local_rag/typescript
```

(Setup for both is in [../README.md](../README.md).) With no `TAGURU_URL` a real
server is spawned; with no `OLLAMA_MODEL` every LLM role — extract and answer —
runs on a deterministic fake model, so the wiring is visible without any local
model installed. Neither version ever runs `ollama pull`: point `OLLAMA_MODEL`
at a model you've already pulled (`ollama list`) for the real thing, e.g.:

```sh
OLLAMA_MODEL=qwen2.5:7b-instruct .venv/bin/python local_rag/python/main.py
OLLAMA_MODEL=qwen2.5:7b-instruct npm start --workspace=local_rag/typescript
```

The TypeScript version additionally needs `pdf-parse` for PDF text extraction
(installed automatically with `npm install`; the Python side's equivalent is
`pypdf`) and, for the real-model path, `@langchain/ollama` (the Python side's
`langchain-ollama`).

The embedding responsibility (the server's semantic passage lane) is
independent of this script — it is server configuration
(`TAGURU_EMBED_URL`/`TAGURU_EMBED_MODEL`), not something a client sets. Retrieval
and citations here work the same with or without it, on BM25 and the graph
alone if not. The full walkthrough — server setup with Ollama's embeddings
endpoint, version pinning, and the modeling behind the context/group choice —
is <https://t0k0sh1.github.io/taguru/local-rag-walkthrough.html>.

## What to look for

- Ingest prints one line per **section** (not per paper) — `TaguruIngester`
  is re-created with a new `context=` each time, and the fact/alias counts
  come from the (fake, by default) extract model's decomposition of that
  section alone.
- The retrieval line and the answer line are printed as two **separate**
  phases: the retrieved documents (lane, citation label, locator) are shown
  in full before generation ever runs, so a wrong or thin answer is
  diagnosable — did retrieval bring back the right section, or did the
  answer model use it badly?
- `CITATION_LABELS` in `main.py` / `main.ts` is the entire mapping between a
  Taguru **source id** (`tanaka2024/3`, the retract-then-apply idempotency
  unit chosen at ingest) and a human-readable **citation label** (`Tanaka et
  al. 2024, §3`) — the API only ever deals in the former.
- The final `cite_passage` / `citePassage` call traces the answer's claim
  back to the exact paragraph the fake extraction says it came from — the
  same call works identically against a PDF ingested for real.
- Run it twice in a row: `ensure_group` / `ensureGroup` and
  `TaguruIngester`'s `create_context: true` both make the write path
  idempotent, so a second run neither fails nor double-counts facts.
