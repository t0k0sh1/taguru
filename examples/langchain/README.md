# LangChain examples (Python + TypeScript)

Use-case examples for the [SDK packages](../../sdk/): the `taguru` core
client plus `langchain-taguru`'s `TaguruRetriever` / `TaguruIngester`. One
directory per use case; each contains a Python and a TypeScript version of
the same program, mirrored line for line — pick your language and diff them
if you're porting.

| Use case | One run teaches |
|---|---|
| [rag_qa](rag_qa/) | The full RAG loop: LLM-decomposed ingest, then a retriever → prompt → LLM chain answering questions with verbatim per-paragraph citations |
| [document_ingestion](document_ingestion/) | The write path's governance: dry-run review of the exact NDJSON, apply the reviewed batch, re-ingest as per-source REPLACE (no double counting), retract |
| [conversational_memory](conversational_memory/) | Taguru as an assistant's long-term memory: one source per session, recall in later sessions, correction by negative weight |
| [local_rag](local_rag/) | A fixed, fully local RAG corpus from PDFs: `TaguruIngester(context=...)` switched per paper section, groups searched across papers, source id vs. citation label kept separate |
| [evidence_assembly](evidence_assembly/) | `Context.assemble_evidence()`: ranked, deduplicated, budgeted evidence for an external answer model — the same call `taguru evaluate --assembly` compares against fixed-limit retrieval at equal budget |

## Running them

Every example is self-contained and runs offline: with no `TAGURU_URL` it
spawns a real server binary (built with cargo on first run — the Rust
toolchain is the only requirement), and with no `OPENAI_API_KEY` it drives
the LLM steps with a canned deterministic fake model, so what the wiring
does is visible without any credentials. Point `TAGURU_URL` (and
`TAGURU_API_TOKEN`) at a running server and/or export `OPENAI_API_KEY`
(with `langchain-openai` / `@langchain/openai` installed) for the real
thing. [local_rag](local_rag/) is the exception: it drives its real-model
path with a local Ollama model instead (`OLLAMA_MODEL`, with
`langchain-ollama` / `@langchain/ollama` installed and the model already
pulled) — see its own README.

### Python

```sh
cd examples/langchain
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python rag_qa/python/main.py
```

### TypeScript

The examples consume the built SDKs, so build those once first:

```sh
cd sdk && npm install && npm run build --workspaces && cd ..

cd examples/langchain
npm install
npm start --workspace=rag_qa/typescript
```

The npm workspace here installs the SDKs as pack-copies of the `file:`
dependencies (see [.npmrc](.npmrc)). npm never refreshes those copies while
the SDK version stands still, so after rebuilding the SDKs re-pack them
with:

```sh
rm -rf node_modules/taguru node_modules/langchain-taguru && npm install
```

When the packages are on npm, a copied-out example only needs the `file:`
specifiers swapped for registry versions.
