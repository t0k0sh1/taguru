# document_ingestion — dry-run review, apply, replace, retract

The write path's governance, staged like a code review. `TaguruIngester`
decomposes a fictional tea shop's docs; the stages show where a human (or a
pipeline gate) fits between the LLM and the graph:

1. **dry_run** renders each document's exact NDJSON batch — header (with
   the context-creating block), the verbatim passage, doc2query questions,
   facts with paragraph locators, aliases. Nothing is written.
2. **apply** feeds that reviewed NDJSON byte-for-byte to the core SDK's
   `import_batches` — the dry-run artifact IS the import payload, so what
   was reviewed is what lands.
3. **read back** with the core SDK: `list_sources`, `describe`, `query`,
   and `list_labels` — the live label vocabulary that seeds the next
   ingest prompt (reuse over synonym-coining).
4. **re-ingest a revision** under the same source id: `POST /import` is
   retract-then-apply per source, so the 創業年 weight stays 1.0 instead
   of doubling, and only the revision's new fact (支店) is new. A repeated
   triple in the model output is folded and reported, not double-written.
5. **retract** a source outright: its facts' evidence and its passage go.

## Run

```sh
# Python                                                # TypeScript
cd examples/langchain                                   cd examples/langchain
.venv/bin/python document_ingestion/python/main.py      npm start --workspace=document_ingestion/typescript
```

(Setup for both is in [../README.md](../README.md). No `TAGURU_URL` → a
real server is spawned; no `OPENAI_API_KEY` → a deterministic fake model.)

## Why source ids matter

The source id is the idempotency unit: re-running an ingest replaces that
document's contribution. That is why `TaguruIngester` refuses documents
without a `source` in metadata instead of hashing content — a hash would
mint a new source on every edit, orphaning the old contribution forever.
