# evidence_assembly — budgeted evidence assembly, ranked and handed to an answer model

The same two-document fictional-brewery corpus `rag_qa/` uses goes in
through `TaguruIngester`. The read side calls the core SDK's
`Context.assemble_evidence()`/`assembleEvidence()` directly —
`POST /contexts/{name}/evidence` — under two budgets for the same
query: generous (server defaults, nothing omitted) and tight
(`max_items: 1`, forcing a choice). This is the same call
[`taguru evaluate --assembly`](https://t0k0sh1.github.io/taguru/evidence.html#equal-budget)
drives for the equal-budget comparison against fixed-limit retrieval.

## Run

```sh
# Python                                             # TypeScript
cd examples/langchain                                cd examples/langchain
.venv/bin/python evidence_assembly/python/main.py    npm start --workspace=evidence_assembly/typescript
```

(Setup for both is in [../README.md](../README.md). No `TAGURU_URL` → a
real server is spawned; no `OPENAI_API_KEY` → deterministic fake models.)

## What to look for

- The generous-budget package admits everything — nine items, an empty
  `omitted`. The tight-budget one keeps only the single highest-fused-rank
  item and accounts for the rest under `omitted_total`/`omitted_by_reason`
  (`{"budget_exceeded": 8}`) — nothing simply vanishes.
- `items[]` mixes `kind: "association"` (graph facts, reached via the
  always-run `activate` lane) and `kind: "passage"` (BM25 hits) in one
  ranked, deduplicated list — reciprocal-rank fusion, never a raw
  BM25-vs-graph-weight comparison.
- The negatively-weighted 青嶺酒造–行う–大量生産 fact (weight `-1.0`) is
  admitted like any other — a denial is evidence, never silently dropped;
  `corroboration`/`contradicts` are how a real disagreement between two
  sources would surface here.
- `plan.reranker.configured: false` — no `TAGURU_RERANK_URL` is set, so
  selection is fully deterministic at no network or credential cost (ADR
  0006 §14 configuration 2).
- The closing line marks the boundary explicitly: everything above
  `answer:` is evidence Taguru assembled and budgeted; everything after it
  is the (fake) answer model's own prose. Taguru never generates it.
