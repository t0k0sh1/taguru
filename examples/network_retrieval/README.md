# network_retrieval — the mental model in one sitting

The consumer of a `Context` is an LLM: it distills prose into weighted
associations on the way in, and rebuilds prose from one retrieval's
result on the way out. This example shows exactly the middle step —
the package a single `explore` call hands the LLM to write from — and
asks the two questions that matter before any LLM is involved:

1. Does one call gather enough *connected* material to rebuild the
   original passage, or does it stop at directly-adjacent facts?
2. What can never be handed over at any depth — where is the hard
   ceiling on reconstruction fidelity?

## Run

```sh
cargo run --example network_retrieval
```

## What the output shows

- **Depth 1 → 3**: the retrievable neighborhood around
  `10大脅威選考会` grows hop by hop, each line one association with
  its hop distance — the raw material for prose reconstruction.
- **Depth 100 changes nothing**: the fact `決定 -(手段)-> 投票` sits in
  a disconnected component, because `決定` the concept and `決定する`
  the label were never unified during extraction. No retrieval from
  this anchor can ever return it. The ceiling is visible mechanically —
  this is why ingestion makes implicit membership explicit, and why
  `unreachable_from` exists (see `paragraph_corpus`).
- **One spelling, two referents → two `Context`s**: Apple the fruit
  and Apple the company each live in their own 文脈, and the same
  anchor string retrieves only its own context's facts. Inside one
  `Context`, one spelling means one referent — by contract, not by
  disambiguation machinery.
