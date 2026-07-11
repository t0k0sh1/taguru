# accurate_retrieval — provenance, ranking, and the entry

Three additions on the road to "knowledge itself, accurately
searchable", demonstrated in the order they were built:

1. **Provenance** — `associate_from` tags every assertion with its
   source. Weight backed by two independent sources stays
   distinguishable from one emphatic assertion, and every attribution
   points back at the original text.
2. **Ranked retrieval** — `activate` spreads from the origins,
   attenuated per hop and split by each association's share of
   |weight|: near beats far, heavy beats light, exclusive paths beat
   hub paths.
3. **Entry resolution** — `resolve` maps free-form wording onto stored
   concept names lexically. It is the bottom tier of entry handling;
   the server layers normalization and (optionally) an embedding tier
   above it.

## Run

```sh
cargo run --example accurate_retrieval
```

## What the output shows

- **`recall("投票")`** — the twice-asserted fact reports its
  corroborated weight and `count 2` *separately*, with one attribution
  line per source. Two sources × 1.0 is no longer confusable with one
  source × 2.0.
- **`activate(["10大脅威選考会"])`** — the twice-asserted 協力する edge
  outranks the once-asserted ones; two-hop facts trail with decayed
  strength. Weights and distance both matter at read time.
- **`resolve("選考会")` → `activate`** — a query fragment lands on the
  stored name, which then anchors the ranked walk: resolve for the
  door, activate for the knowledge. This is the same
  resolve-then-retrieve loop `GET /protocol` teaches agents.
