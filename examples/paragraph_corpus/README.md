# paragraph_corpus — a realistic corpus under the full discipline

Five paragraphs describing one coherent 文脈 (the fictional brewery
青嶺酒造 — ground truth fully known), hand-ingested the way an
extracting LLM is instructed to work. The source text sits in the
comment at the top of `main.rs`; the 34 assertions below it apply the
ingestion discipline that `GET /protocol` (and `taguru extract`'s
system prompt) demand:

- **One spelling per referent, reused exactly** across paragraphs
  (雲居山の伏流水, 六代目当主, 高瀬 …), so recurring mentions land on
  the same nodes instead of fragmenting.
- **Negation as negative weight** — 「大量生産を行わない」 becomes the
  affirmative label 行う with weight −1.0.
- **One paragraph = one source**, so a fact stated by two paragraphs
  (仕込み水) lands as the corroborated weight with two attributions —
  visibly different from one emphatic claim.
- **Implicit membership made explicit** — 杜氏の高瀬 appears inside the
  brewery's own paragraphs, but until 青嶺酒造-(杜氏)->高瀬 was
  asserted, the 高瀬・蔵人 cluster was an island: only 24 of 31
  associations reachable. This very example caught that omission; the
  two membership edges in 第2段落 repair it.
- **Accepted losses** stay honest: speaker attribution (「…と高瀬は
  語る」), and manner folded into labels. Detail like that belongs in
  the text lane, not in triples.

## Run

```sh
cargo run --example paragraph_corpus
```

## What the output shows

- **`resolve` / `resolve_label`** — entry candidates for concepts and
  for the label vocabulary (the check-before-mint step of ingestion).
- **`activate(["青嶺酒造"])`** — the corroborated fact ranks first;
  direct facts follow with the negated one keeping its sign; the
  strongest two-hop fact just makes the cut.
- **`query(Some("高瀬"), …)`** — a person profile assembled across
  paragraphs by pinning one role.
- **`explore(["金賞"], 2)`** — award context pulled from 第1・2・4・5
  段落, while 第3段落 (people, unrelated to the award) stays out.
- **`unreachable_from(["青嶺酒造"])` → 0件** — the coverage audit an
  ingesting pipeline should run after every document. Retrievability
  is the hard ceiling on everything downstream.

The fictional corpus is the same family `tests/qa_recall.rs` uses as
the retrieval-quality regression floor.
