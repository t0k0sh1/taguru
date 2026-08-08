# 0014. Dictionary-free candidate names for the extraction prompt

- **Status**: Accepted
- **Date**: 2026-08-08
- **Issue**: #496 (S2)
- **Related**: ADR 0013 (S1), ADR 0012 §4 (twin detection), ADR 0009
  (schema block precedent), ADR 0001 §12.2 (default-off discipline)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru extract` offers a document's own names to the model as
preferred subject/object spellings (#496 S2's "前段の候補制約"), and which
segmenter produces them. Out of scope: label candidates (labels keep the
run-vocabulary block), extract-time embedding resolve (S3), coverage
verification (S4), and the SDK producers.

## 2. Context

The 2026-08-08 bench quantified spelling variance as extract's dominant
quality tax: `cargo-nextest`/`nextest` twins across sessions, particle-glued
compounds respelled per answer, and — on the fastest local model — label
and name collapse. Today every layer downstream pays for it: the
consolidation audit detects twins after import (ADR 0012 §4), the
occurrence check removes fabrications after answering (ADR 0013), and
merge() folds only exact duplicates. Nothing yet *prevents* variance at
answer time, which is where it is cheapest to prevent.

#496's own 検討事項 left the segmenter open: real morphological analysis
(lindera with an embedded dictionary; vibrato with a runtime dictionary
file) buys precise noun-phrase boundaries at the price of either a
substantially larger binary and audit surface (lindera) or a
dictionary-distribution burden on every operator (vibrato). This tree's
dependency discipline (fastembed feature-gated, tree-sitter accepted with
measured cost) demands that price be paid only against a measured gap.

## 3. Decision

**Candidate names are produced by a deterministic, dictionary-free
segmenter (`candidates.rs`, algorithm `seg1`), offered in a
non-restrictive system-prompt block, behind a default-off control.**

1. **Segmentation (`seg1`)**: a token is a maximal run of word
   characters — CJK ideographs, katakana, Latin alphanumerics and the
   identifier connectors `-._/+#`. Hiragana is a separator by design:
   in technical Japanese prose it is where particles and inflections
   live (`リリースの署名鍵` → `リリース`, `署名鍵`), and script-adjacency
   merges compounds a dictionary would also keep whole (`約40分`,
   `コネクションプール枯渇`). Tokens that are single characters, purely
   numeric, or over 64 bytes are dropped; the rest dedup exactly, keep
   first-appearance order, and cap at 100 per document.
2. **Prompt contract, non-restrictive AND anti-checklist by
   construction** (the 検討事項's third point): the block asks the model
   to *copy a listed spelling when its subject/object refers to that
   entity*, states that unlisted entities remain allowed, and — the
   measured clause — forbids adding associations or aliases "just to
   cover this list." The first wording (without that clause) made the
   fastest model treat the list as a checklist: 37-alias tables per
   answer, mostly self-referential, and one timeout per three runs.
   Constraining spelling must never become constraining (or inflating)
   what may be extracted — the same reasoning that keeps relation
   labels out of the wire schema's `enum` (ADR 0009). The block is
   document-scoped and rides the system prompt after the vocabulary
   and schema blocks; `--structured-output` keeps owning syntactic
   shape, candidates own naming — the role split the 検討事項 asked
   for, and a measured dependency, not just a division of labor: under
   the candidate block a mid-size model started emitting the offered
   spellings as UNQUOTED JSON values (`"label": デプロイ先`),
   deterministically, on the densest document — a failure class
   schema-constrained decoding removes structurally. docs/extract.html
   therefore recommends pairing `--candidates` with
   `--structured-output`.
3. **Default-off** (`--candidates` / `TAGURU_EXTRACT_CANDIDATES`,
   ADR 0001 §12.2's discipline): the default prompt stays byte-for-byte
   pre-S2. The control's value (`""` off / `"seg1"` on) is a manifest
   and checkpoint fingerprint field, so toggling it — or revising the
   algorithm under a new name — re-extracts like any other computation
   input, and `PROMPT_VERSION` stays untouched. The default flips only
   on benchmark evidence, in its own PR.
4. **A heavier segmenter buys its way in with measurement, not
   anticipation**: if the extraction benchmark shows `seg1` boundary
   errors costing recall or consistency, lindera/vibrato arrive as an
   optional feature (the fastembed precedent) under a new algorithm
   fingerprint — nothing in this decision precludes that, and nothing
   funds it before the evidence.

## 4. Known limits of `seg1`, accepted deliberately

- All-hiragana nouns are never offered (they are indistinguishable from
  function words without a dictionary). The model still extracts them;
  they simply get no spelling anchor.
- Multi-word Latin noun phrases arrive one word at a time
  (`connection`, `pool` — never `connection pool`).
- Pure numbers are dropped: a value is copied, not named; the anchor
  belongs to the unit-bearing compound (`約40分`) when the document
  writes one.
- Katakana/kanji adjacency can overmerge across a missing particle;
  no such case appeared in the bench corpus, and the block being
  non-restrictive bounds the damage to a wasted candidate slot.

## 5. Consequences

- `benchmark extract` forwards `--candidates` as a global task setting
  (the fairness invariant) and records it in `extraction_settings`;
  the same field addition closes a pre-existing resume gap for
  `--lossy`, which was forwarded but never recorded.
- Rust-only, like `--schema`: the SDK producers gain nothing until a
  follow-up mirrors the block, and the parity tests continue to assert
  only the shared surfaces.
- S3 (extract-time vocabulary resolve) layers on top: candidates
  prevent NEW variance inside one document; S3 will align a document's
  spellings with the TARGET CONTEXT's existing vocabulary — different
  reference set, same prevention posture, audit (ADR 0012) stays the
  detection net under both.
