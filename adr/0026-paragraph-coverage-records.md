# 0026. The unreflected side of a document is readable in the original

- **Status**: Accepted
- **Date**: 2026-08-24
- **Issue**: #787
- **Related**: #784 axis 2 ("which part of the document is not
  reflected"), ADR 0016 (the `--coverage` gap check these records give
  text to), ADR 0023 (the trace file), ADR 0024 (the item side of the
  same question — losses; this is the document side), #792 (the
  coverage-rate aggregation these records feed)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What the trace records about the document's own side of coverage:
which canonical paragraphs no kept item cites, with their text, and
the ADR 0016 gap sentences in full. Out of scope: the rate arithmetic
(#792), any change to what `--coverage` flags or how stderr reports
it, and re-asking the model about a gap (ADR 0016 rules it report-only).

## 2. Context

ADR 0024 answers "what did the model write that the batch lost";
the inverse — "what did the document say that nothing extracted
touches" — existed only as stderr lines with byte-capped quotes
(`--coverage`, ADR 0016) and an `uncovered` count. A paragraph no item
cites was not visible at all: the coverage check is sentence-level and
candidate-gated, so a paragraph can be uncited without ever being
flagged. #784 wants both sides readable in the original, by id.

## 3. Decision

Two record kinds in the per-document trace (ADR 0023 §3.4's rule:
new kinds, never new fields on existing kinds), after the `loss`
records:

- `paragraph` — one per canonical paragraph, always: `paragraph`,
  `bytes`, `items` (how many kept associations and questions cite it),
  `covered` (`items > 0`), and `text` — present **exactly when
  `covered` is false**, so the unreflected paragraphs are listable in
  the original from the trace alone while the covered ones (already
  addressable through the batch passage) add no bulk. Coverage rate —
  count- or byte-weighted — is a fold over these records; #792 does
  the rolling up.
- `uncovered` — one per ADR 0016 gap, exactly when `--coverage` is on
  (matching stderr line for line): `paragraph`, `sentence` — the
  sentence **in full** (`CoverageGap` now keeps it beside the
  120-byte `quote`, which stderr keeps using unchanged), `text` (the
  paragraph), and `chunk_index`/`chunk_sha256` — the first chunk of
  the plan whose paragraph range holds it (an oversized paragraph
  straddling chunks repeats its number across them, ADR 0003 §7; the
  first is recorded).

Coverage gaps are computed before the trace is written — moved, not
duplicated: the same `coverage_gaps` value feeds the trace, stderr,
the report line, and the diagnostics `document` record, which all
stay byte-for-byte as they were.

## 4. Consequences

- "Which paragraphs does the batch not touch" is answerable from the
  trace alone, in the original text, for every written document — and
  `covered`/`items`/`bytes` make the coverage rate one pass over the
  file.
- A fully-covered document adds one small line per paragraph; an
  uncovered one adds its uncited paragraphs' text — bounded by the
  document's own size, the same bound ADR 0024 accepted.
- A trace consumer that filters on known `kind`s (ADR 0023 §3.4 told
  it to) is unaffected.
