# 0024. Every item the model wrote that the batch does not keep is recorded with its original text

- **Status**: Accepted
- **Date**: 2026-08-23
- **Issue**: #786
- **Related**: #784 must-have (1) (the requirement), ADR 0023 (the
  trace file and ids this record rides on), ADR 0013 (mechanical
  removal — whose accounting this completes), ADR 0022 / #758 / #759
  (the Stage 2 and cross-document prunes, now recorded item by item),
  ADR 0001 §8 (never-silent-drop)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What `taguru extract` records about an item the model's accepted
answer contained that the written batch does not: mechanical removals
(ADR 0013, Stage 1 and the Stage 2 prunes), `merge`'s drops, and
`merge`'s duplicate folds. Out of scope: items in an answer that was
itself rejected and re-asked (the corrective turn's subject, #790),
and the loss *rate* itself (#792 computes it from these records).

## 2. Context

Until now a loss was a count (`duplicates`/`dropped`/`removed` on the
report line and the `document` diagnostics record) plus, for
removals, a path-addressed message. The item itself — what the model
actually wrote — survived nowhere once the batch landed, and the text
it was about had to be found by hand. #784's first must-have: the
dropped sentence may be the one that mattered, so every loss must be
readable in the original, by id, after the run.

## 3. Decision

### 3.1 Removals are structured at the point of removal

`Removal { path, reason, item }` replaces the removal string: `item`
is the element as the model wrote it (the raw JSON array element for
a Stage 1 removal; the parsed alias for a Stage 2 prune), `path` its
address in the accepted answer, `reason` the finding. `Display` is
the former string byte for byte, so stderr, the report line, and the
sidecar's `removed_items` are unchanged. Every prune — unresolvable,
claimed (#758), uncorrected (ADR 0022) — records onto the output it
pruned from, so a removal is always attributable to one piece; the
document-level report is built from those per-output records after
every prune, chunk-prefixed as before. Stderr order changes from
"Stage 2 first, then per chunk" to "per chunk, in removal order".

### 3.2 Merge records what it counts

`merge` pushes a `Loss { kind, reason, rule, item, origin, kept_origin,
paragraph }` wherever it increments `dropped` or `duplicates`: the
item re-serialized from the parsed struct (absent fields `null`), the
rule in the report's own words, the output it came from, and for a
duplicate the output whose copy was kept. The counters are these
records' lengths by reason — they are not computed separately.

### 3.3 The trace's `loss` record

The per-document trace (ADR 0023 §3.3) gains `kind: "loss"`, after
the `item` records: removals piece by piece, then merge's losses.
Fields: `item` (`association` | `alias` | `question`), `reason`
(`removed` | `dropped` | `duplicate`), `rule`, `path` (removals only),
`raw` (the item as written), `piece_id`, `attempt`, `paragraph`, and
`text` — **the original text the item was about**: the cited paragraph
(the document's canonical paragraph, not the `[N] `-labeled rendering)
when the item cited a valid one, else the whole piece the model was
shown, as sent. A duplicate adds `kept_piece_id`. `text` is never
empty and never truncated: a loss is readable on its own line.

### 3.4 Checkpoints

A checkpoint unit's `removed` is now a list of `Removal`s. A unit
written before this ADR holds strings; they load as item-less
removals (`path`/`reason` split back out, `item: null`), so a resumed
document keeps its accounting and only its pre-ADR units lack `raw`.

### 3.5 The loss rate (for #792)

Per piece: `lost = count(loss records for the piece)`, `kept =
count(item records for the piece)`, `parsed = kept + lost`; loss rate
= `lost / parsed`, by `reason`, rolled up to document, context, group,
and run. No field is added for the denominator: the trace already
holds every kept and every lost item.

### 3.6 Lossy mode's parse-time drops

Under `--lossy`, an array element that is not an object never reaches
`merge`: lossy parsing returns `None` for it and discards the issue.
`evaluate_answer` records each such element (`unparsed`, a `Removal`
with the element verbatim and the path), the output and the
checkpoint carry it (defaulted on older files), and the trace writes
it as a `dropped` loss after the piece's removals. Strict mode never
fills it — the mechanical pass removes those elements with accounting.
Lossy's report line and stderr are unchanged: the record is trace-only.

## 4. Consequences

- Every loss can be read in the original by anyone holding the
  `--out` directory, with no model call and no re-parse of answers.
- A trace grows by one line per loss, each carrying up to one piece
  of text (24 KiB default) when the item cited nothing — bounded by
  the document's own size times its loss count. #788's size policy
  covers the per-piece text separately; this record keeps the
  paragraph where it can.
- Every item an accepted answer held that the batch does not is
  recorded — under `--lossy` too (§3.6).
