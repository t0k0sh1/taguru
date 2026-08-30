# 0034. The overview pass asks per chunk, and a failed ask is recorded as no answer

- **Status**: Accepted
- **Date**: 2026-08-30
- **Issue**: #782
- **Related**: ADR 0033 (the chunk-context design this amends one
  paragraph of), ADR 0019 (the one escalated resend the pass gets),
  #179 (checkpoints — the resume this keeps stable)
- **Supersedes**: ADR 0033 §3.5's granularity — "asks the model, unit
  by unit in document order and within the same chunk cap … same
  checkpoint store (keyed by the unit text …) … an `overview` trace
  record per unit" and §4's "whose piece count is the unit count".
  Everything else in ADR 0033 §3.5 (what is asked, one pass before
  extraction, `stage: "overview"` attempts, its answer never a source
  of associations) stands. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The unit of one overview completion, what it is keyed by in the
checkpoint and the trace, and what the checkpoint holds for a chunk
whose ask the pass could not land. Out of scope: the block's content
and order (ADR 0033 §3.6, unchanged), the modes (§3.3, unchanged).

## 2. Context

ADR 0033 §3.5 was written with "structural unit" as the natural unit
of a synopsis — and it is, for the *answer*. As the unit of a *model
call* it does not survive the minutes corpus: a Diet committee
transcript detects one `◆` unit per utterance (90 units in a 65 KB
document, #780's `2024-05-28_衆議院_総務委員会`), which would be 90
overview calls for a document extraction reads in six chunks. The
call count must be bounded by the chunk count, as extraction's own is.

The second problem surfaced in review (#845): a chunk whose overview
ask fails (cut off even at the escalated resend, refused, empty, not
JSON) was skipped and not cached, so a resumed document re-asked it;
a different answer the second time changed the merged overview's
digest, and ADR 0033 §3.5's binding then discarded every checkpointed
extraction unit — a transient overview failure costing the whole
extraction over again.

## 3. Decision

**The overview pass asks once per chunk of the plan, in document
order, listing the structural units that open in that chunk by number
and heading; the answer is keyed by `chunk_sha256` in the checkpoint
and recorded as one `overview` trace record per chunk. A chunk whose
ask the pass cannot land is recorded in the checkpoint as an empty
answer — no synopsis, no cast — so a resumed document neither re-asks
it nor changes the overview its extraction units were bound to.**

1. **Per chunk, per-unit answers.** The synopsis is still per
   structural unit (ADR 0033 §3.1's definition holds): the ask lists
   the units opening in the chunk, and the answer carries one summary
   per listed unit plus the cast. A chunk with no unit opening in it
   is asked for cast only. The call count is the chunk count.
2. **Keyed by the chunk.** `chunk_sha256` is what extraction is keyed
   by too, so the checkpoint's `overview` map and its `units` map
   name the same things; the trace's `overview` record carries
   `chunk_index` and the per-unit summaries inside it.
3. **A failed ask is an empty answer, not an absence.** The
   difference matters only for resume: an absence would be re-asked,
   and a *different* answer on the retry re-binds every extraction
   unit (ADR 0033 §3.5's digest) — the cost the failure was meant to
   avoid. Recorded empty, the document's overview is fixed for the
   life of its checkpoint (cleared when the batch lands, or by
   `--force`), and the stderr line says so. Context is advisory; a
   missing synopsis for one chunk is the cheaper outcome.

## 4. Consequences

- ADR 0033's text stays as written; a reader of §3.5 who wants the
  unit of a call reads this ADR (§3.5 names it in *Related* only by
  this document's *Supersedes*, as the convention has it).
- `docs/extract.html`, the CHANGELOG, and the code already describe
  the per-chunk shape; the failure-caching rule is the one behavior
  change this ADR introduces, with a test: a chunk whose overview
  failed in run 1 is not re-asked in run 2, and the checkpointed
  units survive.
- Cost bound: the overview pass is at most one call per chunk plus
  one escalated resend per cut-off chunk — never more than twice the
  extraction's top-level chunk count.
