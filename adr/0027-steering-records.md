# 0027. What taguru put into the prompt is recorded as data

- **Status**: Accepted
- **Date**: 2026-08-24
- **Issue**: #789
- **Related**: #784 axes 2/3 ("did what we passed distort the
  answer"; "which parameter to adjust"), #759 (the amplification
  incident this makes auditable), ADR 0014 (candidates), ADR 0015
  (context vocabulary), ADR 0009 §11.1 (the schema block), ADR 0023
  (the trace), ADR 0025 (the attempts log — the same information as
  rendered prose, inside the system prompt), #782 (per-chunk context,
  which will extend this record kind)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

A structured record of the steering taguru itself adds to the prompt
— candidate names, the reuse-label list with its counts, `--vocabulary`
context names, the schema block's lists. Out of scope: the prose
rendering (the attempts log holds the system prompt verbatim, ADR
0025), and any new steering (#782 adds its own).

## 2. Context

#759's failure mode — a meaningless label enters the reuse list once
and every later chunk is nudged into it — was diagnosed by rereading
prompts. The attempts log now keeps those prompts, but answering
"what was on the list when chunk N ran, at what count" from prose
means parsing the prompt back. The lists are data taguru computed;
they should be recorded as data, exactly as prompted (ranking and
caps included), so label concentration at offer-time and at
output-time can be compared mechanically (#792).

## 3. Decision

One `kind: "steering"` record in the per-document trace, right after
the `document` record. Fields:

- `chunk_index`: `null` — document scope. Today every chunk of a
  document sees the same steering (ADR 0014 computes candidates from
  the whole document once; the vocabulary grows only between
  documents). When #782 adds per-chunk context, its records carry the
  chunk's index; a chunk's steering is the document-wide record plus
  its own. New steering kinds extend this record's fields on new
  records, never retroactively.
- `candidates`: ADR 0014's list as offered (capped at 100; empty when
  `--candidates` is off or the document yields none).
- `vocabulary`: `[{label, count}]` in prompt order — #759's ranking
  (count desc, label asc) and the 200 cap, computed by the **same
  function** that renders the prompt block (`ranked_vocabulary`,
  factored out of `system_prompt`), so record and prompt cannot
  drift.
- `context_names`: ADR 0015's capped list, as prompted.
- `schema`: `{types, constrained_relations}` — the schema block's two
  capped lists, by the same factoring (`schema_type_names`,
  `schema_constrained_relations`); `null` exactly when no schema
  block was prompted (`mode: off` included).

## 4. Consequences

- "What did the model see on the lists for this chunk" is one record,
  joinable by the trace's ids; #759-type amplification is traceable
  list-state by list-state across a run's documents.
- #792 can compute offer-time label concentration against final-output
  concentration with no prompt parsing.
- The record duplicates information the attempts log holds as prose.
  Deliberate: one is evidence (verbatim prompt), the other is data
  (what to aggregate); both are computed from the same inputs, and
  the list builders are shared code paths.
