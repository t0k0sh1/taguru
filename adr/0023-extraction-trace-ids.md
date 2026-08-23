# 0023. Every extracted item is traceable to the piece and completion that produced it

- **Status**: Accepted
- **Date**: 2026-08-23
- **Issue**: #785
- **Related**: #784 (the umbrella: records that make `extract`'s own
  problems findable — every child of it builds on the identifiers ruled
  here), ADR 0003 §7 (the paragraph-index provenance `chunk` records
  already carry), ADR 0008 (the manifest — what this trace is NOT),
  #179 (checkpoints — the content-hash key this reuses as the piece
  id), ADR 0001 §7/§10 (the ladder and the diagnostics sidecar), #262
  (the `chunk`/`document` diagnostics records)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The identifiers `taguru extract` assigns to the things it works on —
runs, documents, chunks, pieces, completions, and the items of the
batch it writes — where those identifiers are recorded, and how the
records join. Also the shared conventions every later record (#786–
#791) follows: record kinds, the file they go to, and the size policy.
Out of scope: what those later records *contain* (each child issue
adds its own record kind under these rules), persistence for *resume*
(#781 — the trace is an observation, never something a rerun reads),
and any change to the batch format itself (§3.1 explains why there is
none).

## 2. Context

The 0.9.3 field verification (#758–#764) was diagnosed from stderr
lines and counts. The diagnostics sidecar records every completion
(`attempt`), every chunk, and every written document, and the batch
records every item — but nothing connects an item to the completion
that produced it. `attempt` records carry `(source, chunk_index,
attempt)`, which is not unique: the ladder runs several rounds per
piece (base, escalated, after a demotion restart) each numbering its
attempts from 1, and a split sub-piece shares its parent's
`chunk_index`, distinguished only by `piece_bytes`. `Fact.chunk_index`,
the one provenance field `merge` keeps, is the position of the output
in `merge`'s input list, which stops being the chunk index the moment
one chunk splits into two outputs — a latent bug hidden by the field
being read only by tests.

#784's two must-haves — "see every item the model wrote that the batch
did not keep, in the original text" and "audit every corrective turn
after the fact" — both start from "which completion, on which text,
produced this". The identifiers have to exist before either record
can.

## 3. Decision

### 3.1 The batch does not change; items are identified by content

`taguru import` parses batch lines with `deny_unknown_fields`, and the
batch is the wire contract `sdk/spec` pins. No id field is added to any
batch line. An item's identity is the content key the batch already
makes unique: `merge` folds duplicate association triples
(`subject`, `label`, `object`), keys aliases by spelling within
`concept`/`label`, and keys questions by (`paragraph`, `question`).
Every trace record that names an item names it by that key, verbatim.

### 3.2 Identifiers

| Thing | Identifier | Stability |
|---|---|---|
| Run (one `taguru extract` invocation) | `run_id`: 16 hex characters from the OS random source | Unique per invocation, by design — it identifies an execution, not an input |
| Document | `source` (the path as the run saw it) + `document_sha256` (the manifest's `sha256`: the document text's hash) | Content-addressed |
| Chunk (top-level unit of the plan) | `chunk_index` (0-based, within the document) + `chunk_sha256` (the chunk text as sent) | Content-addressed; same document and `--chunk-bytes` → same chunks |
| Piece (a chunk, or a sub-piece the split rung made of one) | `piece_id`: sha256 of the piece text — **the checkpoint unit's key** (#179), unchanged | Content-addressed; an unsplit chunk's `piece_id` equals its `chunk_sha256` |
| Completion (one `/chat/completions` call from the run's point of view — the client's transport retries are inside it, ADR 0001 §10) | `attempt_seq`: 1-based counter over the extraction completions the run issues, in issue order | Run-local; meaningful only with the `run_id` |
| Item | its content key (§3.1) | As stable as the batch |

`attempt_seq` is issued by the client (`ChatClient::next_attempt_seq`)
at the three extraction call sites — the Stage 1 item loop, the
ladder's rounds, and the Stage 2 cross-chunk correction — so two
records with the same `(run_id, attempt_seq)` describe the same HTTP
conversation wherever they appear. The ADR 0021 auto probe is not an
extraction completion and takes no number. The existing per-round
`attempt`/`max_attempts` fields stay as they are — they describe the
corrective budget, not identity.

### 3.3 Where the records go: a per-document trace file, beside the batch

A new directory `--out/.extract-trace/` holds one JSONL file per
written document, named exactly like the document's batch file
(`flattened_hashed_name`, `.jsonl`). It is **on by default** and has
no flag: like the manifest, it is part of what "written" means.

- Lifecycle is the batch's: written atomically in the same step as the
  batch (before the manifest records the document), replaced when the
  document is re-extracted, left alone when the manifest skips the
  document. A trace therefore always describes the batch beside it —
  unlike the diagnostics sidecar, which is truncated per run and is
  empty for a skipped document.
- Not a manifest input (ADR 0008): a trace is an observation of a
  computation, never a reason to redo it. A batch written by a version
  without this ADR simply has no trace.
- Hidden (dot-prefixed) like `.extract-checkpoints/`, and a
  subdirectory rather than a sibling file, so `taguru import DIR` —
  which expands a directory to its `*.jsonl` files — never reads one as
  a batch.

The diagnostics sidecar (`--diagnostics-out`, opt-in, run-scoped)
keeps its role: per-completion telemetry for *this* run. It gains what
it needs to join the trace — one `kind: "run"` record first, carrying
`run_id`, and `run_id`/`attempt_seq`/`piece_id` on every `attempt`
record (always present, additive; a consumer filtering on `kind ==
"attempt"` is unaffected).

### 3.4 Record kinds in the trace file

Every line is one record; `kind` discriminates, and a reader skips
kinds it does not know (later children add kinds, never fields to an
existing kind's key set). The kinds this ADR defines, in file order:

- `document` — once, first: `run_id`, `source`, `document_sha256`,
  `batch_path`, `chunk_total`.
- `chunk` — once per chunk of the plan: `chunk_index`, `chunk_total`,
  `chunk_sha256`, `chunk_bytes`, `paragraph_first`, `paragraph_last`
  (the same fields the diagnostics `chunk` record carries, ADR 0003
  §7).
- `piece` — once per output the document's chunk loop produced, in
  output order: `piece_id`, `chunk_index`, `chunk_sha256`,
  `piece_bytes`, `paragraph_first`, `paragraph_last`, `reused` (the
  output came from a checkpoint, not a completion of this run), and
  `attempt` — `{run_id, attempt_seq}` of the completion whose answer
  this output *is*: the accepted Stage 1 answer, or the Stage 2
  corrective answer when one replaced it. For a reused unit, the
  checkpoint carries the attempt of the run that produced it (§3.5);
  `null` only for a checkpoint written before this ADR.
- `item` — once per batch item, in batch order: `item`
  (`association` | `concept` | `label` | `question`), the content key
  (`subject`/`label`/`object`; `alias`/`canonical`;
  `paragraph`/`question`), and `piece_id`.

Join rules: item → piece by `piece_id`; piece → chunk by `chunk_index`
(or `chunk_sha256`); piece → diagnostics `attempt` by
`(attempt.run_id, attempt.attempt_seq)` when a sidecar from that run
exists; anything → document by the file, whose `document` record names
the batch. A duplicate folded across chunks is attributed to the
output `merge` kept (the first), matching what the batch contains —
the folded copies are #786's records, not this one's.

### 3.5 Checkpoints carry the attempt

A checkpoint unit (#179) gains `attempt: {run_id, attempt_seq}`
(serde-defaulted, so pre-ADR checkpoints still load) so a unit reused
across an interrupted-then-resumed document still names the completion
that made it. The checkpoint fingerprint is unchanged — the attempt is
data about the unit, never part of what decides whether the unit is
reusable.

### 3.6 Size policy for everything under this directory

The trace this ADR writes is one short line per item and per piece —
kilobytes. Children that add *text* (prompts and answers, #788; lost
items with their paragraphs, #786) add it to this same file, and must
state their own cap and opt-out in their section of this ADR's
successor or in their own ADR: the rule here is only that the default
stays on, that a record is never truncated silently (a cap is marked
in the record, as `TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES` marks one),
and that a trace write failure is reported and never fails the
document (the batch and the manifest are the truth; the trace is
advisory, the diagnostics sidecar's own ruling).

## 4. Consequences

- Any batch item can be followed to the text the model was shown and
  to the completion that produced it, by ids alone, after the run —
  and across a resume, through the checkpoint's `attempt`.
- `Fact.chunk_index` is replaced by `Fact.origin`, the index into the
  chunk loop's output list, which is what it always was; the trace
  maps it back to the real `chunk_index` and `piece_id`.
- The diagnostics sidecar's first line is now a `run` record. A
  consumer that assumed the first line is a `chunk` or `attempt` must
  filter on `kind`, which docs/extract.html has always said to do.
- `--out` grows a second hidden directory. `--dry-run` writes none of
  it; a failed document writes none of it (its checkpoint keeps the
  attempt ids for the resume).
