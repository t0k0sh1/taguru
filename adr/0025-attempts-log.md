# 0025. Every completion's full prompt and full answer are kept beside the batch

- **Status**: Accepted
- **Date**: 2026-08-24
- **Issue**: #788
- **Related**: #784 (the umbrella; axes 1 and 2 read this record),
  ADR 0023 (the trace directory and the ids this log joins on), ADR
  0001 §10 (the diagnostics sidecar's metadata-only ruling, which this
  does not change), #179 (checkpoints — the resume this log follows),
  #790 (corrective tuples, built on these records)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

Where `taguru extract` keeps the text of every conversation it has
with the model — the prompt as sent and the answer as received, for
every completion, succeeded or failed — by default. Out of scope: the
sidecar's opt-in `response_text` (unchanged), resume from these
records (#781), and what a reader does with them (#790, #792).

## 2. Context

Judging an association after the fact — "is this right; if not, did
the model misread or did taguru mishandle?" — needs what the model was
shown and what it said. Until now that text lived in two places,
neither of which survives a successful run: the checkpoint (cleared
when the batch lands) and, only under
`TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES`, a byte-capped answer on the
sidecar's `attempt` record, with the prompt never recorded at all. The
sidecar is metadata by design (ADR 0001 §10: it is a troubleshooting
file that should not leak document text by default) — that ruling is
right for a file meant to be shared, and wrong as the only home for
the full text, which belongs with the batch it produced.

## 3. Decision

### 3.1 A per-document attempts log, on by default

`--out/.extract-trace/<batch stem>.attempts.jsonl` — the batch file's
name with `.attempts.jsonl` for `.jsonl`, in the trace directory (ADR
0023 §3.3: hidden, never read by `import`), written incrementally:
one line per record, flushed, so a killed run keeps every completion
already made. `TAGURU_EXTRACT_TRACE_ATTEMPTS=off` (also `0`, `false`,
`no`) is the opt-out; a log that cannot be opened or written is one
stderr line, never a failed document.

Document text in this file is no more exposed than in the batch beside
it (which carries the whole passage) or the checkpoint: `--out` is
already where the document's own words live, so the log needs no
separate permission story. The sidecar keeps its metadata-only
default and its capped opt-in, unchanged — it is the file to hand to
someone else; this one stays with the data.

### 3.2 Lifecycle: the runs that built the batch

The log opens when a document's extraction starts (after the manifest
skip and the dry-run exit, so neither touches it): **truncated** when
the document starts fresh, **appended** when the checkpoint holds
units to resume from — the same span the checkpoint itself covers, so
the completions behind a reused unit are still in the file the trace
points at. A failed document keeps its log (that is what it is for);
a skipped document's log is left alone; `--force` starts fresh.

### 3.3 Records

- `document` — once per run over this document: `run_id`, `source`,
  `document_sha256`, `resumed`.
- `system` — the system prompt in full (`sha256`, `bytes`,
  `content`), written the first time an attempt of this document
  sends it. The prompt is fixed per document (the vocabulary grows only
  between documents), so it is written once, not per attempt; the
  Stage 2 prompt is the same text and hashes the same.
- `attempt` — one per completion, in issue order: `run_id`,
  `attempt_seq`, `piece_id`, `source`, `chunk_index`, `stage`,
  `attempt`, `max_attempts`, `state`, `length_limited`,
  `elapsed_seconds`, `requested_max_tokens`, `finish_reason`,
  `input_tokens`, `output_tokens` (the sidecar's fields, same names);
  `messages` — every turn as sent, the system turn as
  `{role, system_sha256}` and every other turn as `{role, content}` in
  full (the replayed prior answer and the corrective ask included, so
  a corrective round is readable as the model saw it); `answer` — the
  assistant's final text in full, `null` for `timeout`/`transport`;
  `parse_error`, `validation_issues`, `removed_items`.

Nothing is capped and nothing is truncated. Joins: to the sidecar by
`(run_id, attempt_seq)`; to the trace's `piece` by `piece_id` and
`attempt`; to the trace's `loss` by `piece_id` and `path` into
`answer`.

### 3.4 Size

Measured from the 0.9.3 field run's token counts (the upper bound,
before the system prompt is deduplicated): 5–70 KiB per document
across eight document kinds, 2–3 completions per document, ~2.5
bytes per input token and ~3 per output token for Japanese text. The
largest (minutes, 70 KiB; techdoc, 52 KiB) are dominated by the
prompt, most of which is the system prompt written once here. A
thousand-document corpus is tens of MB — on the order of the batches
themselves. No tiering (full text for failures only, say) is worth
its complexity at that size; the opt-out covers the rest.

## 4. Consequences

- Every completion of every document is readable after the run, by
  id, with no environment variable set — #784's axis 1 ("is the model
  adequate") and the "model misread or taguru mishandled" question
  have their raw material.
- The sidecar is unchanged; `TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES`
  stays as it was, for readers who only have the sidecar.
- Every call site that classifies an attempt reports it through one
  `Observers` value (sidecar + log) instead of an optional sink — the
  attempt is recorded exactly once wherever it is classified.
- `--out/.extract-trace/` now exists for a document that fails (its
  attempts log), where before it appeared only with a written batch.
