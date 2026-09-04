# 0037. Failure output addressability: name the piece, point at the record, read it back

- **Status**: Accepted
- **Date**: 2026-09-04
- **Issue**: #850
- **Related**: ADR 0023 (trace ids: `piece_id`), ADR 0025 (the attempts
  log this reads), ADR 0029 (move records), ADR 0022 (the failure line's
  resume hint), ADR 0020 (split: the pieces a chunk becomes), #851
  (document / chunk / piece vocabulary), #863–#865 (the same gap in
  `import`, `anchoring`, `evaluate`)
- **Supersedes**: nothing. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What a person can read off `taguru extract`'s failure output, and how
the records behind it are opened without a script: the stderr line a
failed document prints, the pointer from that line to its records, and
`taguru inspect` as the reader of an attempts log. Out of scope, each
with its own issue: the same gap in `taguru import --url` (the server's
`issues[]` are dropped by the CLI), `taguru anchoring` (no unanchored
association is ever named), and `taguru evaluate` (failing cases are
counted, not named); reading the trace file (`<batch stem>.jsonl`) and
the diagnostics sidecar through `inspect`; and the vocabulary question
of what to call a document / chunk / piece (#851 — this ADR uses the
three words as the code does: a document is one input file, a chunk is
what `--chunk-bytes` cut it into, a piece is a chunk or what the ladder
split a chunk into).

## 2. Context

The #783 field run failed `hiargs.rs.txt` on one malformed answer. The
operator saw:

```
taguru: extract: …/hiargs.rs.txt: chunk 1/3: the model would not produce the JSON object:
  not a JSON object: invalid escape at line 1 column 1799
extract: 0 written, 0 unchanged, 1 failed of 1 document(s)
```

and could not get from there to the text that failed. Chunk 1 had been
split twice (ADR 0020): seven pieces were attempted, three of the four
leaves succeeded, one failed — and `chunk 1/3` names the unit *before*
the split, so it points at 23 KiB when the failure is 4 KiB of it
(paragraphs 6–11, as it turned out). Nothing on stderr said which
piece, which paragraphs, or that an attempts log (ADR 0025) beside the
batch held the piece text and the answer in full. The log did hold
everything: `piece_id`, `chunk_index`, the user turn with its `[N]`
labels, the answer, the parse error. Reading it meant parsing JSONL,
finding the last `user` turn of the right record, and unescaping
`content` — a script, and knowledge of the record layout.

The survey behind this ADR (issue #850's table) found the same shape
elsewhere: records that exist but are not pointed at, and outputs that
count failures without naming one. It also found the repository's own
best examples — `evaluate`'s `invalid locator — reproduce with: POST
…` and `evalset`'s `<file>: line N: case 'x': <field>: …` — which name
the unit, the position, and the next command. That is the bar.

## 3. Decision

**A failure names the smallest unit that failed and where it sits in
the document; the document's failure line points at the records it
left; and `taguru inspect` reads those records back as text, addressed
by piece and by paragraph.**

### 3.1 A piece failure names its piece

Every `Err` that `extract_piece` returns for the piece it was given —
a malformed answer that exhausted its attempts, a provider refusal, a
runaway (ADR 0035), the split floor (a piece that cannot split further
and still ends at the cap or times out) — is prefixed:

```
piece <12 hex of piece_id> (paragraphs <first>–<last>, <bytes> B): <message>
```

`paragraphs` is the piece's `[N]` label range (`trace::paragraph_range`,
the same reading the trace's `piece` record makes); a single-paragraph
piece says `paragraph N`; an unlabeled piece (never produced by
`extract`'s own rendering, but the function is total) says only the
bytes. A sub-piece's failure propagates through its parents unchanged
— the innermost piece is the one named, never re-prefixed — and the
chunk loop's `chunk K/N:` prefix stays in front, so the line reads
outer to inner: source, chunk, piece, paragraphs, what happened.

### 3.2 The document's failure line points at its records

The document-level line (`taguru: extract: <source>: <message>`) gains,
after ADR 0022's resume hint, a pointer to what the run left behind:

```
— records: <out>/.extract-trace/<batch stem>.attempts.jsonl
  (taguru inspect <that path> --piece <12 hex>)
```

with `--piece` filled from §3.1's piece when the failure named one,
and `, diagnostics: <path>` appended when `--diagnostics-out` is on.
The pointer is omitted when the attempts log is off
(`TAGURU_EXTRACT_TRACE_ATTEMPTS=off`) or could not be opened — the
line never names a file that does not exist.

### 3.3 `taguru inspect` reads an attempts log

`taguru inspect PATH` with `PATH` ending `.attempts.jsonl` renders the
log for a person:

- **Unfiltered**: a header (source, run id, resumed or not; the
  `settings` record's model, prompt version, structured output, output
  budget, chunk size, chunk context, rung when present), then one line
  per `attempt` record in issue order — `#seq`, `chunk K/N` (N read
  from the user turn's `part K of N` line; a Stage 2 correction shows
  `cross-chunk`), `piece <12 hex>`, `paragraphs a–b (bytes)`, the stage
  and attempt count, the state, the finish reason when not `stop`, the
  requested cap, rung, elapsed time, tokens, and the attempt it
  corrects or was replayed from — with the attempt's parse error, up
  to three validation issues, and its removal count beneath, and each
  `move` record (ADR 0029) as `↳ escalate / split / demote / runaway`
  after the attempt it followed. A footer tallies attempts by state
  and moves, counts unreadable lines (a killed run's torn tail — said,
  never a failure), and names the piece to open next: the last
  attempt of a piece that never reached `stop_valid`, else the last
  attempt.
- **`--piece ID`** (the id or any prefix) and **`--paragraph N`**
  (every piece whose label range covers `[N]`): the matching attempts'
  rows, each followed by the piece text exactly as sent (the user
  turn's document part — `user_message_document`, the same inverse
  the trace uses — printed once per piece and referenced by `#seq`
  afterwards), a retry's corrective ask, and the answer, unescaped
  between `--- … ---` fences so it can be read or copied whole. An
  empty match is reported as `0 attempt(s) matched …` with exit 0.
- **`--json`**: the same report as one document — `{target, kind:
  "attempts", document, runs, settings, filter, attempts[], moves[],
  unreadable_lines, matched}` — built once and rendered twice, as the
  rest of `inspect` does (#371). The texts (`piece_text`,
  `corrective_ask`, `answer`) appear only under a filter, so the
  unfiltered document stays the size of its summary.
- Exit codes: 0 for a readable log, 1 for an unreadable file or one
  holding no `document`/`attempt`/`move` record at all, 2 for the
  flags misapplied (`--piece`/`--paragraph` on anything but an
  attempts log, a non-hex id, a non-numeric paragraph). A reader that
  closes the pipe early (`| head`) is not an error.

### 3.4 A corrupt checkpoint is said

`DocumentCheckpoints::load` treats a missing file, an unreadable one,
and a fingerprint mismatch alike — "nothing cached" — and said nothing
for any of them, so a damaged checkpoint cost a document's every unit
again in silence. It now prints one line for a present-but-unreadable
file (`ignoring an unreadable checkpoint at <path> — every unit of this
document re-extracts`, the manifest's own wording) and one for a
fingerprint mismatch that discards units (`checkpoint at <path> was
written under different settings — <n> unit(s) re-extract`). A missing
file stays silent: that is the normal first run.

### 3.5 What stays as it is

The message texts behind the prefix are unchanged, as are the
attempts log, the trace, the sidecar, the manifest, and every exit
code of `extract`. Nothing here is a fingerprint input.

## 4. Consequences

- **Behavior change, named in the changelog**: stderr lines for a
  failed document grow a piece prefix and a records pointer; a script
  matching `chunk K/N: the model would not produce` still matches
  (the prefix is inserted after the chunk clause). `taguru inspect`
  accepts a fourth kind of path.
- **The issue's example** (`hiargs.rs.txt`, paragraphs 6–11) is now
  two commands: read the failure line's `--piece`, run it.
- **Tests**: unit tests pin the rendering (rows, moves, filters, the
  pointer heuristic, torn tails, cross-chunk rows) and the
  `part K of N` inverse; an end-to-end test drives a stub server to a
  malformed answer and reads the log back down to the piece text;
  CLI tests pin the flag validation. §3.1/3.2/3.4 land with the
  failure-line change and carry their own.
- **Follow-ups**, filed from the survey: `import --url` printing the
  server's `issues[]` and naming the earlier owner in duplicate
  refusals; `anchoring` naming unanchored associations; `evaluate`
  naming failing cases and the `evaluation.json` field to read.
