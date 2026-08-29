# Changelog

Notable changes to taguru. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [SemVer](https://semver.org/) (pre-1.0: minor bumps may break).
Entries that change an on-disk format or a response shape say so.

## [Unreleased]

### Fixed

- `scripts/extract_metrics.py` now counts a document that failed — an
  attempts log with no trace beside it, the log ADR 0025 keeps for
  exactly this — as `failed`, apart from `documents`, with its attempts,
  ladder moves, and time/token/money cost rolled into every scope's sums
  and its quality metrics left empty; the document table marks the row
  `(failed)`. Before, such a document was silently absent from every
  table (an out-dir holding only failures answered `no traced documents
  found`), so lost cost and move counts were understated. Additive
  `failed` keys in the JSON report; the markdown tables gain a `failed`
  column. #807.

- `taguru extract --vocabulary DIR` and `taguru anchoring --vocabulary DIR`
  read the directory's `*.jsonl` files only — sorted by name,
  non-recursive, the rule `taguru import DIR` and `taguru anchoring DIR`
  already read by — instead of every regular file. A previous extract's
  `--out` directory can now be handed back as vocabulary: its batches are
  read, and its `.extract-manifest.json`, `.extract-checkpoints/`,
  `.extract-trace/`, and any other non-`.jsonl` file are ignored; before,
  the manifest sidecar was parsed as a batch stream and failed the run at
  startup. A directory with no `.jsonl` fails as `no .jsonl files under
  DIR` (was `the directory holds no files`). #805.

### Changed

- `taguru extract`: the Stage 2 cross-chunk corrective turn now climbs
  the output-cap ladder — a corrective answer that ends at
  `--max-output-tokens` is resent once at the escalated budget
  (`TAGURU_EXTRACT_ESCALATION_FACTOR` ×, as ADR 0019 already does for
  Stage 1), and a correction cut off even then (or with no budget
  configured) is treated as leaving its issues standing: an alias issue
  is removed with ADR 0022's accounting and the document lands, any
  other standing issue still fails the source. Used to fail the source
  on the first cut-off — the #780 ripgrep README reproducer. The resend
  writes an `escalate` move record (ADR 0029) naming the cross-chunk
  correction; **no on-disk format change** (additive use of existing
  record kinds). ADR 0032 (#811).

### Added

- `docs/extract.html` gains "The pipeline" section naming `taguru extract`'s
  steps (`read`/`plan`/`steer`/`prompt`/`call`/`parse`/`validate`/
  `reconcile`/`merge`/`render`/`verify`) as a public, extensible contract —
  ADR 0030 (#814) — plus a "by step" index into the existing trace/attempts
  records. No behavior change; no on-disk format or response shape change.
- `docs/extract.html` gains "Replaying a recorded run", the contract for
  satisfying a completion from the attempts log instead of a live model
  call — matching key, `--replay {auto|strict|off}`/`--replay-from`, file
  safety, manifest/checkpoint bypass, system-prompt pinning, and running
  the model call on one machine and replay on another — ADR 0031 (#815).
  Documents a contract only; the flags land in #781's later children. No
  behavior change; no on-disk format or response shape change.
- The attempts log's `attempt` record gains `rung` (the structured-output
  rung a completion was asked under; absent off the ladder), and a new
  `settings` record lands once per document right after `document` — a
  diagnostic snapshot of the run's compute inputs, mirroring what the
  manifest/checkpoint fingerprint checks, never a computation input itself
  or a replay matching key; `settings.rung` is also absent off the ladder
  — ADR 0031 §3.2/§3.9 (#817). **On-disk format change**: additive fields
  and a new record kind in `--out/.extract-trace/*.attempts.jsonl`; a
  reader filtering on known `kind` values is unaffected.
- `taguru extract --replay {auto|strict|off}` (default `off`) /
  `--replay-from DIR` (default `--out/.extract-trace`) satisfy completions
  from a prior run's attempts log instead of a live call — `auto` falls
  through to a live call when nothing matches; `strict` fails the document
  instead, running with no `TAGURU_EXTRACT_URL` at all (`TAGURU_EXTRACT_MODEL`
  stays required). Bypasses the manifest skip and the checkpoint store while
  replaying; matching is by exact conversation content, never by piece or
  attempt number, so a genuinely changed request always falls through to a
  live call on its own. A `settings` record recorded under different
  settings than the current run is named field by field on stderr — a
  hint, never a gate — ADR 0031 (#818, #819). **On-disk format change**:
  two new record kinds (`replay`, `replay_summary`) in
  `--out/.extract-trace/*.attempts.jsonl`, written only under `--replay`.
- `docs/extract.html` gains "Recording here, replaying there": the two-machine
  procedure `--replay strict` with no `TAGURU_EXTRACT_URL` is for (extract
  where the model is, carry the documents/`.extract-trace`/`--schema`,
  replay where it isn't) — ADR 0031 §3.8 (#820).
- `--replay` pins a document's system prompt verbatim from its attempts log
  when the log names exactly one distinct `system` record, instead of
  recomputing it — so replaying document two of a multi-document run no
  longer depends on this run's own vocabulary accumulation matching the
  original's (ADR 0031 §3.6, #821). A pinned prompt whose hash differs from
  what this run would have recomputed is reported once on stderr; a log
  naming more than one distinct `system` record is also reported and falls
  back to recomputing, never guessing which one applies. The trace's
  `steering` record gains `system_sha256` (the system prompt actually sent,
  by hash) and `pinned_from` (the run_id it was pinned from; absent when
  this run computed its own).
  **On-disk format change**: two additive fields on the `steering` record in
  `--out/.extract-trace/*.jsonl` (the trace).
- `taguru extract --resume-from STEP` selects a `--replay` mode by naming
  one of ADR 0030's pipeline step names instead — "steps before STEP are
  satisfied from the attempts log where recorded" (#822). This version's
  log only ever records `prompt`/`call` in full, so it folds onto one of
  three behaviors: `call` through `verify` fold onto `--replay auto`;
  `read`/`plan`/`steer` fold onto a plain, unreplayed run (nothing before
  them is recorded at all); `prompt` is `--replay auto` with the #821
  system-prompt pin turned off, so a settings change the pin would
  otherwise absorb falls through to a live call instead. Whatever step it
  names, `--resume-from` also bypasses the manifest skip and the
  checkpoint store — naming a resume point is a deliberate ask to redo
  the document, and an unchanged-document skip would silently answer
  that ask with nothing at all. Mutually exclusive with `--replay` (a
  usage error, not a silent override); an unknown step name is also a
  usage error naming the closed vocabulary. `docs/extract.html`'s "The
  pipeline" section gains the step → behavior table.
- The `attempt` record gains `replayed_from` — the original `{run_id,
  attempt_seq}` a completion satisfied from `--replay` reused, absent on
  every live completion (ADR 0031 §3.2, #823). `scripts/extract_metrics.py`
  now skips any `attempt` record carrying it entirely (state, transport
  retries, elapsed seconds, tokens): the replay's own numbers describe
  the replay itself (near-zero time, the original's tokens restated),
  so counting it too would double the cost while zeroing out the time.
  `docs/extract.html` documents the field, three known limits of
  content-addressed matching (corrective-turn strict failures,
  ambiguous-pin fallback, and `rung` never gating a match), and
  `docs/long-running.html` cross-links `--resume-from` as the cheaper
  alternative to the manifest skip's all-or-nothing redo.
  **On-disk format change**: one additive field on the `attempt` record
  in `--out/.extract-trace/*.attempts.jsonl`, written only under
  `--replay`.

### Changed

- The `--structured-output auto` + no-endpoint usage error (#819) now names
  the recorded attempt's own `rung` field as the way to pin a rung
  explicitly instead of probing (#820). Wording only; still a usage error
  under the same conditions.

## [0.9.5] - 2026-08-25

An `extract` observability release (#784): a run now records enough to
find product problems after the fact — always on, beside the batches,
with the batch wire format untouched. Every written batch gets a trace
joining each item back to the text and the completion that produced it
(ADR 0023), every loss paired with the original text it was about (ADR
0024), the document's own paragraph coverage (ADR 0026), and what
taguru itself put into the prompt (ADR 0027); an attempts log keeps
every prompt and every answer in full, correction links, and the retry
machinery's own moves (ADR 0025 / 0028 / 0029). `taguru anchoring`
judges written batches against their own passage text, and
`scripts/extract_metrics.py` folds the records into per-document /
context / group / run metric tables; `docs/extract.html` gains the
reading guide — trace walkthrough, observation → knob table, and the
four axes (#794). No response-shape change; the new records are
additive sibling files and additive sidecar/checkpoint fields (older
checkpoints still load).

### Added

- `docs/extract.html` gains a reading guide for the records above
  (#794): a `jq` walkthrough from one batch line back to the piece,
  the full conversation, the correction chain, and the losses; an
  observation → knob tuning table (thresholds land after the #780
  baseline); and the four #784 questions each metric answers.

- `extract` writes a per-document **trace** beside every batch —
  `<out>/.extract-trace/<batch name>`, always on — joining every batch
  item (by its content key; the batch format is unchanged) to the piece
  of text and the completion that produced it: `document`, `chunk`,
  `piece` (`piece_id` = the checkpoint unit's sha256; `attempt` =
  `{run_id, attempt_seq}`, the Stage 2 corrective completion when one
  replaced the answer; `reused` for a checkpointed unit), and `item`
  records (ADR 0023, #785). The diagnostics sidecar now opens with a
  `kind: "run"` record and every `attempt` record carries `run_id`,
  `attempt_seq`, and `piece_id` (additive). Checkpoint units record the
  completion that produced them (`attempt`, defaulted on older files).
- The trace records every item the model's accepted answer held that
  the batch does not — mechanically removed (ADR 0013, Stage 1 and the
  Stage 2 alias prunes), dropped by `merge`'s contract, or folded as a
  duplicate — as a `loss` record with the item exactly as written, the
  rule, and **the original text** it was about (the cited paragraph,
  else the whole piece) (ADR 0024, #786). Removals are structured at
  the point of removal (`path`/`reason`/`item`; the report line,
  stderr, and the sidecar's `removed_items` strings are unchanged);
  checkpoint units store them so, and pre-0.9.5 checkpoints still
  load. Under `--lossy`, array elements that are not objects — dropped
  at parse — are recorded the same way.

- `extract` keeps every completion's **full prompt and full answer**
  beside the batch — `<out>/.extract-trace/<batch stem>.attempts.jsonl`,
  on by default (`TAGURU_EXTRACT_TRACE_ATTEMPTS=off` to disable): the
  system prompt once by hash, every other turn and the answer in full,
  the sidecar's ids and classification on each record; kept when a
  document fails, appended to on a checkpoint resume (ADR 0025, #788).
  The diagnostics sidecar and `TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES`
  are unchanged.

- The trace records the document's own side of coverage: one
  `paragraph` record per canonical paragraph (`bytes`, `items`,
  `covered`, and the paragraph's text exactly when no kept item cites
  it) and, under `--coverage`, one `uncovered` record per gap with the
  full sentence — stderr's byte-capped quote is unchanged — the
  paragraph's text, and the owning chunk (ADR 0026, #787).

- The trace records the prompt's steering lists as data (ADR 0027,
  #789): one `steering` record per document, placed right after the
  `document` record, with `chunk_index: null` (document scope — every
  chunk sees the same lists; per-chunk context, when it arrives, adds
  records with the index set). Fields: `candidates` (ADR 0014's list
  as offered, capped at 100; `[]` when `--candidates` is off or the
  document yields none), `vocabulary` (`[{label, count}]` in prompt
  order — count desc then label asc, capped at 200, computed by the
  same code that renders the prompt block; `[]` when nothing has
  accumulated), `context_names` (`--vocabulary`'s prompted list,
  capped at 200; `[]` without the flag), and `schema`
  (`{types, constrained_relations}`, each capped at 200; `null`
  exactly when no schema block was prompted — no `--schema`,
  `mode: off`, or a schema with no types and no constrained
  relations).

- Corrective attempts name the attempt they correct: `attempt` records
  in the attempts log and the diagnostics sidecar gain `corrects`
  (`{run_id, attempt_seq}`, present exactly on corrective attempts —
  Stage 1: the same piece's previous attempt; Stage 2: the accepted
  attempt whose output is corrected; absent for a Stage 2 correction
  of a pre-0.9.5 checkpointed unit). The correction tuple — flagged
  issues → ask → answer → adoption, against the original text — and
  the correction success rate are joins over existing records (ADR
  0028, #790).

- The ladder's moves are records (ADR 0029, #791): the attempts log
  gains `kind: "move"` — `escalate` (`from_max_tokens`/`to_max_tokens`),
  `demote` (`from_rung`/`to_rung`), `split`
  (`piece_bytes`/`split_cap`/`sub_pieces`, the reason telling the
  output cap from a timeout) — each with `run_id`/`piece_id`/
  `chunk_index` and the stderr line's reason. Attempt records in the
  sidecar and the attempts log gain `transport_retries` (always
  present; the 429/5xx/transport tries folded into one attempt, now
  counted). stderr is unchanged.

- `scripts/extract_metrics.py` (#792): folds the trace and attempts-log
  records into metric tables — loss rates by item kind and reason,
  paragraph coverage (count- and byte-weighted), correction success,
  attempt-state and ladder-move counts, label concentration, graph
  shape, and time/token/money cost — per document, context (via a
  `--ledger` mapping), group, and run, as JSON and Markdown;
  `--compare` diffs two runs per document with improved/worsened
  counts. python3 standard library only.

- `taguru anchoring` (#793): judges extraction batches against their
  own passage text, offline — the anchoring rate (subject and object
  present in the cited paragraph, else the passage, as a
  `normalize_entry` substring match) in a **strict** (own spelling)
  and a **with-aliases** (alias group: the batch's own concept
  aliases plus `--vocabulary`'s context aliases) variant, the gap
  being the alias-dependent share, and **locator validity** (cited
  paragraphs that actually hold the subject or object). Needs only
  batch files, so it applies to pre-0.9.5 output unchanged; `--json`
  writes the report `scripts/extract_metrics.py --anchoring` folds
  into its tables.

### Changed

- `extract`'s stderr lists a document's mechanical removals per chunk
  in removal order, rather than the Stage 2 alias removals first
  (#786).

## [0.9.4] - 2026-08-23

An `extract` hardening release, driven by running the pipeline over
real Japanese corpora against a local model. The escalation ladder is
now finite and always reaches the split rung: the resend is capped at
`TAGURU_EXTRACT_ESCALATION_FACTOR` × the budget (ADR 0019, #761), a
timeout descends to the split rung instead of four same-size retries
(ADR 0020, #762), `--structured-output auto` demotes its rung at run
time when a chunk exhausts the ladder (ADR 0021, #760), and
`--chunk-bytes` exposes the per-call document size (ADR 0020). Output
quality fixes ride along: an alias an earlier document already settled
on, or one the corrective turn cannot fix, is removed with accounting
instead of breaking import or failing the source (#758; ADR 0022
/ #763), and single-character relation labels no longer snowball
through the reuse vocabulary (#759). Docs gain Japanese
embedding-model guidance (#764). No on-disk format or response-shape
change.

### Added

- `taguru extract --chunk-bytes N` / `TAGURU_EXTRACT_CHUNK_BYTES`
  (default 24576, at least 512): the document bytes per model call,
  previously a fixed 24 KiB. Lower it for a slow local provider or
  output-dense documents (statutes, minutes) whose answers outrun the
  time or token budget at the default cap. A manifest/checkpoint
  computation input when non-default, so existing manifests keep
  matching (ADR 0020, #762).

### Changed

- `taguru extract`: an association whose relation label is a single
  character (most often a bare Japanese particle picked up as the
  whole relation, e.g. `は`) is now removed mechanically, with the
  usual accounting — a single character functions as no relation at
  all (unusable for query/paths/schema), and because labels
  accumulate into the run's reuse vocabulary, a survivor got
  suggested back to every later chunk, snowballing into unrelated
  associations sharing one meaningless label (measured: 41/41
  associations in one Japanese document). A two-or-more-character
  label is unaffected — same anchor-nothing judgment `--candidates`
  already applies to single-character names. The "relation labels
  already in use" prompt block now also carries a reuse count
  (`label (×N)`) and ranks the most-reused label first, so a
  one-off label is no longer visually indistinguishable from an
  established one (#759).
- `taguru extract`: a cross-chunk alias issue (a spelling that shadows
  an association name, a mapping that conflicts with an earlier one)
  that the one Stage 2 corrective turn leaves standing is now removed
  with accounting — named on stderr, counted on the report line, in
  the diagnostics sidecar — instead of failing the whole source (ADR
  0022, supersedes ADR 0013 §3.3 for alias items only). An alias
  records a spelling variant, never a fact; failing the document over
  one uncorrectable alias cost every fact it held. A standing issue
  about an association (schema domain/range) still fails the source.
  A document that fails on a later chunk now says how many units are
  checkpointed and that a rerun without `--force` resumes from them
  (#763).
- `taguru extract --structured-output auto` now demotes its rung at
  run time (ADR 0021, supersedes ADR 0001 §6's once-per-run
  resolution for `auto` only): a probe-verified json_schema rung can
  still loop on a real document (15,000 tokens of schema-shaped
  output for a 2 KB abstract, measured), so when a chunk exhausts the
  ladder under a constrained rung — `length` at the budget and again
  at the escalated resend, or a timeout — the run drops one rung
  (json_schema → json_object → prompted JSON), reports it on stderr,
  and restarts that chunk at the ladder's top; only a chunk with
  nothing left to demote splits. Run-wide, never reversed, and only
  for `auto` — the pinned modes keep ADR 0001 §7's split (#760).
- `taguru extract`: under the escalation ladder (`--max-output-tokens`
  or `--structured-output`), a completion that runs
  `TAGURU_EXTRACT_TIMEOUT_SECS` out now descends to the split rung —
  the same next step as `finish_reason: length` — instead of being
  retried four times at the same size and then failing the source
  without ever splitting. A timeout is never escalated (a larger
  output cap cannot make a slow piece faster); at the split floor the
  source fails after one attempt with the timeout named and the two
  knobs that would have helped. Transport failures, 429 and 5xx keep
  their four attempts, and the legacy path (no controls engaged) is
  unchanged (ADR 0020, #762).

- `taguru extract`'s escalation rung is now capped (ADR 0019,
  supersedes ADR 0001 §7's "resend with no cap"): when an answer ends
  at `--max-output-tokens`, the one neutral resend is sent at
  `TAGURU_EXTRACT_ESCALATION_FACTOR` × the budget (default 2) instead
  of uncapped. A local model that loops under constrained decoding
  never ended the uncapped resend with `length` — it ran the
  `TAGURU_EXTRACT_TIMEOUT_SECS` timeout out, was retried as a
  transport failure, and failed the source without ever reaching the
  split rung (10–25 minutes per chunk measured). Capped, the loop
  ends with `length` and falls through to the split. `0` restores the
  uncapped resend. The factor is a manifest/checkpoint computation
  input only when non-default under a budget, so existing manifests
  keep matching; diagnostics' `requested_max_tokens` now shows the
  escalated cap on the resend attempt (#761).
- docs: `nomic-embed-text` is a poor pick for a Japanese corpus (a
  measured resolve inversion); the local RAG walkthrough and
  troubleshooting now point at a multilingual model
  (`embeddinggemma` over Ollama, or the in-process `local`
  provider's `multilingual-e5-small`/`-base`) instead, and explain
  why `taguru calibrate` can still report `OVERLAP` on a healthy
  multilingual model when concept-name glosses are only a few
  characters long (#764).

### Fixed

- `taguru extract` no longer writes an alias whose spelling an earlier
  document of the same run (or the `--vocabulary` context) already
  settled on as a different concept or label. Import refuses that
  rewire (`409 … already resolves to a different record`) and stops
  the stream with the earlier batches applied; the alias is now
  removed by the mechanical pass (ADR 0013) with the usual
  accounting — named on stderr, counted on the report line, in the
  diagnostics sidecar — and the consolidation audit proposes the
  merge later if the two spellings really are one entity. Names are
  claimed exactly where the label vocabulary accumulates: when a
  document lands and when a manifest-skipped document's batch is
  reread (#758).

## [0.9.3] - 2026-08-20

A hardening release: the codebase-wide audit series finished (#538
through #560), and every real bug it surfaced ships fixed here — the
router's single-instance divergences (#727), ingest deadline threading
(#728), the MCP bridge's tool-name refusal (#732), `taguru-code sync`'s
gitignore re-check (#733), benchmark's extraction-knob pinning (#734),
Bearer parsing per RFC 7235 (#731), both SDKs' packaging fixes
(#735/#740), extract's batch-name injectivity (#730), and the
export/remote/communities fixes (#751/#752). The
passages/hydrate/embedding consistency follow-ups (#709) and the
fastembed 6.0.0 upgrade (#710) round it out; the audits' test-gap
findings closed alongside, with the touched modules' mutation sweeps
back at zero missed (#708, #723/#724). One on-disk naming change:
`taguru extract`'s batch output files now carry a hash suffix (see
Fixed).

### Changed
- `taguru communities` and `taguru consolidation` share one
  `POST /import` chunk packer (`remote::pack_import_chunks`) instead
  of two near-identical private copies; consolidation's bodies now
  join batches with `"\n"` like communities' (was `"\n\n"` — the
  import parser skips blank lines, so the wire behavior is
  identical), and its `create` block now rides only the run's first
  batch, the communities pattern (`create` is consumed only when the
  artifact context does not exist yet, so repeating it on every batch
  was pure payload) (#752).
- `fastembed` updated to 6.0.0 (from 5.17.4); its only breaking change
  is an internal error-type refactor with no source-level impact here.
  `LOCAL_MODELS.download_mib` (and `taguru-code models`'s
  `download_mib` field) is re-measured against each model's actual
  cache-directory footprint — weights plus tokenizer/config, not just
  the ONNX file — and is slightly higher for all five models than
  before; `dims`/`license`/`e5_prefix`/`multilingual` are unchanged
  (#710).
- `passages.rs`'s snapshot decoder now `checked_add`s every wire-length
  offset before slicing, matching the discipline `embedding.rs::take`
  and `context::image`'s `Reader::take`/`checked_arena_str` already
  hold — no behavior change on 64-bit hosts, but a corrupted or
  hostile snapshot can no longer wrap `pos` around instead of being
  refused (#709).
- `PassageVectorStore::ensure_ann_index` no longer blocks a concurrent
  search through another thread's whole ANN build (0.6–1.3s at
  `PASSAGE_ANN_THRESHOLD`) with no visibility into its own deadline.
  The wait is now chopped into deadline-aware slices, and a search
  that runs out of budget while waiting falls back to the exact sweep
  for that call — the same substitute `top_matches` already takes
  when the deadline is too tight to build the index at all (#709).
  This is an internal behavior change; it does not change public
  declarations, wire formats, or response shapes.

### Fixed
- A full `taguru export` (no CONTEXT arguments, local or `--url`) now
  owns `--out`'s `*.jsonl` files: a stream left by an earlier export
  whose context or group no longer exists is removed (each removal
  reported on stdout), so importing the directory can no longer
  resurrect a deleted entity. Subset exports never prune, non-stream
  files are never touched, and a remote run that could not enumerate
  groups prunes nothing (#751).
- `taguru export --url`'s per-group fetch now validates the response
  through the same parser `taguru import` trusts — exactly one
  `taguru_group` record — instead of "parses as JSON", so an
  unexpected 200 body can no longer land as a `.group.jsonl` file
  reporting 0 members (#751).
- `export --url`, `compact --url` (both modes), and `consolidation
  --url` now refuse an unparseable or non-http(s) base URL upfront as
  a usage error (exit 2), the way `import --url` already did, instead
  of failing mid-run with exit 1 after printing their preamble (#751).
- `taguru extract`'s batch output names carry an unconditional hash
  suffix (`docs__aomine.md-<sha16>.jsonl`), the same injectivity fix
  checkpoint names received in #227: path flattening is not injective
  (`a/b`, `a:b`, and `a__b` all flatten to `a__b`), and while one
  run's collisions were caught, separate runs into the same `--out`
  silently overwrote each other's batch files (#730). Unchanged
  documents recorded under the old naming stay skippable (the skip
  path reads the manifest's recorded output name), and a changed
  document re-extracting under the new name removes its recorded old
  file so `taguru import DIR` never sees the pair as a duplicate
  source.
- `taguru extract`'s fabrication diagnostic names BOTH positions when
  an association's subject and object are each absent from the
  document, instead of only whichever field was checked first (#730).
- Bearer credentials parse per RFC 7235's `auth-scheme 1*SP token68`:
  one or MORE spaces between `Bearer` and the token authenticate
  alike (strictly one space used to be required); no separator at all
  is still refused (#731).
- `taguru-mcp`'s refusal for a tool call whose `params.name` is
  missing, empty, or not a string now says "tool name is required"
  instead of the baffling `unknown tool ''` (#732).
- `taguru-code sync` re-checks every incremental candidate against the
  current universe (`git ls-files --cached --others
  --exclude-standard`) before importing (ADR 0010 §3, issue #733): a
  file newly gitignored — `git rm --cached` plus a staged
  `.gitignore`, nothing committed — still exists on disk and used to
  re-import (the path secrets would leak in by). A candidate outside
  the universe now retracts instead, sweeping out whatever an earlier
  sync had imported for it.
- `taguru benchmark` pins the remaining five `TAGURU_EXTRACT_*` knobs
  explicitly per cell (ADR 0003 §5, issue #734):
  `CORRECTIVE_CONTEXT_BYTES`, `COVERAGE`, `DIAGNOSTICS`,
  `DIAGNOSTICS_RAW_BYTES`, and `SCHEMA` are now set (at extract's own
  defaults) after the namespace scrub and recorded in
  `manifest.json`'s `extraction_settings`. The two numeric knobs read
  `""` as unset now, matching the path-valued keys, so "explicitly the
  default" has a spelling.
- `taguru benchmark compare` warns when one model's runs resolved
  DIFFERENT structured-output rungs (`structured_output_resolved` in
  the manifest) — the ADR 0003 §6 safety mechanism that had a writer
  but no reader; mixed-rung aggregates are called out instead of
  passing silently (#734).
- `taguru router` closes four divergences from a single instance
  (#727): `GET /groups`/`GET /contexts` floor an explicit `limit=0` to
  one row (`clamp_page`), matching the shard's own keyset-page
  contract so SDK iterators never read a sized-zero page as
  end-of-collection; `POST /maintenance/compact` answers 502
  `shard_unreachable` when NO shard could be asked, like `/flush`,
  instead of an empty 200 sweep report; a group PATCH's
  `remove_contexts` members are no longer existence-checked (a single
  instance treats removals as an idempotent set difference); and a
  group import outcome is labeled from the union's view — an empty
  created projection beside unchanged siblings answers `unchanged`,
  never a false `replaced`. Import refusal rewraps also count landed
  batches/schemas from each request's own ranges (a shard success with
  an unreadable envelope still landed), and `TAGURU_ROUTE_MAP` refuses
  a host-less shard URL (`http://`) at parse time with its line
  number.
- `apply_batch` (the shared write path behind `POST /import`, `POST
  /contexts/{name}/promote`, `taguru import`, and `taguru-code sync`)
  now threads its caller's deadline into the association writes
  instead of writing under `Deadline::unbounded()` between the HTTP
  loop's own checkpoints — a spent budget refuses inside the batch
  (`deadline_exceeded`; the import marker stays and the retry is
  exact). The offline CLI entrances keep their unbounded runs (#728).
- `taguru import --url`'s mid-stream refusal diagnostics now count
  unsent GROUP records beside the existing batch and schema tallies
  (#728).
- TypeScript SDK: `require("taguru/testing")` no longer crashes at
  load — the CJS bundle derived its module directory from
  `import.meta.dirname`, which tsup's esbuild leaves empty in CJS
  output, so `REPO_ROOT` initialized from `resolve(undefined, ...)`.
  The directory now comes from `import.meta.url` in the ESM output
  and `__dirname` in the CJS one, and a new `check:cjs` step in the
  SDK CI executes both CJS bundles under `require()` so this class of
  break cannot ship silently again (#740).
- Python SDK: `_models.py`'s `__all__` names all 102 defined model
  classes (39 were missing); a new AST-based unit test keeps the list
  and the definitions from drifting again. No runtime change —
  `taguru`'s public re-exports were already complete (#735).
- Hydration's published-file fetcher (`fetch_published_if_stale`) now
  sends only `InvalidData`/`NotFound` errors through the manifest
  re-read arbiter, matching the log-lane fetcher's existing
  selection. A permanent failure (a permission error, or anything
  else `ship::fetch` collapses to `io::ErrorKind::Other`) used to
  burn all `FETCH_REFRESH_ROUNDS` re-reads (~600ms) before surfacing;
  it now fails on the first attempt (#709). This does not change
  public declarations, wire formats, or response shapes.

## [0.9.2] - 2026-08-17

An observability release: the #549 audit of the metrics/tracing layer
and everything it surfaced, plus the cross-search span asymmetry #690
had flagged since v0.9.1. No wire format or API response shape
changes; the Prometheus exposition is byte-identical (the breaker
families' renderer was deduplicated, not reworded). Alongside the
fixes, the audit's measured test gaps closed (issue #699): the
assemble-evidence tree, the search lanes' cross-thread span hand-off,
and the stdio bridge's `taguru.tool_call` span are now verified
against a real OTLP collector, and the embedding/hydrate/passages
modules got their own audited mutation sweep (issue #550, findings
tracked in #708–#710).

### Added
- Cross-context passage search (`POST /sources/search`) now traces:
  one `taguru.passage_search` span per request (`taguru.context.count`
  marks the fan-out width, cache outcomes recorded exactly as the
  single-context span does) with one `taguru.passage_search.target`
  child per target — which also re-parents each target's BM25/ANN/fuse
  lane spans into the request's trace; previously they exported as
  parentless traces of their own (issue #690).
- The router's transparent per-context proxy routes
  (`/contexts/{name}`, `/contexts/{name}/{*rest}`) now dispatch under
  the same `taguru.shard_call` client span the fan-out verbs use, and
  inject the router's own span into the forwarded trace headers — the
  one hop where a trace still stopped dead at a process boundary.
  Response bytes stay untouched, and with OTLP export off the
  pass-through is bit-identical to before (issue #696).
- ADR 0008's registered-but-unimplemented vocabulary now ships (issue
  #697): `taguru.rerank.outcome` (the `taguru.rerank` span moved into
  the §12 decision driver, so pre-flight refusals and the permutation
  verdict reach traces too), `taguru.search.floor` (the effective
  cosine floor, on the single and per-target search spans),
  the `deadline_exceeded_before_start` skip event on the remote-MCP
  pre-flight refusal, and the communities lane's `taguru.op`/
  hit-count attributes plus a stable `no_communities_artifact` skip
  reason. `taguru.resolve.tier` is documented as deliberately
  metrics-only (a per-cue verdict has no per-cue span to ride).

### Changed
- Every `taguru.*` count/index/bytes span attribute now exports as a
  real OTLP intValue (issue #697): the code recorded raw `usize`/`u64`,
  which this tracing stack serializes as text — backends could neither
  aggregate nor compare any count. 66 sites swept to explicit `i64`.
- `taguru.search.ann`'s `taguru.search.pool` records the EFFECTIVE
  pool cap (clamped to the row count) instead of the raw request —
  explain's unbounded sweep passes `usize::MAX`, which previously
  wrapped to `-1` on the wire.
- With OTLP export enabled, the proxy hop rewrites `traceparent` to
  the router's own span (see Added); a deployment that relied on the
  bare pass-through of the caller's header will now see the router's
  hop in its traces — which is the fix, but worth knowing when
  comparing traces across the upgrade.

### Fixed
- The flaky `tracing_router` shutdown hang (issue #693): the test
  harness's `stop_gracefully` now escalates — 30s grace, a second
  SIGTERM, then SIGKILL with bounded reaping — and self-diagnoses
  which shutdown stage wedged, so a recurrence names its cause
  instead of timing out the whole suite.
- `Histogram::observe` computed `elapsed.as_micros()` twice per
  observation; the metrics module doc no longer claims a cheaper
  hot path than the per-route counters actually have (issue #700).
- docs/tracing.html and sdk/spec/tracing.yaml realigned with the
  shipped vocabulary (issue #698): the never-existent
  `taguru.mcp.retrieve` span name corrected to `taguru.retrieve`, the
  assemble-evidence tree described as it actually nests (no
  `describe`, `taguru.passages` as a first-class lane, opt-in
  `taguru.communities`), `taguru.embed` placed as
  `taguru.passage_search`'s direct child (not under the ANN span), and
  the reason/attribute tables completed (`bridge_unreachable`,
  `taguru.fallback.reason`'s four values, `schema_load_failed`/
  `schema_write_failed`, the count-attribute family).

## [0.9.1] - 2026-08-16

### Added
- Python/TypeScript SDKs: `promote`/`analyze_communities`/`embeddings_status`
  — the last three HTTP endpoints with no client-side coverage
  (`POST /contexts/{name}/promote`, `GET /contexts/{name}/communities`,
  `GET /contexts/{name}/embeddings`), all three already documented in
  `src/llm-protocol.md` but missing from `sdk/spec/surface.yaml` and
  both SDKs (issue #625).
- `/metrics` gains four series: `taguru_embed_slot_waiters`,
  `taguru_embed_slot_waits_total`, and `taguru_embed_slot_timeouts_total`
  expose contention on the embed concurrency slots (issue #563), and
  `taguru_replication_permanent_errors_total` counts shipper/tailer
  errors classified as permanent deployment problems
  (`PermissionDenied`/`Unauthenticated`/`NotSupported`/
  `UnknownConfigurationKey` — escalated to `taguru::audit` instead of
  being retried forever as if they were network blips) (issue #616).
- Compaction outcomes say whether the rebuilt image actually landed
  on disk: `CompactOutcome` gains `image_persisted: bool` and
  maintenance-compaction reports gain `skipped` (a context whose entry
  failed to load surfaces as a skip, not silence). Both additive with
  serde defaults, so payloads written before the upgrade still parse
  (issue #586).
- `sources/search/explain` responses gain `limit_to_reach_reason`
  (omitted unless set): `"unreachable"` distinguishes "no limit would
  surface this paragraph" from a merely absent `limit_to_reach`
  (issue #601).

### Changed
- `ApiError` gains an additive `issues_total: usize` field (present only
  when nonzero) — the failure-side counterpart to
  `ApiResponse.schema_violations`: `issues`' true count surviving
  `validation_error`'s own `MAX_LISTED_ISSUES` truncation, machine
  readable rather than only ever parseable out of the refusal's prose
  ("N issues total; showing the first 20"), which stays as-is alongside
  it (issue #623). The same series also tightens the error surface: the
  source-filter tag-count refusal now answers `over_limit` instead of
  `invalid_argument`; `POST /contexts/{name}/schema/audit` rejects
  unknown body fields with `400 malformed_request` instead of silently
  ignoring them; the keyset listings' `prefix`/`after` cursor strings
  are capped at 1032 bytes (previously unbounded, the one request
  string that never reached a length gate); and a capacity-exceeded
  race deep inside `/import`/`/promote` apply streams now answers with
  the structured `stream_refusal` shape instead of a bare access
  message.
- `taguru restore`'s exit codes now follow the documented house rule
  end to end: a malformed URL, unrecognized scheme, or bad flag exits
  2 (usage), while a store that parsed fine but refused to open — bad
  or missing credentials, a rejected cloud config, an unusable local
  path — exits 1 (bucket unusable) alongside "no fence, no complete
  generation". A failed restore also cleans up what it already wrote
  under `--out`, so a retry no longer needs the operator to empty the
  directory by hand (issue #616).
- `taguru restore` refuses a generation whose `complete` marker
  predates the manifest format instead of attempting the legacy
  segment-listing restore, which has been removed (issues #618/#619).

### Fixed
- `DELETE /contexts/{name}` racing a rename of the same context now
  returns `409 conflict` ("mid-rename; retry after it completes")
  instead of `500`, and no longer writes a "context deleted" audit
  line for a delete that did not happen (issue #561).
- `limit=0` on the paged listings — the context directory,
  source/alias/group/coverage listings, and
  `GET /contexts/{name}/changes` — now returns one item instead of an
  empty page, so `0` no longer reads as EOF to a cursor loop
  (issues #585, #676).
- `sources/search/explain` contract repairs: early-return refusals now
  still carry the `filter` report (previously always absent on those
  paths), and `score`'s scale switch and `cutoff_score` both derive
  from whether the vector lane actually ran — explain can no longer
  claim a fused scale for a search that scored lexical-only
  (issue #601).

## [0.9.0] - 2026-08-09

### Added
- docs: `taguru-code` now has its own reference page
  (docs/taguru-code.html) — the full verb surface
  (`sync`/`watch`/`find`/`tree`/`status`/`evalset`/`eval`/`models`
  with every flag and default), the ripgrep-equivalent universe, the
  `.taguru` data directory, ranked lookup without aliases, the
  accuracy gate, and the `TAGURU_USAGE_LOG*` knobs. Previously only
  `taguru-code models` appeared anywhere outside `--help`.
- `taguru router` hot-reloads `TAGURU_ROUTE_MAP` (#515): SIGHUP swaps
  the map at runtime (unix; an unreadable file logs a warning and the
  map already serving stays in effect) and a content-digest file
  watch picks up edits with no signal at all — keyring-style, no
  restart. In-flight requests keep the one map snapshot they started
  with, so a swap never mixes two maps inside one fan-out.
- `POST /contexts/{name}/promote` and the `promote` MCP tool (#466
  S2, ADR 0018): graph-path memory promotion — the named scratch
  sources move into an established destination context as the
  export/import round trip in one call, no LLM anywhere in the path.
  Each source moves whole (passage, `date`, tags, only its own share
  of every edge's weight; aliases exactly when their canonical is
  live in the promoted slice, the rest counted in `aliases_dropped`),
  source ids survive so promoted citations still name the session,
  and applying is per-source retract-then-apply — re-promotion is
  idempotent. The destination is never created and its own schema
  judges the incoming batches; a missing source id refuses the whole
  request path-addressed. After a real apply the destination's
  consolidation audit (all three checks, default ceilings) rides back
  under `audit` — candidates to judge, never applied — and
  `?dry_run=true` previews the same `batches` shape with nothing
  written. Write role, `retract_source`'s classification; the
  destination named in the body is scope-checked like `/import`'s
  body contexts.

- `taguru extract --source-id ID`, `--date WHEN`, `--tag TAG` (#466
  S1, ADR 0017): bake the promotion runbook's source conventions into
  the written batch — the `session:{agent}:{id}` header source (with
  the `/{doc}` stem suffix across several documents, collisions
  refused), and the passage line's `date` (`YYYY-MM-DD` or epoch
  seconds) and `tags`. All three are manifest computation inputs
  (`serde(default)` — older manifests keep matching default runs) but
  deliberately not checkpoint inputs: a metadata change rewrites the
  batch while reusing every cached chunk answer. `--date`/`--tag` with
  `--no-passage` is a usage error (metadata rides the passage line).
- `taguru extract --coverage` / `TAGURU_EXTRACT_COVERAGE` (#496 S4,
  ADR 0016): report every sentence that holds a candidate pair (two or
  more deterministically segmented document names) yet is covered by
  no extracted association — the systematic recall ceiling made
  visible, per document, with each sentence quoted on stderr and
  counted on the report line and in the diagnostics `document` record
  (additive `uncovered` field). Report-only and off by default: the
  batch is unchanged, the flag is not a computation input, and a
  manifest-skipped document is judged from its already-written batch
  with no model call.
- `taguru extract --vocabulary PATH` / `TAGURU_EXTRACT_VOCABULARY`
  (#496 S3, ADR 0015): steer a new document's spellings toward a
  target context's existing vocabulary, loaded from exported batch
  streams (file or directory). Concept names and labels are offered as
  preferred spellings (alias spellings never — they are the twins a
  canonical folds), harvested labels seed the relation-label block
  from the first document, and a context spelling is admitted by the
  mechanical occurrence check. Off by default; the harvested name
  set's content digest is a manifest/checkpoint computation input
  (`serde(default)`, so older manifests keep matching default runs).
  `taguru benchmark extract` forwards `--vocabulary` and records
  `extraction_settings.vocabulary_sha256` (additive, default `""`).
- `taguru extract --candidates` / `TAGURU_EXTRACT_CANDIDATES` (#496 S2,
  ADR 0014): offer the document's own names — segmented
  deterministically, dictionary-free — to the model as preferred
  subject/object spellings, preventing spelling twins at answer time
  instead of detecting them at audit time. Non-restrictive by contract
  and off by default; the control is a manifest/checkpoint computation
  input, so toggling it re-extracts. `taguru benchmark extract` gains
  the matching `--candidates` forwarding, and its `manifest.json`
  `extraction_settings` records two new fields — `candidates` and
  `lossy` (the latter closes a pre-existing resume-fairness gap):
  both additive with `serde(default)` (`false`), so existing results
  directories keep loading and match default-off/non-lossy reruns.
- Mutation testing for the core Python SDK: `mutmut` seeds faults and the
  new `sdk/spec/check_mutants.py` gate (a `python-mutation` CI job, run on
  the latest Python) fails on any surviving mutant not on the reviewed
  `sdk/python/mutation-baseline.txt` allowlist. `sdk/python-langchain`
  carries a `[tool.mutmut]` config for manual runs but is not gated (its
  coverage leans on the server-backed integration suite).
- Mutation testing for the core TypeScript SDK, the Stryker twin of the
  gate above: `npm run test:mutation` (sdk/typescript) seeds faults via
  Stryker against the hermetic unit suite, and the new
  `sdk/spec/check_mutants_ts.mjs` gate (a `typescript-mutation` CI job)
  fails on any escaped mutant not on the reviewed
  `sdk/typescript/mutation-baseline.txt` allowlist. `client.ts`/
  `testing.ts` lean on the server-backed integration suite (as
  `sdk/python`'s `_async`/`_sync` do), and `sdk/typescript-langchain` is
  not gated, mirroring `sdk/python-langchain`.

### Fixed
- `taguru-code <verb> --help` (and `-h`) now answers the usage and
  exits 0 for every verb — it used to fall into each verb's own
  parser as `unknown flag '--help'` (exit 2), and `watch --help`
  would have looped instead of answering. The usage text also now
  names every flag the verbs accept: the shared
  `--context`/`--data-dir`, plus `find --limit`, `evalset --sample`,
  and `eval --thresholds` were all undocumented in `--help`.
- Full-code audit, first pass (#520): `POST /import`'s group-restore
  timeout refusal now carries `integrity`/`durable_batches`
  machine-readably, like the batch loop's own timeout always did
  (additive response-shape change; a resuming importer no longer has
  to parse the durable count out of the message on this one arm), and
  a golden wire fixture now pins the `/import` refusal envelope.
  `promote` refuses over 1,000 `sources` with `over_limit` like every
  other list-shaped input — previously the one uncapped list, buying
  per-id validation work proportional to attacker-chosen size.
- Also #520: the embeddings reads (`GET /contexts/{name}/embeddings`,
  semantic twins/resolve/explain) now hold the entry's tombstone
  fence over their sidecar loads — racing a `DELETE` answers the same
  404 the context endpoint gives, never a 200 assembled from unlinked
  (or a successor generation's) sidecar files. Smaller: `benchmark
  compare` drops end-before-start wall-clock samples and lets a
  re-processed document's LAST `end` record supersede the first;
  evalset `case_id` emptiness is trimmed like every other field's;
  the passage ANN index no longer has a latent empty-store panic; a
  replica lane's behind-age clears on a shipped seq at or below the
  applied one.
- Full-code audit, second pass (#522): the consolidation audit's
  `contradiction.objects[].sources` no longer lists a source whose
  own accumulated sum cancelled to exactly zero (the posture
  `contested` already took) — **behavior change**: evidence rows and
  the fingerprints built over them move for groups that carried such
  a source; standing judgments on those fingerprints re-judge once.
  `checks` gains the shared 1,000-item `over_limit` cap.
- Also #522, replica correctness: a tailed context's in-memory meta
  (pinned, description, revision bookkeeping) can no longer freeze
  indefinitely when a per-request load wins the hydration race
  against a failed tailer poll — the tailer now carries the refresh
  debt across polls (recorded before the shared pass can fail) and
  pays it on the next poll. `replica_refresh` also re-stats
  `taguru_wal_bytes`/`taguru_passages_wal_bytes` (tailed copies never
  pass through the writer's live byte accounting; a cold unpinned
  context understated both indefinitely), zeroing only on `NotFound`.
  The pre-v5 image migration compares its two summation orders under
  a first-order rounding bound instead of exact f64 equality, so
  regrouping noise no longer credits a phantom attribution that
  `retract_source` can never remove. `taguru extract --source-id`
  refuses whitespace-only ids like `--tag`; `search_passages`/
  `explain_search` declare `minimum: 0` on `since`/`until` like their
  sibling tools.
- `POST /import?dry_run=true` now seeds each batch's checks with what
  the batches before it would intern and create, so two spurious
  mid-stream refusals the real import never raises are gone: an
  `UnknownCanonical` alias rejection when a stream's aliases trail
  their canonicals (every export — aliases ride the last batch), and
  a `no_context` refusal on every post-first batch of a restore into
  a fresh context name (the create block rides only the first batch).
  Cross-batch alias conflicts remain un-predicted — that direction
  only lets a preview pass what the real run would refuse, like the
  capacity caps the preview contract already documents as advisory.
- TypeScript SDK: ported the Python SDK hardening that had not reached
  the TypeScript twin — `contexts.delete`/`groups.delete` no longer
  auto-retry after an ambiguous transport failure (the retry turned an
  applied delete's success into `NotFoundError`), and
  `exportStream`/`exportToFile` now raise the SDK's `TransportError` on
  a failed connect or a mid-stream connection drop like every other
  call path.
- TypeScript LangChain: ported the Python LangChain hardening that had
  not reached the TypeScript twins. `DocxConnector`/`PptxConnector`'s
  decompression-bomb guard now measures the real decompressed size by
  bounded streaming inflation (shared `unzipWithinCap` in structure.ts)
  instead of trusting the zip's forgeable declared entry sizes.
  `HtmlConnector` detects UTF-16 (BOM'd or BOM-less) instead of reading
  it as UTF-8, refuses decoded text carrying NUL characters as
  `corrupt`, reports post-redirect failures under the final URL, and no
  longer flags a hidden `iframe` as `partial_extraction`. Only
  control-character-free vocabulary labels are folded into the
  extraction prompt (second-order prompt-injection path via planted
  labels). Single-context retrieval isolates the graph/text lanes like
  the cross-context path, and the graph lane fetches citations
  concurrently. `FilesystemCheckpointStore` holds an advisory
  per-source lock (PID-stamped lock file with dead-holder reclaim —
  Node has no stdlib `flock`) so concurrent runs stop clobbering each
  other's checkpoints, and the ingester proceeds without checkpointing
  on contention. `FileObjectStore.list()` skips an unreadable
  subdirectory instead of aborting the whole scan (an unreadable root
  still surfaces). Presigned-URL canonicalization also strips
  `X-Amz-Date`/`X-Amz-Expires`/`X-Amz-Algorithm`/`X-Amz-SignedHeaders`
  and their `X-Goog-*` equivalents. `PdfConnector` offers
  extraction-raising pages to a configured `OcrAdapter`, and
  `DocxConnector` with `extractHeadings: false` no longer derives
  `metadata.title` from a heading.
- Python and TypeScript LangChain: single-context retrieval with only
  ONE lane enabled now raises when that lane fails, instead of returning
  an empty result that reads as "nothing found" — completing the
  lane-isolation fix (which only covered the both-lanes-enabled case).
- Python LangChain: `HtmlConnector` now detects BOM-less UTF-16 (and
  refuses any text that still decodes with stray NUL bytes) instead of
  reading it as UTF-8 — a page that decoded without error but left
  `<script>` bodies in the extracted text now decodes correctly or is
  reported `corrupt`. A fetch that fails *after* a redirect
  (`>= 400` status or a non-HTML content type) now reports the final
  URL as the source id, matching the success path. `DocxConnector`/
  `PptxConnector`'s decompression-bomb guard now measures the real
  decompressed size by (bounded) decompression instead of trusting the
  zip's `file_size` header, which an attacker can forge small.
  `FileObjectStore.list()` skips an unreadable subdirectory instead of
  aborting the whole enumeration (a permission denial on the store root
  still surfaces). `TaguruRetriever`'s async graph lane cancels its
  still-in-flight citation fetches when one raises, and its
  cross-context walk no longer swallows a `CancelledError` returned by
  `asyncio.gather`. `FilesystemCheckpointStore`'s advisory lock is now
  fork-safe: a forked child re-acquires rather than trusting an
  inherited lock entry, so it correctly conflicts on a source the
  parent holds.
- Python SDK: `decode()` now refuses a wrong-shaped container inside a
  response (a `list` field fed a dict, a `dict` field fed a list) with
  `ResponseShapeError`, instead of silently taking a dict's keys or
  escaping as a bare `AttributeError` past `except TaguruError`.
  `contexts.delete`/`groups.delete` no longer auto-retry after an
  ambiguous transport failure — a repeat of an applied delete answers
  404, so the retry turned success into `NotFoundError`.
  `export_stream`/`export_to_file` now raise the SDK's
  `TransportError` on connection failures like every other call path.
  `backoff_delay` no longer overflows at extreme retry counts.
- Python LangChain: `PdfConnector` now offers pages whose text
  extraction *raised* to a configured `OcrAdapter` (previously only
  sparse-but-successful pages were recovered). `DocxConnector` with
  `extract_headings=False` no longer derives `metadata.title` from a
  heading. Presigned-URL canonicalization also strips
  `X-Amz-Date`/`X-Amz-Expires`/`X-Amz-Algorithm`/`X-Amz-SignedHeaders`
  and their `X-Goog-*` equivalents, so the same object presigned twice
  keeps one source id. `TaguruRetriever` with a single `context=` now
  isolates the graph and text lanes like the multi-context path
  already did (one lane failing no longer discards the other's hits),
  and the async graph lane fetches citations concurrently.

### Changed
- SDK client tracing (Python and TypeScript): the aggregate
  citation-miss telemetry is now the same event shape the server
  emits — one `taguru.skip` event with
  `taguru.reason=citation_passage_missing` and the count in
  `taguru.citation.missing`, on the `taguru.citations` phase span —
  instead of the SDK-only `taguru.citation_missing` event name
  (**telemetry shape change**: a dashboard or alert filtering on the
  old event name needs the new filter; one query now reads both
  producers). `citation_passage_missing` joins the shared skip-reason
  vocabulary in `sdk/spec/tracing.yaml`.
- Python LangChain ingest checkpoints are now written as JSON Lines
  and appended per chunk (**on-disk format change**; files from older
  versions still load, and a torn trailing line from a crash is
  discarded on resume). Saving is O(chunk) instead of rewriting the
  whole accumulated state each chunk. `FilesystemCheckpointStore`
  additionally holds an advisory per-source lock so two runs over the
  same source no longer clobber each other's progress — the loser
  proceeds without checkpointing and says so in a `RuntimeWarning`.
  `DocxConnector`/`PptxConnector` gain `max_decompressed_bytes`
  (default 8× `max_file_bytes`): a zip whose declared uncompressed
  size exceeds it is refused as `content_too_large` before parsing,
  closing a decompression-bomb hole the compressed-size cap left open.
  `TaguruIngester` folds only control-character-free labels from the
  context's vocabulary into its extraction prompt, cutting off a
  second-order prompt-injection path via planted labels, and now
  validates `chunk_bytes`/`vocabulary_cap` at construction.
- `taguru extract`'s strict (default) mode now removes mechanically —
  before spending any LLM corrective turn — items that could never
  import as answered: associations/aliases with a required field
  missing or empty, self-referential aliases, aliases whose canonical
  resolves to nothing, and subjects/objects that never appear in the
  document text (#496 S1, ADR 0013). The corrective turn is demoted to
  the last resort for what removal cannot judge (wrong-typed or
  out-of-range values, shadowing/conflicting aliases, schema
  violations). **Behavior change**: a source that previously failed
  after fruitless corrective turns on such items now succeeds with the
  removals named on stderr, counted on the report line as
  `removed (mechanical validation)`, and listed in the
  `--diagnostics-out` sidecar (`attempt` records gain `removed_items`,
  `document` records gain `removed` — both additive). `--lossy` is
  unchanged byte for byte. Extraction checkpoint files gain a
  `removed` field per unit (`serde(default)`; older files still load).

## [0.8.0] - 2026-08-05

### Added
- The document-erasure lifecycle is now complete (#437). `POST
  /contexts/{name}/sources/retract?dry_run=true` previews the same
  `{associations_touched, passage_removed}` the real call reports,
  with nothing written — no WAL op, no import marker, no audit line —
  exposing the exact read-only footprint check `/import?dry_run=true`
  already ran internally. And `POST /contexts/{name}/compact` (the
  maintenance sweep included) now also rewrites the passage log
  without retracted sources' text: previously the graph image dropped
  its dead records but a withdrawn passage's bytes lingered behind a
  tombstone until the log's own size-triggered compaction happened to
  run, so "retract, then compact" did not yet mean "erased." The
  compact response gains `passages_compacted` (additive,
  `#[serde(default)]` so `compact --url` reads older servers). The
  runbook — retraction withdraws truth, compaction removes bytes, both
  are needed for a deletion request, names survive, replicas compact
  separately — is documented in troubleshooting and the `/protocol`
  manual. `dry_run` rides the `retract_source` MCP tool and both SDKs
  (`sdk/spec/surface.yaml`).
- `GET /contexts/{name}/changes` (#422) — the polling change feed: one
  page of content-change events after an opaque cursor, so a client
  cache, an external index, or a recomputation trigger can ask "what
  changed since I looked" instead of re-listing everything. Events
  aggregate per write call, never per line (`associations_added{count}`,
  `association_retracted`, `aliases_added`/`removed`,
  `source_stored`/`source_retracted`, `schema_updated`), emitted
  inside the same lock that advances the revision counters so the two
  can never disagree, across every entrance — HTTP, MCP, and `/import`
  alike. Omitting `since` starts tailing (an empty page whose `next`
  is the position to poll from, the bootstrap after a full sync);
  `more: true` says events past `limit` are already waiting. The feed
  is a bounded in-memory ring per context, deliberately not persisted
  history — the per-context WAL truncates on every image flush, so
  there is nothing durable to serve — and a lost position (restart,
  delete-and-recreate, or falling further behind than the ring
  retains) answers the new 410 `stale_cursor` (additive to the error
  vocabulary) rather than a silently incomplete page: full resync,
  then tail again. Cursors are opaque and node-local. Exposed as the
  `changes` MCP tool and in both core SDKs (`changes`/`changes()`,
  `sdk/spec/surface.yaml`), pinned as wire-contract fixtures, and
  documented in the `/protocol` manual.
- `langchain-taguru` grows GCS and Azure Blob object-storage backends
  (#414, the connectors ADR 0007 §13 deferred): `GCSObjectStore` (the
  `gcs` extra) and `AzureBlobObjectStore` (the `azure` extra) implement
  the same `ObjectStore` protocol `S3ObjectStore` does, so
  `sync_object_storage` ingests a GCS bucket or an Azure container with
  the identical dispatch, two-layer checkpoint, and deletion policy —
  and `open_object_store` now speaks the exact scheme set the server's
  own replication path (`src/ship.rs`) does: `s3://`, `gs://`, `az://`,
  `file://`. Each backend reads only its cloud's standard credential
  chain (ADC, `DefaultAzureCredential`) with no parameter that could
  carry a key or SAS token, and classifies failures into the same
  transient/permanent split ADR 0007 §9 fixed for S3. GCS maps its
  always-present `generation` onto the checkpoint's strongest
  fingerprint tier (not gated on bucket versioning the way S3's
  `VersionId` is) and stands custom metadata in for object tags;
  Azure's blob index tags map onto object tags directly, with the same
  tags-only permission degrade.
- `watch_directory` (#414, the third and last connector ADR 0007 §13
  deferred): the continuous form of the local-tree sync — one
  `sync_object_storage` pass over a `file://` directory per poll,
  yielded as a `RunReport` from a generator the caller owns, with a
  cooperative `should_stop` honored before, during, and between
  passes. Polling over the existing two-layer checkpoint by design
  (an unchanged file is skipped on its `(size, mtime)` listing
  fingerprint without being opened), rather than a platform
  filesystem-event API that would add this package's first non-parser
  runtime dependency and still miss events on network and bind
  mounts. Stdlib-only; deletion stays report-only by default.
- `POST /contexts/{name}/paths` (#418) — the 手繰り between two
  concepts: every simple path from an origin to a target, shortest
  first, each trail carrying the whole concept `path` plus its
  associations in walk order with full attributions. `activate`
  spreads outward and `explore` sweeps a neighborhood; neither could
  answer "how are these two related?" without the client re-walking
  the graph by hand. Traversal follows `explore`'s exact discipline —
  bidirectional, labels never bridge, retracted edges never bridge,
  ADR 0009 §6.3's `schema:type` exclusion applies once a schema
  document exists — and ranking is deterministic: distance ascending,
  then weakest-link strength descending (the smallest raw cumulative
  |sum| along the trail — corroboration outranks a single emphatic
  assertion, the same discipline `activate` ranks by), then insertion
  order. Simple-path enumeration is combinatorial in the worst case,
  so one call examines at most a fixed edge budget and reports
  `capped: true` when it bites — `total` is then a lower bound, never
  a silently complete-looking count. `max_depth` shares explore's
  ceiling (10); `limit` defaults to 10, capped at 100 (each trail is a
  whole chain of associations, so pages weigh more than single-match
  endpoints). Exposed as the `paths` MCP tool, in both core SDKs
  (`paths`/`paths()`, `sdk/spec/surface.yaml` like every other
  cross-language method), pinned as a wire-contract fixture
  (additive: `HTTP_CONTRACT` unchanged), counted on `/metrics` as
  `taguru_searches_total{op="paths"}`, and documented in the
  `/protocol` manual's endpoint table and retrieval discipline.

## [0.7.0] - 2026-08-05

### Added
- Both core SDKs gain the rest of the schema surface:
  `put_schema`/`putSchema`, `audit_schema`/`auditSchema`, and
  `validate_schema`/`validateSchema`, alongside the existing
  `get_schema`/`getSchema` — closing the parity gap with HTTP and MCP,
  which already exposed all four; an SDK-only integration previously
  had to drop to raw HTTP to install, audit, or dry-run a schema.
  `audit`/`validate` decode the shared `SchemaAudit` shape
  (`{total, violations: [{association, issues}], untyped_concepts,
  undeclared_types, unknown_labels, reserved_alias_conflicts}`,
  ADR 0009 §10) with `violations` paging like every other match list;
  Python's `put_schema`/`validate_schema` accept either a plain mapping
  or the decoded `SchemaDocument` dataclass. Recorded in
  `sdk/spec/surface.yaml` like every other cross-language method.
- `POST /contexts/{name}/unreachable_from` joins ADR 0009 §6.3's
  traversal exclusion, amending #381's three-exclusion list: once a
  schema document is installed, `schema:type` edges are invisible to
  the coverage audit — never a bridge in the reachability walk (a
  shared type name would otherwise put every typed instance in one
  reachable component and silently under-report genuine orphans, the
  one failure mode this audit exists to catch) and never reported as
  orphans themselves, the same "never reported, never a bridge"
  contract `explore_excluding` documents. Backed by the new additive
  `Context::unreachable_from_excluding` (the same
  monomorphized-`visible`-closure pattern as `explore_excluding`, so
  the unfiltered path pays nothing); gated, like every §6.3 exclusion,
  on document existence alone, never `mode` — a schema-free context
  answers byte-identically to before.
- Schema metrics and a documentation reference page (#388, S10 of
  #218's ADR 0009 split §15) — closing the split: strict/warn's actual
  effect was previously invisible on `/metrics`, and the feature had no
  reference page or README mention. A closed `SchemaOutcome
  {Ok, Warned, Refused}` (`src/metrics.rs`, the same `const ALL`/
  zeros-always-emitted discipline `RerankOutcomeKind` uses) backs
  `taguru_schema_checks_total{outcome}`, counted only at the two write
  entrances a schema actually gates — `POST
  /contexts/{name}/associations` and a real (non-preview) `POST
  /import`/`taguru import` apply — for a context with an installed
  schema document (ADR 0009 §6.3's single condition, never `mode`); a
  schema-free context never touches it, so `ok` means "checked, no
  violation," not "no schema." `?dry_run=true`/`preview_batch` and
  `POST /schema/validate`/`/schema/audit` are diagnostics, not write
  gates, and are deliberately excluded — otherwise a validate-then-
  apply workflow would double-count the same refusal;
  `predicted_schema_rejection` now takes a `CheckPurpose` so its two
  callers (`apply_batch`, `preview_batch`) cannot disagree about which
  is which. `SchemaCheck::outcome` (`src/schema/check.rs`) is the one
  place §7.2 step 7's dispatch is named, so the metric can never
  describe a check differently than the check itself decided.
  `taguru_context_schema_violations_total{context}` is the per-context
  breakdown, riding the existing `PerContextMetrics {Off, All, Top(n)}`
  opt-in-and-bounded pattern behind `TAGURU_METRICS_PER_CONTEXT` — no
  new env var — backed by a plain `AtomicU64` on `Entry` (not
  `ContextUsage`, which rides the sidecar and `GET /contexts`'s wire
  shape; this is metrics-only and resets on restart). A new
  `docs/schema.html` reference page (linked into the sidebar of all 22
  existing pages plus `docs/index.html`'s card list) covers the
  document shape, the three modes, the reserved `schema:type` label,
  write-time enforcement, the HTTP/MCP surface, type-aware reads, and
  these two metric families; `README.md`, `docs/import.html`, and
  `docs/extract.html` (stale since #386 — `--schema`/
  `TAGURU_EXTRACT_SCHEMA` were undocumented, and its existing
  `--structured-output json-schema` text collides in name with the
  unrelated context schema) gain cross-links and the missing coverage.
  The file-family/downgrade hazard this feature's file family carries
  is documented under #379/#384 above, not repeated here.
- Types on `describe`/`resolve`, and type filters on `query` (#387, S9 of
  #218's ADR 0009 split §12) — the read-side minimum, closing the last
  gap between what a schema document declares and what retrieval can
  ever surface or narrow by. All three are gated by §6.3's single
  condition (an installed schema document for the context), never by
  `mode`: a schema-free context is byte-identical to today. `describe`'s
  `ConceptDescription` gains `types: Vec<String>` — every live
  `schema:type` object on the concept, collected inside the existing
  outgoing-chain walk (`Context::describe_typed`,
  `src/context/query.rs`); `schema:type` is not excluded from the label
  tally itself, only `activate`/`explore`, the vocabulary block, and the
  vocabulary audit's twin sweep are (§6.3). `resolve`'s `TieredResolution`
  gains `types: Option<Vec<String>>`, attached to the top 8 candidates
  the same one shared read `gloss` already rides
  (`Context::concept_types`); `resolve_label` candidates never carry it
  — a relation label has no type. `query` and cross `POST /query` gain
  optional `subject_types`/`object_types`: an OR-set of declared types,
  `is_a`-expanded through the schema's precomputed ancestor closures
  (`schema::expanded_type_sets`, factored out of the same live-read
  `SchemaEnv::build` already uses, so a filter can never disagree with
  what `strict` itself would treat as a concept's type) — a filter, never
  an anchor (leaving subject/label/object all unset is still refused).
  Evaluated after the position pins and before paging, so `total`
  reflects the filtered count; a schema-free target answers empty for a
  non-empty filter rather than erroring, evaluated per-context in the
  cross variant. Explicitly out of scope, per the ADR: using types to
  score or rank retrieval candidates. The MCP `describe`/`resolve`/`query`
  tools and the core Python and TypeScript SDKs (`sdk/spec/surface.yaml`)
  carry the same fields/options.
- Schema vocabulary in `taguru extract` and both LangChain ingesters
  (#386, S8 of #218's ADR 0009 split §11) — the producer side of the
  schema layer, closing the last gap between what a `strict`/`warn`
  context enforces on write and what the model is ever told about it.
  `taguru extract` gains `--schema FILE`/`TAGURU_EXTRACT_SCHEMA` (the
  same document shape `{stem}.schema.json`/`GET /contexts/{name}/schema`
  persist and serve); the offline extractor has no server to fetch one
  from (ADR 0009 §13 gives it no new credential surface either), so the
  operator hands it the document explicitly, and a file that fails to
  parse or fails `schema::install`'s own checks is a startup error, not
  a silent skip. Both LangChain SDKs instead pull it live, mirroring
  `_fetch_vocabulary`'s own best-effort `NotFoundError → None` posture
  — a schema-unaware server or a schema-free context works unchanged.
  When a schema is present and its `mode != off`, `system_prompt` gains
  one block after the existing vocabulary block: the allowed entity
  type names, one `label: domain → range` line per constrained relation
  (budget-capped like the vocabulary block, live-vocabulary relations
  first), and the instruction to assert types on the reserved
  `schema:type` label — deliberately never rendered as a JSON Schema
  `enum`, so a schema-constrained model can still propose a new
  relation. `PROMPT_VERSION` bumps 2 → 3 in all three producers
  (non-negotiable: a cached output from the schema-free prompt must
  never be silently reused now that a schema can shape it). A new
  `schema_output_issues`/`schemaOutputIssues` — structurally a sibling
  of `cross_output_issues`, same union-before-judgment, per-output-index
  shape — judges domain/range and `closed_labels` violations across the
  full answer set the same way `schema_issues`/`SchemaEnv`
  (`src/schema/check.rs`) already does server-side, and feeds the
  existing corrective-retry machinery unchanged: one targeted corrective
  turn per offending output, never a second round. The core Python
  and TypeScript SDKs gain `Context.get_schema()`/`Context.getSchema()`
  (`GET /contexts/{name}/schema`, `sdk/spec/surface.yaml`) as the
  client method both LangChain ingesters' `_fetch_schema`/`fetchSchema`
  call.
- `POST /contexts/{name}/schema/audit` and `POST
  /contexts/{name}/schema/validate` (#385, S7 of #218's ADR 0009 split
  §10). `audit` judges every live association against the resident
  document; `validate` takes a *proposed* document instead and never
  persists it — the pre-flight §7.1 promises before a `strict` flip.
  Both share one judgment (`schema_audit`, `src/api/schema.rs`) built
  on the same `schema_issues`/`SchemaEnv` pure check every write
  entrance already uses (S3, #381), so a finding here is exactly what
  `strict` would refuse for the identical fact. Deliberately
  mode-independent: §7.1's whole reason for this route to exist is that
  pre-existing violations are otherwise invisible, so `audit`/`validate`
  judge as `strict` would regardless of the document's actual `mode` —
  an `off` or `warn` context reports the same violations a `strict` one
  does. Five candidates-not-verdicts sections in one `DriftAudit`-shaped
  response, framed exactly like `audit_vocabulary`'s own doc — nothing
  is ever auto-applied: `violations` (domain/range mismatches, the only
  section that pages, worst-magnitude-first like every other match
  list), `untyped_concepts`, `undeclared_types` (§6.2, always reported
  regardless of `closed_labels`), `unknown_labels` (§6.4, only under
  `closed_labels`, never naming `schema:type` itself), and
  `reserved_alias_conflicts` (§6.3 guard 2's install-time bullet, read
  back — only reachable through `validate`, since `PUT /schema` itself
  already refuses to install over such a conflict). Deprecated-relation
  usage is explicitly **not** in scope here — §9.2 defers it until a
  follow-up ADR gives the document a field to mark a relation
  deprecated. Both routes are `Role::Read` and O(edges) with no cheap
  variant, joining the unconditional heavy-ops group beside
  `audit_vocabulary`/`compact_context` rather than `audit_drift`'s
  conditional-extension pattern. `audit_schema`/`validate_schema` MCP
  tools round-trip onto the same two routes.
- The `taguru_schema` export/import stream record and its replication
  parity (#384, S6 of #218's ADR 0009 split §13). `export::render`
  emits it as the FIRST line of a context's stream, only when the
  context has a schema and `mode != off` — a schema-free export, or
  one left in `off`, is byte-identical to before. `parse_stream` gains
  the record kind (`Stream.schemas`), `deny_unknown_fields` on every
  field (unlike a group record, none defaults — matching
  `SchemaDocument`'s own at-rest posture), and refuses an unread
  `taguru_schema` version in `parse_group`'s exact wording shape
  (`"taguru_schema N is not a version this taguru reads (it reads
  1)"`). Schema records install AFTER every batch of a stream, BEFORE
  any group — a record can name a context a batch of the SAME stream
  just created, and a group can rely on the schema already having
  landed — each installing independently through the same
  `AppState::put_schema` `PUT /schema` uses, so the first one that
  fails refuses the request right there with every batch before it
  already durable (unlike a group record set, which validates whole
  before any of it applies). `POST /import`'s response gains an
  additive `schemas: [{context, mode, types, relations}]` (omitted
  when the stream carried none — every existing response byte-
  identical); `taguru import --json` matches. `taguru router` gives a
  schema record its own routing: unlike a group record it is never
  broadcast, only sent to the one shard owning its context.
  `version_facts()`'s existing `schema_formats` gains its first real
  consumer: `export --url` and `import --url` (the latter only when
  the stream actually carries a schema record) read the peer's `GET
  /version` and refuse before a byte ships — naming both sides — when
  the peer cannot carry this build's schema format, replacing what
  used to be a schema-carrying stream falling through an old server's
  `parse_stream` to a misleading "not a batch header" refusal. A
  schema-free export/import is unaffected. **Downgrade hazard**, named
  per ADR 0009 §5.1: an older binary's `context_files` is a shorter
  array with no `{stem}.schema.json` entry, so downgrading past this
  change requires deleting any stray `.schema.json` files by hand —
  the older binary will not delete, move, or hydrate them itself.
- `strict`/`warn` schema enforcement on `POST /import` and `taguru
  import` (#382, S4 of #218's ADR 0009 split — the first write entrance
  to call S3/#381's `schema::schema_issues`; `POST
  /contexts/{name}/associations` is S5/#383). A new
  `predicted_schema_rejection`, checked right after
  `predicted_alias_rejection` and shared verbatim between `apply_batch`
  and `preview_batch` so a dry run can never disagree with the real
  import: `strict` refuses a batch whose associations would violate the
  context's schema — 400 `invalid_argument` (409 `conflict` for a
  batch's own `labels` alias resolving to the reserved `schema:type`,
  mode-independent per ADR 0009 §6.3), before anything mutates
  (`nothing_written`, or `durable_prefix` naming how many earlier
  batches in the stream already landed), with a path-addressed `Issue`
  per violation (`batches[{b}].associations[{a}].subject`/`.object`,
  `kind: "domain"`/`"range"`). `warn` applies the batch and reports
  instead: `ApiResponse` gains an additive `issues` array (present only
  under `warn` with at least one violation — every other response,
  `off` included, is byte-identical to before), and `ImportOutcome`
  gains `schema_violations: usize`, the true count surviving `issues`'
  own truncation. `off` and a schema-free context are unaffected —
  `predicted_schema_rejection` returns before a single lock is taken.
  `taguru import`'s report line gains `, schema warnings: N`; a remote
  `--url` run's per-chunk summary line does too. Wire fixture
  (`tests/fixtures/wire/http/import.json`) and both SDKs'
  `ImportOutcome` (`schema_violations`, defaulted/optional so an older
  server's response still decodes) updated to match.
- The reserved `schema:type` label's remaining namespace guards, its
  three read-side exclusions, and the shared pre-write check (#381, S3
  of #218's ADR 0009 split — library-level only, no write entrance
  wired yet, that is S4/#382 and S5/#383). `POST /contexts/{name}/aliases`
  now refuses a `labels` alias resolving to `schema:type` once a schema
  document exists for the context, in every mode including `off` (the
  gate is "a document exists," never "mode != off," per ADR 0009 §6.3).
  Once a schema exists, `schema:type` is invisible in three places it
  would otherwise quietly distort: `POST /contexts/{name}/explore` and
  `/activate` (plus the `assemble-evidence` endpoint's own activate
  lane) never traverse it — a type name is a hub, so without this a
  neighborhood search would put every instance of one type a couple of
  hops from every other; `GET /contexts/{name}/labels`'s default page
  and `?prefix=` both omit it; and `POST /contexts/{name}/vocabulary/audit`
  (and `audit_drift?include_twins`) never proposes a type name as a
  concept-spelling fork candidate. A schema-free context is unaffected
  in all three places — the exclusion costs one extra lock probe
  (`AppState::hidden_label`) and nothing else. `taguru extract`'s
  accumulated relation-label vocabulary (the `system_prompt` block a
  producer sees) also never offers `schema:type`, unconditionally,
  since extract has no notion of whether a target context has a schema
  at all. New library-level pieces, none yet reachable from a write
  path: `Context::explore_excluding`/`activate_excluding`/
  `label_page_excluding`/`canonical_concept` (additive siblings of the
  existing methods, so the published `taguru` crate's signatures are
  unchanged), and `schema::schema_issues` — the one pure function ADR
  0009 §7.2's `TypeEnv` union and §8's two new `Issue` kinds
  (`"domain"`, `"range"`) compile down to, built so the two future write
  entrances share it and cannot drift apart.
- `GET`/`PUT /contexts/{name}/schema` (#380, S2 of #218's ADR 0009
  split — the management routes over S1's file family; still no
  enforcement against the graph, that is S3/#381). `PUT` installs a
  schema document wholesale — `schema::install`'s validation (version,
  caps, `is_a` cycles/depth) plus a newly-added guard refusing a
  `relations` entry named `schema:type` (the reserved type-assertion
  label, ADR 0009 §6.3) and a migration-boundary guard refusing the
  install when an already-persisted `label_alias` resolves to that same
  reserved label. A successful `PUT` bumps the context's `config`
  revision and re-mints `cache_identity` under the entry write lock —
  durable write order is revision-then-content, so a crash between the
  two always fails toward extra cache invalidation, never a served
  mismatch — and is a no-op (nothing bumped, nothing re-minted) when
  the document is byte-identical to what is already installed. `GET`
  answers 404 `no_schema` for a context that never installed one,
  distinct from 404 `no_context` for a missing context (ADR 0009 §6.3's
  load-bearing distinction). `DirectoryEntry` gains a read-only,
  additive `schema_mode: Option<String>` field (`GET /contexts` and
  `GET /contexts/{name}`), echoing the installed document's `mode`
  (`null` before install). Auth: `GET` is `Read`, `PUT` is `Write` (an
  ingest-loop verb, not Admin) — both replica-refused/scoped like every
  other context route.
- The optional per-context schema document's on-disk file family (#379,
  S1 of #218's ADR 0009 split — foundation only: no enforcement, no
  `PUT`/`GET` route yet). A new standalone `{stem}.schema.json`
  (`SCHEMA_VERSION: u64 = 1`, `mode`/`closed_labels`/`types`/
  `relations`, `is_a` cycle/depth validation with an `MAX_TYPE_DEPTH =
  8` ancestor-closure precompute) follows `GroupRecord`'s
  write-then-rename pattern, with one deliberate divergence: a schema
  file that is unreadable, does not parse, does not validate, or whose
  digest disagrees with what `ContextMeta.schema_digest` recorded — in
  either direction, including a digest recorded with the file itself
  missing — refuses the boot outright, never a fresh-empty-record
  fallback, since an empty schema is indistinguishable from `mode: off`
  and would silently disable `strict` for a context whose operator
  explicitly turned it on. A schema-free context (no file, no recorded
  digest) boots byte-identical to every context before this change. Widens
  `context_files` from nine entries to ten (schema last, so a missing
  or lagging schema file never blocks a context rename) and adds
  `schema_formats: [1]` to `GET /version`/`version_facts()` beside
  `batch_formats`. **Downgrade note**: a data directory touched by this
  version may carry stray `{stem}.schema.json` files an older binary's
  nine-entry `context_files` will not delete on `DELETE
  /contexts/{name}` or move on rename — harmless litter on a writer
  (remove by hand before downgrading), automatically swept on a
  replica's next `hydrate_shared`.
- `compact --dry-run` and `compact --json`, `import --json`, and
  `inspect --json` (#371, split from #248's flag-consistency audit — the
  fuller mechanical rollout of `--dry-run`/`--parallel`/`--json` across
  every subcommand was reconsidered and NOT done; see the issue for what
  was left out and why). `compact --dry-run` reports each context's
  standing dead weight (`dead_edges`, `dead_ratio`, `arena_slack`,
  `dead_attributions`, `footprint_bytes`) without rewriting anything,
  offline and with `--url` alike — the `--url` case needed no server
  change, since `GET /contexts` already carries these per-context stats
  (`ContextStats`); a cold/remote row is marked `stats_are_snapshot` since
  its numbers are the last-saved snapshot, not a live recomputation.
  `compact --json` without `--dry-run` and `import --json`'s successful
  batches reuse existing HTTP response types — `MaintenanceCompactionEntry`
  and `ImportOutcome`/`ImportStreamOutcome`/`GroupImportOutcome` — rather
  than inventing new schemas, so the CLI's structured output and `POST
  /contexts/{name}/compact`/`POST /import`'s own bodies can't drift apart.
  `compact --dry-run --json` answers with a new `DeadWeight` shape instead
  (there is no HTTP endpoint that previews dead weight to reuse), and
  `import --json` adds a small CLI-only `error`/`failed_batches` envelope
  around the reused batch/group arrays so every `--json` exit path —
  including a refused batch, a validation failure, a registry that
  wouldn't boot, or a remote transport/refusal error — prints exactly one
  parseable JSON document, never silent stdout. `import --dry-run --json`
  offline is the one exception among successful runs: it never boots the
  registry (the read-only-without-a-lock property `--dry-run` already
  had), so it can't know `created`/`retracted` the way the server's own
  `?dry_run=true` (which does boot, via `preview_batch`) can — those
  fields report 0/false rather than a guess, and `groups` is always
  absent, documented in `--help`. `inspect --json` (`InspectReport`/
  `ContextRow`/`GroupRow`/`Notice`/`Totals`) is the only new schema built
  for a command with no HTTP counterpart to reuse at all (ADR 0002 §6
  rules out a remote `inspect`) — built alongside the existing
  human-readable report from the same computed values in the same scope,
  never a second pass over the data, so the two renderings cannot
  disagree about what was found.
- Cross-connector observability (#353, implementing ADR 0007 §11 and
  completing #217's observability requirement): every connector driver now
  shares one event/summary shape instead of `sync_object_storage` (#351)
  being the only one with any. New
  `taguru_langchain.ingest_connectors.observability`: `RunReport` (the
  seven-state `discovered`/`unchanged`/`parsed`/`extracted`/`imported`/
  `skipped`/`failed` tally — each source counted once, under its LAST
  phase only — plus `duration_ms`/`interrupted`, and
  `tags_dropped`/`deleted_detected`/`retracted` for connectors that use
  them), `SourceEvent` (one phase transition; `to_dict()`/
  `RunReport.events_jsonl()` render the same JSONL shape
  `taguru extract`'s own diagnostics sidecar uses), `RunRecorder` (the
  shared bookkeeping, including `attached()` — a scoped, chaining
  `TaguruIngester.on_event` hook that reports the `extracted` phase from
  `ImportStarted` without ever losing a caller's own already-installed
  callback), and `SourceEventSink` (an append-only per-source JSONL
  sidecar, `events_out=` on both drivers below, truncate-on-open like
  `DiagnosticsSink`, written even under `dry_run=True` since it is a dry
  run's entire product). New `taguru_langchain.ingest_connectors.
  references`: `sync_references`/`plan_references`/`default_connectors` —
  the local-file/`http(s)://` counterpart to `sync_object_storage`, for
  every non-S3 connector (`.md`/`.txt`/PDF/HTML/DOCX/PPTX). Classifies a
  reference's kind (path vs. URL) before ever asking a connector whether
  it "supports" the reference — every non-HTML connector's own
  `supports()` checks only the extension, so a naive dispatch would hand
  `https://example.com/a.md` to `open()` as a local path. Implements ADR
  0007 §11's per-kind `--dry-run` table: a local file is `unchanged` only
  when a new `FileProbeCheckpoint` (cheap `stat`-only metadata, mirroring
  `S3ObjectCheckpoint`) matches, `parsed` on any mismatch (never a false
  `unchanged`); a URL is always `parsed` — no `HEAD`, no network access at
  all under `dry_run`. A reference resolving to a source id already
  claimed earlier in the same run gets the new `duplicate_source`
  diagnostic and is never fetched — `RunRecorder.duplicate()` records it
  without disturbing the winning occurrence's own tally. An HTTP redirect
  is handled by `RunRecorder.retarget()`: `discovered` keeps the
  pre-redirect URL (the honest history), every later phase and the tally
  itself move to the post-redirect one, and the reference counts exactly
  once either way. New `FileProbeCheckpoint`/`FileProbe` in
  `taguru_langchain.ingest_connectors.checkpoint`.
- PPTX connector and an external OCR adapter boundary (#352, implementing
  ADR 0007 §10 and completing #217's Office/OCR requirements): `PptxConnector`
  in `taguru_langchain.ingest_connectors.pptx`, reading `.pptx` files — `pip
  install "langchain-taguru[pptx]"` for its `python-pptx` dependency, kept
  optional per ADR 0007 §3/§4's packaging decision, the same one
  `DocxConnector`'s `python-docx` already follows. A slide's shapes are
  walked in document order, recursing into a group shape's own nested
  shapes; every non-empty text-frame paragraph and every table (rows joined
  with `\n`, cells with `" | "`, one paragraph per table) carries a
  `{"kind": "slide", "value": ...}` locator. Unlike `DocxConnector` (whose
  one-locator-per-paragraph budget goes to tables, since a DOCX has no
  page-like structure to spend it on instead), a PPTX slide already has a
  number, so this connector spends that budget distinguishing a slide's
  body from its speaker notes instead: notes are read as their own
  paragraph(s), each carrying `{"kind": "speaker_notes", "value": ...}`
  rather than being folded into the paragraph next to it (ADR 0007 §7.3). A
  slide's title is read like any other paragraph (same `slide` locator as
  its neighbors) and additionally becomes the paragraph-anchored `section`.
  No OCR engine ships: a presentation left with no extractable text (an
  image-only deck) is `ocr_required` with empty `text`. A chart, a SmartArt
  diagram, and an embedded/linked OLE object are each unreachable through
  this connector's own shape walk — named in a single `partial_extraction`
  diagnostic rather than silently short-changed. Only `.pptx` is read;
  `.ppt` (legacy binary) and `.pptm` (macro-enabled) are both
  `unsupported_format`. Also new,
  `taguru_langchain.ingest_connectors.ocr`'s `OcrAdapter` (ADR 0007 §10): the
  external OCR engine boundary a connector calls out to when one is
  configured — no OCR engine ships in any connector or this package. Given
  the raw document bytes and the locators naming which pages/units are
  unusable, an adapter returns whatever text it recovered for them, each
  still tagged with the locator it was asked about (an adapter is itself
  just another producer of ADR 0007 §5's normalized-document contract, for
  the pages it recovers). `PdfConnector` gains a new `ocr_adapter=`
  parameter (issue #348's own connector, wired here): it offers a
  configured adapter exactly the pages its own `min_chars_per_page`
  threshold found unusable, never the whole document, and splices back only
  a recovered unit that both names a page it actually asked about and
  clears that same threshold — an adapter's own exception, an unrequested
  page, or text too thin to count, all degrade to the unconfigured-adapter
  behavior for the page(s) involved, never a hard failure. Configuring,
  swapping, or removing an adapter changes `PdfConnector`'s own
  `parse_options_digest`, so a §6.3 connector-level checkpoint's prior skip
  decision is correctly invalidated. No change to `src/`, `http_contract`,
  or `mcp_contract` — this connector, like #348-#351's, is entirely
  client-side per ADR 0007 §3/§4's packaging decision.
- S3 (object-storage) connector (#351, implementing ADR 0007 §9):
  `S3Connector`/`sync_object_storage` in
  `taguru_langchain.ingest_connectors.s3`, plus the
  `taguru_langchain.ingest_connectors.objectstore` store boundary
  (`S3ObjectStore`, optional `boto3` dependency via the new `s3` extra —
  `pip install "langchain-taguru[s3]"`, the same packaging decision
  `PdfConnector`/`DocxConnector` already follow — and `FileObjectStore`,
  stdlib-only, the `file://` backend this package's own tests and an
  air-gapped deployment both use in place of a live bucket). Each listed
  object is dispatched by extension (falling back to content-type) to
  whichever installed connector above handles it, then re-stamped onto the
  S3 source id (`s3://bucket/key`) — the delegate's own
  `fingerprint_inputs` (parser/version/raw content hash/options) is kept
  untouched. A two-layer checkpoint composes with (never replaces)
  `CheckpointStore`: a cheap listing-metadata check (`version id` >
  `content hash` > `(size, last-modified)` > bare `ETag` as the last
  resort — an `ETag` alone is never the first choice, since some
  S3-compatible stores compute it in a way that isn't a reliable content
  hash) skips the fetch entirely when nothing changed; a second check
  skips the model call/import when the fetched-and-parsed content is
  unchanged despite a metadata bump. Credentials always come from
  `boto3`'s own standard chain — no parameter anywhere in this path
  accepts an access key/secret directly, and none ever reaches a
  checkpoint, batch file, log line, or Taguru source metadata. Object
  tags map onto `metadata.tags`, capped and drop-counted like every other
  per-source limit, never silently truncated. Deletion is `report`-only
  by default (a vanished object is only counted, never retracted);
  `deletion_policy="retract"`/`"mirror"` are both explicit opt-in, backed
  by a persisted prefix inventory that degrades to "nothing to compare
  against" — never a false deletion — on its own absence or corruption.
  `dry_run=True` reports discovered/would-fetch/unchanged without any
  network access, parse, ingest, or checkpoint/inventory write. No change
  to `src/`, `http_contract`, or `mcp_contract` — this connector, like
  #348-#350's, is entirely client-side per ADR 0007 §3/§4's packaging
  decision.
- DOCX connector (#350, implementing ADR 0007 §7/§8): `DocxConnector` in
  `taguru_langchain.ingest_connectors.docx`, reading `.docx` files — `pip
  install "langchain-taguru[docx]"` for its `python-docx` dependency, kept
  optional per ADR 0007 §3/§4's packaging decision, the same one
  `PdfConnector`'s `pypdf` already follows. The document body is walked in
  real document order (paragraphs and tables interleaved via python-docx's
  own `iter_inner_content()`, never `document.paragraphs`/`.tables`
  separately); a heading is recognized by its `Heading N`-named style or,
  falling back, its own `w:outlineLvl`, and its breadcrumb becomes a
  `section` (`"Guide > Installation"`, the same convention `HtmlConnector`
  already uses). A table — top-level or nested inside another table's own
  cell — becomes exactly one paragraph (rows joined with `\n`, cells with
  `" | "`) carrying a `{"kind": "table", "value": ...}` locator (`"3"`, or
  `"3.1"` for a table nested inside table 3's own cell); an ordinary body
  paragraph never carries a locator, so "this paragraph has a locator"
  means "this paragraph is a table" for any citation reading this
  connector's `locators` — the opposite trade-off from `HtmlConnector`'s own
  `fragment` locator, since a DOCX has no page/anchor of its own to spend
  ADR 0007 §7.2's one-locator-per-paragraph budget on instead. No OCR
  engine ships (ADR 0007 §10): a document left with no extractable text (an
  image-only `.docx`) is `ocr_required` with empty `text`. A
  password-protected `.docx` is recognized by its own MS-OFFCRYPTO
  OLE2/CFB container signature and reported `encrypted` before ever being
  opened as a zip — never misreported as `corrupt` — while a
  restricted-editing password (`w:documentProtection`, a weaker,
  non-encrypting mechanism) is correctly left alone. Footnote/endnote/
  comment text and text-box content are each unreachable through this
  connector's own paragraph walk (`python-docx` has no read API for the
  first three, and text-box content lives in a nested `w:txbxContent` a
  paragraph's own run text never descends into) — named in a single
  `partial_extraction` diagnostic rather than silently short-changed. Only
  `.docx` is read; `.doc` (legacy binary) and `.docm` (macro-enabled) are
  both `unsupported_format`. No change to `src/`, `http_contract`, or
  `mcp_contract`.
- HTML connector (#349, implementing ADR 0007 §7/§8): `HtmlConnector` in
  `taguru_langchain.ingest_connectors.html`, reading both a local
  `.html`/`.htm`/`.xhtml` file and an `http(s)://` URL fetch (via `httpx`,
  now a direct dependency of `langchain-taguru` — already transitive
  through `taguru`) — no new optional extra, parsing is stdlib
  `html.parser` only. Boilerplate (script/style/nav/aside/hidden regions,
  and a page's own header/footer when no `<main>`/`<article>` scopes the
  content) is stripped before `text` is built; the heading hierarchy
  survives as a breadcrumb `section` per paragraph (`"Guide >
  Installation"`, since `sections` is flat); and each heading's own `id`
  (or its nearest `id`-bearing ancestor's) becomes a `{"kind": "fragment",
  "value": ...}` locator on every paragraph up to the next heading —
  combined with `metadata.canonical_url`, a citation can point at a real
  in-page deep link. A URL fetch's source id is the *final*,
  fragment-stripped, canonicalized URL (ADR 0007 §6.1); a page's own
  `<link rel="canonical">`, when present, only ever populates
  `metadata.canonical_url`, never substituted for the source id itself,
  since a page can claim any canonical and two distinct pages claiming the
  same one would otherwise collide. A page left with no extractable text
  after boilerplate removal (image-only, an unrendered JS-shell SPA) is
  reported `ocr_required` with empty `text`; a 4xx/5xx response, a
  non-HTML `Content-Type`, a raw body over `max_file_bytes` (streamed,
  refused mid-fetch), a per-phase `timeout`, or the total fetch exceeding
  the separate `max_total_seconds` wall-clock budget are each their own
  diagnostic (`unreadable`/`unsupported_format`/`content_too_large`) —
  never a raised exception, including for a pathologically deep tree that
  would otherwise raise `RecursionError`. By default, a URL fetch also
  refuses any destination (including one reached only via a redirect)
  that resolves to a private, loopback, link-local, or multicast address,
  so an otherwise-trusted URL cannot be turned into a probe of
  `localhost` or a cloud metadata endpoint by a redirect the origin
  server controls; `allow_private_networks=True` disables this for a
  caller that intentionally targets one. An `<iframe>`/`<frame>` whose
  content was not fetched is named in a `partial_extraction` diagnostic
  rather than silently dropped. A Windows drive-letter path
  (`C:\docs\a.html`) is recognized as a local path rather than
  misclassified by its `urlsplit` scheme. Nested `<table>`s and a
  `{"kind": "table"}` locator are out of scope for this connector (ADR
  0007 §7.3's table locator is left to a future revision). No change to
  `src/`, `http_contract`, or `mcp_contract`.
- PDF connector (#348, implementing ADR 0007 §7/§8/§10): `PdfConnector`
  in `taguru_langchain.ingest_connectors.pdf`, the first standard format
  connector built on #347's protocol — `pip install
  "langchain-taguru[pdf]"` for its `pypdf` dependency, kept optional per
  ADR 0007 §3/§4's packaging decision. Emits one `{"kind": "page",
  "value": N}` locator per paragraph, derived from the PDF's own page
  boundaries (never from `taguru extract`'s chunking, ADR 0007 §7.4); its
  outline (bookmarks), when present, becomes `sections` and the document
  `title`. No OCR engine ships (ADR 0007 §10): a page whose extracted
  text falls under a connector-documented, configurable per-page
  character threshold (`min_chars_per_page`, default 16, after
  whitespace normalization) is named in an `ocr_required` diagnostic with
  empty `text` for that page's contribution — never silently passed
  through as low-quality text — leaving the external OCR adapter
  boundary to #352. Encrypted PDFs requiring a user password this
  connector does not have (a PDF unlockable with an empty user
  password — owner-restrictions-only — extracts normally instead) and
  structurally corrupt PDFs are reported as `encrypted`/`corrupt`, a
  single page's own decode failure
  as `partial_extraction` without failing the rest of the document, and
  a raw file over `max_file_bytes` (default 64 MiB) or extracted text
  over `MAX_PASSAGE_BYTES` (8 MiB) as `content_too_large` — never a
  raised exception. No change to `src/`, `http_contract`, or
  `mcp_contract`.
- Connector protocol and normalized document contract (#347, implementing
  ADR 0007 §5/§6/§8): a `taguru_langchain.ingest_connectors` submodule
  (no new Rust dependency, no new binary — `langchain-taguru` reuses
  `TaguruIngester`/`CheckpointStore` directly, per ADR 0007 §3/§4's
  packaging decision) defining `ConnectorDocument` — the one shape every
  standard connector (PDF/HTML/DOCX/S3, tracked as #348-#352) produces:
  paragraph-joined `text`, paragraph-indexed `sections`/`locators`,
  `metadata`, `fingerprint_inputs`, and a closed `diagnostics` vocabulary
  (`unreadable`, `unsupported_format`, `encrypted`, `corrupt`,
  `ocr_required`, `source_id_too_long`, `content_too_large`,
  `partial_extraction`) — a non-empty `diagnostics` with an empty `text`
  is the required encoding of "nothing usable was extracted," never a
  silently empty passage. Source id derivation follows ADR 0007 §6.1
  (`path`, `path#fragment`, mandatory URL canonicalization stripping
  userinfo and signed-query credentials); a connector's own fetch/parse
  work gets an independently resumable `ConnectorCheckpoint` (§6.3),
  layered over `CheckpointStore` under its own key namespace so it never
  collides with `TaguruIngester`'s own chunk checkpoint. `.md`/`.txt`
  ship as the reference connector (`TextFileConnector`, extracting ATX
  headings as sections), wired end to end via `ingest_connector_document(s)`
  into `TaguruIngester.ingest_text`'s new `sections=`/`locators=`
  parameters (`aingest_connector_document(s)` into `aingest_text`'s) —
  which, in turn, `IngestOutcome.sections_stored`/`sections_dropped`/
  `locators_stored`/`locators_dropped` now report. ADR 0007 §6.2's
  `locator_digest` addition to `taguru extract`'s own manifest fingerprint
  is deliberately not part of this issue: the staleness it guards against
  is specific to `taguru extract`'s Rust-side manifest skip, which
  `TaguruIngester` never takes (every ingest re-renders and re-imports
  its batch). No change to `src/`, `http_contract`, or `mcp_contract`.
- Typed citation locators (#346, implementing ADR 0007 §7): a new,
  independent, paragraph-indexed `locator: {kind, value}` — a page,
  slide, sheet, table, or other position — alongside the existing
  free-text `section` heading label. Unlike `section`, a locator does
  not extend to the next paragraph. Landed as a new batch line
  (`{"paragraph": N, "locator": {"kind": "page", "value": "12"}}`),
  a new `locators` map on `POST /contexts/{name}/sources`, and a new
  optional field on `Citation` and every `attributions[]` entry
  (`recall`/`query`/`explore`/`activate`/`unreachable_from`/
  `/retrieve`'s citations) — never omitted, `null` when absent, per
  ADR 0005 §4's compatible-optional-field-addition classification.
  `taguru export`/`taguru import` round-trip it losslessly. Python/
  TypeScript SDKs gain the `Locator` type and `store_passages`/
  `storePassages`'s new `locators` option. `http_contract`/
  `mcp_contract` stay `1`.
- OpenTelemetry tracing for the whole composed retrieval loop (#224,
  implementing ADR 0008): where previously only three spans existed
  server-wide (`request`/`embed`/`rerank`), `taguru.retrieve` and
  `taguru.assemble_evidence` now each export a full span tree — a phase
  span per step (resolve/describe/query/activate/citations/passage
  fallback), and passage search nested further into
  `taguru.search.bm25`/`.ann`/`.fuse` lane spans — so a slow or
  degraded `POST /mcp` can be diagnosed from one trace instead of
  guessed at from a flat request span. Skip/degrade/cache decisions are
  recorded as `taguru.skip`/`taguru.degrade`/`taguru.cache` events
  carrying a stable `taguru.reason` code, never a raw string; attribute
  values are bound to existing `/metrics` enum spellings so no parallel
  vocabulary can drift out of sync. Context propagation is now
  bidirectional everywhere it previously only extracted: router →
  shard fan-out (`taguru.shard_call`) and the stdio bridge
  (`taguru-mcp`, via optional `params._meta.traceparent`) both inject
  their own current span outbound rather than passing inbound headers
  through bare, and `tracestate` now forwards through the router
  (previously dropped). Two real defects this closed along the way:
  an embedding-provider degrade no longer paints the whole request
  span `ERROR` for what is actually a successful, merely-degraded
  retrieval (`.with_error_events_to_status(false)`, plus no span-event
  field is ever named `error` going forward); and
  `TAGURU_LOG_SEARCHES=1` combined with OTLP export can no longer leak
  a raw question into a span event (a target-level export firewall
  forces `taguru::search` `OFF` on the OTel layer regardless of level,
  verified by a sentinel integration test). Fully opt-in
  (`OTEL_EXPORTER_OTLP_ENDPOINT`, as before); disabled, every new span
  call short-circuits to `Span::none()` without opening the tracing
  registry's storage. `http_contract`/`mcp_contract` are unaffected —
  spans and events carry no wire-visible behavior change. See
  [Tracing](https://t0k0sh1.github.io/taguru/tracing.html).
- Python/TypeScript SDK tracing parity (#224): `retrieve()` composes
  the identical client-side span tree the server's own
  `taguru.mcp.retrieve` does, sharing one reason-code vocabulary
  pinned in `sdk/spec/tracing.yaml` and asserted by both languages'
  test suites. Both SDKs treat OpenTelemetry as strictly optional and
  add no required dependency: Python via the new `taguru[otel]` extra
  (soft `opentelemetry-api` import; `taguru._tracing` no-ops without
  it), TypeScript via an optional peer dependency loaded through a
  lazily cached dynamic `import()` and excluded from the bundle
  (`tsup --external @opentelemetry/api`). Neither SDK opens a
  client-side HTTP span for the request itself, and neither ever calls
  `set_tracer_provider`/`setGlobalTracerProvider` — that stays the
  host application's call.

### Changed
- **Breaking (SDKs):** `add_associations`/`addAssociations` returns
  `AddAssociationsResult {applied, issues, schema_violations}` instead
  of the bare applied count, and `BatchApplyResult`/`ImportResult` gain
  `issues`/`schema_violations` fields (`ImportResult` also `schemas`,
  one `SchemaImportOutcome` per `taguru_schema` record the stream
  restored). Both SDKs previously unwrapped only the envelope's
  `result`, which made ADR 0009 §8.3's `warn`-mode carrier — the
  `issues`/`schema_violations` fields riding *beside* `result` on a
  write whose associations violated the schema — unreachable through
  the SDK by any means: a `strict` refusal's issues survive in the
  error body, but flipping a context to `warn` silently hid the same
  violations from SDK callers, the exact asymmetry §8.3's "identical
  `Issue` values in both modes" contract exists to prevent. Migration:
  `applied = ctx.add_associations(ops)` becomes
  `ctx.add_associations(ops).applied`; a caller that ignored the
  return value is unaffected, and every new field is empty/zero for
  `off` mode, no schema, a conforming write, or a server predating the
  fields.
- `ManifestEntry`/`CheckpointFingerprint` (Rust) and their SDK
  checkpoint-fingerprint twins gain a `schema_digest` field (#386, S8 of
  #218's ADR 0009 split §11), so that swapping in a different schema
  document for `taguru extract`/both LangChain ingesters re-extracts
  even when the source text is byte-identical. On-disk format change:
  an entry from before this field existed defaults to `""` and still
  matches a schema-less rerun, the same precedent `structured_output`/
  `lossy` already set.
- `sync_object_storage` (#351) is rewritten onto the shared `RunRecorder`
  (#353) and gains an `events_out=` parameter: a path (truncated on open,
  like `taguru extract`'s own `--diagnostics-out`) or an already-open text
  stream to also stream the per-source phase trail as JSONL — one UTF-8
  JSON object per line, written the moment each phase transition happens,
  never appended to across separate runs. Three observable behavior
  changes: (1) a `SourceEvent`'s `bytes` on the `parsed` phase is now the
  object's own raw byte size (matching every other connector driver's
  convention — `SourceEvent.bytes` is never the parsed text's byte count,
  which would make a sum over that column meaningless for phases that
  carry no text at all), previously `len(document.text.encode())`; (2) a
  duplicate key in one `store.list()` listing (a store bug/quirk — the
  same key appearing twice in one pass) is now recorded with the
  `duplicate_source` diagnostic instead of a silent, uncounted `continue`;
  (3) the full listing is now always drained and every object reported
  `discovered` before any object is fetched (ADR 0007 §11's "enumerate all
  discovered work before the first fetch"), rather than interleaving
  listing and fetch one object at a time — a `should_stop` interruption
  therefore no longer leaves the listing itself incomplete, and (a useful
  side effect) the run's own prefix inventory stays trustworthy and gets
  saved even after an interrupted pass, since `seen_sources` is always a
  complete snapshot regardless of where fetch/import got interrupted.
- `S3SyncReport` (`sync_object_storage`'s own report type, published by
  #351/#352) is now a deprecated alias of the new `RunReport` (#353)
  rather than its own dataclass — a slotted, frozen dataclass subclass
  re-declares its base's `__slots__` on this SDK's Python 3.10 floor (the
  `inherited_slots` fix landed in 3.11), so an alias is the only option
  correct on the whole supported range. Consumer-visible: code that
  subclassed `S3SyncReport`, checked `type(report) is S3SyncReport`, or
  relied on it carrying only S3's own fields now sees `RunReport`'s wider
  field set (`connector`, `duration_ms`, `interrupted`, `events`,
  `events_path`); `isinstance`/attribute access and every field #351/#352
  already published are unaffected.

- The no-poisoning lock policy (Cargo.toml's parking_lot rationale)
  now covers its last three server-state holdouts: `ship.rs`'s
  shipped-seq tracking, `replica.rs`'s fence-holder reporting, and
  `passages.rs`'s whole `PassageStore` move from `std::sync` to
  `parking_lot`, so a panic mid-critical-section can no longer poison
  a lock and brick the shipper's progress, the replica's
  write-refusal text, or a context's passage reads for the rest of
  the process. No wire or on-disk change.

### Fixed
- Documentation drift, in the live protocol manual first: the document
  `GET /protocol` and every MCP `initialize.instructions` actually
  serve (`src/llm-protocol.md`) had no route-table rows for
  `GET/PUT /contexts/{name}/schema`,
  `POST /contexts/{name}/schema/audit`,
  `POST /contexts/{name}/schema/validate`, or the pre-existing
  `POST /contexts/{name}/drift/audit`, no `schema_mode` in
  `GET /contexts`' documented row shape, and no `no_schema` in the
  stable error-`code` vocabulary — all three already shipped and
  fixture-pinned. All added; `docs/schema.html` now also names the
  directory row's `schema_mode` and the SDK schema methods, and its
  §6.3 exclusion list includes the coverage audit. Env-var docs catch
  up too: `docs/getting-started.html`'s table gains
  `TAGURU_PASSAGES_WAL_MAX_BYTES`, `TAGURU_AUTH_FAIL_LIMIT_PER_MIN`,
  `TAGURU_CROSS_SEARCH_CONCURRENCY`, and `TAGURU_EMBED_PARALLEL`, and
  README's MCP section documents `TAGURU_MCP_MAX_CONCURRENT_TOOLS`
  and `TAGURU_MCP_MAX_RESULT_BYTES` — previously in `taguru --help`
  and `KNOWN_KEYS` only. One stale mirror comment
  (`sdk/python-langchain/.../_extract.py`, "PROMPT_VERSION 2" over a
  `PROMPT_VERSION = 3` constant) now matches its TypeScript twin.
- `TAGURU_EXTRACT_SCHEMA`, documented in `taguru extract --help` and
  read since it shipped, was missing from `config.rs`'s `KNOWN_KEYS`
  typo lint, so an `extract --config` file setting it earned a false
  "is not a variable taguru reads (typo?)" warning while the value was
  applied anyway. Added — and because the gap survived the existing
  cli.rs consistency tests (which only see cli.rs's own usage text),
  the check is now a shared helper
  (`config::assert_usage_vars_are_known_keys`) run against every
  command's USAGE — cli, extract, compact, estimate, export, evaluate,
  ingest — with extract also gaining the reverse test that every
  `TAGURU_EXTRACT_*` key is documented in its own `--help`.
- More protocol-manual drift, the same class as the schema rows above:
  `POST /contexts/{name}/rename` and `POST /groups/{name}/rename`
  (registered, MCP-exposed, auth-special-cased — but never in the
  route table), `POST /mcp` itself (the Streamable HTTP transport the
  manual never named), and `POST /flush` (previously only name-dropped
  in the admin-role list) are now documented, and a new main.rs test
  scans the router's registrations — method and path both — against
  `llm-protocol.md` so a route can no longer ship undocumented, nor a
  documented method drift from the registered one.

## [0.6.0] - 2026-08-01

### Added
- `POST /contexts/{name}/evidence` (#305, MCP: `assemble_evidence`,
  implementing ADR 0006 §5/§10/§11/§13.1-13.2): opt-in evidence
  assembly over the composed resolve/query/activate/search_passages/
  cite_passage fan-out `retrieve` already runs, plus an opt-in
  community-summary search — normalizes every graph association, graph
  activation, passage hit, and community hit into one ranked
  (reciprocal-rank fusion over each lane's own rank, never comparing
  raw BM25/cosine/graph-weight/community scores against each other),
  deduplicated, citation-complete package under three independent hard
  budgets (item count, byte length, estimated tokens). Contradictory
  and corroborating evidence are both preserved intentionally — a
  disagreement is admitted or omitted as one atomic group, never
  split, and a fact several sources assert keeps every source named,
  never collapsed to a count. A budget too small for even the smallest
  candidate answers `200` with an empty package and every candidate
  accounted for under `omitted`/`omitted_total`/`omitted_by_reason`,
  never an error; `include_communities: true` without a derived
  artifact degrades the same way (`plan.lanes.communities.ran: false`)
  rather than refusing, since community evidence is one opt-in input
  here, not the whole point of the call the way `communities/search`
  is. `plan.reranker` reports whether an optional reranker (#307) is
  configured and ran. `retrieve` and every direct endpoint this
  feature composes are unchanged; `http_contract`/`mcp_contract` stay
  `1` (a purely additive endpoint and MCP tool, per ADR 0005 §4).
- Python/TypeScript SDK parity for evidence assembly (#306, implementing
  ADR 0006 §5.3/§10): `Context.assemble_evidence()` (Python, both sync
  and async) / `Context.assembleEvidence()` (TypeScript) call `POST
  /contexts/{name}/evidence` with the same typed request options MCP's
  `assemble_evidence` tool already exposes (`budget`, `rerank`,
  `include_communities`, …) and decode into a typed `EvidencePackage` —
  `items`/`citations`/`budget`/`omitted`/`omitted_total`/
  `omitted_by_reason`/`plan`, embedding the existing `Association`/
  `PassageHit`/`CommunityHit`/`Citation`/`LanePlan` models verbatim
  rather than minting parallel evidence-only types. `kind`, every
  `lane`, `omitted[].reason`, and `plan.reranker.reason` decode as open
  strings (Python: plain `str`; TypeScript: `Open<T>`), never a closed
  enum, so a future server-added value never breaks either SDK. Both
  SDKs' golden wire-contract tests now decode the five `evidence_*`
  fixtures (#301) through the real decoder/`unwrapEnvelope`, not just
  the structural enum check. `sdk/spec/surface.yaml` gains a matching
  `assemble_evidence` entry, CI-enforced identically to every other
  method. MCP itself needed no change — #305 already shipped
  `assemble_evidence` on both the stdio bridge and remote MCP from one
  shared tool schema.
- Optional evidence reranker (#307, implementing ADR 0006 §12): `POST
  /contexts/{name}/evidence` accepts `rerank?: {model?}` and, when a
  provider is configured (`TAGURU_RERANK_URL`/`_MODEL`/`_API_KEY`/
  `_TIMEOUT_SECS`, a Cohere/Jina-compatible `POST /rerank` endpoint —
  the same "any compatible adapter plugs in" posture the embedding
  tier already takes), reorders the already fused, deduplicated,
  near-duplicate-suppressed candidate pool immediately before
  diversity-aware admission — strictly a permutation; a reranker can
  never add, drop, or edit a candidate, and every selection invariant
  holds identically whether or not one ran. No credential or network
  access is required by default: absent `rerank`, or with no provider
  configured, selection stays exactly as deterministic as before this
  release. Any failure — unreachable, timeout, an open circuit
  breaker (mirroring the embedding tier's own, `taguru_rerank_breaker_*`
  on `/metrics`), a non-2xx status, or a response that is not a
  complete permutation — degrades to the same deterministic order
  rather than ever answering a non-2xx `POST /contexts/{name}/evidence`
  call; `plan.reranker.reason` names why in a fixed, machine-readable
  vocabulary (`not_configured`/`model_mismatch`/`empty_pool`/
  `invalid_permutation`/`circuit_open`/`timeout`/`provider_error`).
  Candidate text reaches a configured provider and nowhere else — never
  a log line, an error message, or a metric label; `taguru_rerank_outcomes_total`/
  `taguru_rerank_duration_seconds` carry only outcome tokens and
  timings. `http_contract`/`mcp_contract` stay `1` — a purely additive
  request field and response fragment (ADR 0005 §4).
- `taguru evaluate --assembly`/`--max-items`/`--max-bytes`/`--max-tokens`/
  `--rerank` (#308, implementing ADR 0006 §14): proves evidence assembly
  helps at equal budget instead of on a subjective demo, reusing #215's
  own `eval.jsonl`/`taguru evaluate` harness rather than a second
  evaluation path. `--assembly` swaps the passage lane for `POST
  /contexts/{name}/evidence`; the structural lane (`resolve` →
  `query`) never changes, so a `baseline`/`assembly` run pair stays
  comparable on coverage and lane cross-tab. `--max-items`/
  `--max-bytes`/`--max-tokens` apply the identical three ceilings to
  *both* modes — `--assembly` via the request's own `budget`, and
  `baseline` truncated client-side with the exact accounting
  `crate::api::evidence::budget` computes server-side, never a
  reimplemented approximation. Without any budget flag, `baseline`
  behaves exactly as it did before this release — no truncation, no
  `budget` block in `evaluation.json`. New metrics:
  `diversity.sources` (distinct source locators among a case's
  admitted evidence — the one metric #216 names explicitly, "source
  diversity at equal evidence budget"), `budget.items_used`/
  `.bytes_used`/`.tokens_used`/`.omitted_rate`, `latency.evidence_ms`,
  and `rerank.ran` (the configured-reranker success rate; its
  complement is the degrade rate). `diversity.sources` joins the nine
  existing case-scoped/comparison-eligible metrics as the tenth;
  `evaluate compare` now also warns (never refuses) when two runs'
  budgets differ, including one side having no budget flag at all.
  Regression thresholds for the new metrics use ADR 0004 §9.3's
  existing thresholds-file format — no new format. The default
  repository gate stays offline, deterministic, and provider-free — a
  configured reranker is opt-in exactly like ADR 0004's own
  embedding-provider suites, never required to pass. `evaluation.json`'s
  schema grows additively (`taguru_evaluation` stays `1`): `inputs.mode`/
  `.budget`/`.rerank`, and per-case `evidence`/`budget`/
  `diversity_sources`.
- `GET /version` (#300, implementing ADR 0005 §3/§6): contract-version
  discovery, auth-exempt like the other probes and answering `200`
  even while `/health` reports degraded — a compatibility check has to
  run from something the fault it might be diagnosing doesn't itself
  affect. Bare JSON naming every version dimension ADR 0005 §3
  defines: `server`, `http_contract`/`mcp_contract`
  (`{current, supported}`, both starting at `1`), `mcp_protocol`,
  `batch_formats`, `image_formats`, `communities_formats`. Router mode
  answers its own `GET /version` the same way, under the same
  "shards are homogeneous" assumption `/health`/`/protocol` already
  proxy under. The same facts are folded into `GET /protocol`'s
  `## This server` trailer and therefore into every MCP
  `initialize`'s `instructions`, so an MCP client learns them without
  a second connection.
- Both official SDKs (Python, TypeScript) declare a supported
  `http_contract` range (`taguru.SUPPORTED_HTTP_CONTRACTS` /
  `SUPPORTED_HTTP_CONTRACTS`) and run a one-time compatibility
  preflight against `GET /version` before their first real request —
  shared across every concurrent caller on a fresh client, so a
  `gather()`/`Promise.all()` of several calls doesn't let all but the
  first race past an unfinished check. Fails closed only on positive
  proof the two ranges share no version in common, raising a new
  `IncompatibleServerError` (both SDKs) that names which side to
  upgrade and the exact `pip`/`npm` command; fails open on every
  absence of information — a 404 (any server predating this endpoint,
  treated as `http_contract: 1`), a non-JSON body, a missing or
  malformed `http_contract` key — and never compares `server`'s own
  SemVer or `mcp_contract` (neither SDK speaks MCP). `wait_until_ready`
  (Python)/`waitUntilReady` (TypeScript) now surface a confirmed
  incompatibility immediately instead of stalling out the full
  timeout.
- Python's `ResponseShapeError` (ADR 0005 §9.3): the two `_decode.py`
  failure modes — a response's container shape not matching (the
  literal 0.4.0 `PassagePage` incident) and a required field missing —
  now raise this dedicated `TaguruError` subclass (also still a
  `ValueError`, purely additively) instead of a bare `ValueError`
  outside the catchable hierarchy.
- TypeScript's runtime `VERSION` export (`sdk/typescript/src/version.ts`,
  ADR 0005 §9.2): no runtime version constant existed in `src/`
  before this, so nothing local existed to compare `GET /version`
  against. Locked to the server's own version by
  `sdk/spec/check_versions.py`, the same way Python's
  `taguru.__version__` already is.
- Golden wire-contract fixtures and a breaking-change CI guard (#301,
  ADR 0005 §9): `tests/fixtures/wire/` pins the current `http_contract:
  1`/`mcp_contract: 1` shapes — thirteen representative HTTP/MCP
  operations, including five #216 evidence-assembly cases (mixed
  lanes, budget-constrained, duplicate-passage suppression, a
  contradiction group, and the communities/rerank degrade) — generated
  from a live server (`tests/http_api/contract.rs`,
  `TAGURU_UPDATE_WIRE_FIXTURES=1 cargo test --test http_api contract`)
  and read identically by Python
  (`sdk/python/tests/unit/test_wire_contract.py`) and TypeScript
  (`sdk/typescript/tests/unit/wire-contract.test.ts`). New
  `sdk/spec/check_contract.py` diffs the committed fixtures against a
  base ref and fails a PR that ships a field removal, a container-shape
  change (array ↔ object), a known enum value disappearing, a newly
  required request field, or a removed operation without a matching
  `HTTP_CONTRACT`/`MCP_CONTRACT` bump in `src/api.rs` — the mechanical
  half of ADR 0005 §4's compatible/breaking table, run in CI's new
  `contract-guard` job. `tests/fixtures/wire/README.md` documents the
  update procedure, including the contract-version judgment call ADR
  0005 §4/§7 already require.

### Changed
- The pre-1.0 compatibility guarantee (`src/llm-protocol.md`
  `## Compatibility`, ADR 0005 §7): **within one `http_contract` (or
  `mcp_contract`) version, nothing breaks**, effective immediately —
  tighter than the previous "shapes may also change between minor
  versions." A `server` minor bump may still add things; a break now
  requires the matching contract-version bump, landing in the same PR
  as this file's own entry and a migration note.
- Nine of TypeScript's closed enum-like string-literal unions in
  `models.ts` widened to accept an unrecognized value (ADR 0005 §5/
  §9.1) — `TieredResolution.tier`/`.kind`, `NearestResolution.kind`,
  `LexicalExplain.kind`, `ResolveRanking.tier`,
  `ResolveExplanation.verdict`/`.expected_kind`,
  `SearchExplanation.verdict`, `GroupImportOutcome.outcome` — via a
  new `Open<T>` helper type, matching Python's already-open plain
  `str` fields. `AliasEntry.namespace` is deliberately left closed: it
  is synthesized client-side (`iterAliases`), never decoded from the
  wire, so a server can never surprise this SDK with a value it didn't
  itself mint.

## [0.5.0] - 2026-07-30

### Added
- New top-level verb `taguru evaluate` (#273, implementing ADR 0004
  §5/§7/§9.1/§11/§12), the execution harness and `evaluation.json`
  skeleton for #215's retrieval-and-citation quality gate over one
  already-populated context — driven entirely over HTTP, like `taguru
  benchmark search`, and calling no answer-generation LLM anywhere on
  this path (asserted by a source-level test, not merely claimed). Per
  case, two independent lanes run with no fusion: the passage lane
  always calls `POST /contexts/{name}/sources/search`; the structural
  lane (only when a case declares `expected_concepts`/
  `expected_labels`/`expected_associations`) resolves coverage cues via
  `/resolve`/`/resolve_label`, then pins each `expected_associations[]`
  entry via `/query` — but only when every position resolves to
  exactly one candidate; zero is `not_found`, two or more is
  `ambiguous`, and neither ever guesses at or fans out over a
  combination. A third, citation lane (#275, ADR 0004 §8) runs whenever
  a case declares `expected_citations[]`, independent of whether the
  passage lane found anything for that case: one
  `POST /contexts/{name}/citations` call per entry (never batched),
  checking that it resolves (`no_source`/`no_paragraph` recorded
  separately), that `section` matches when the eval case declares the
  key (an explicit `null` included), and that `quote`, when declared, is
  a `normalize_entry`-folded substring of the returned text — never a
  match across a paragraph boundary, since `Citation.text` is exactly
  one paragraph. `recall`/`activate`/`explore`/`describe` are
  deliberately never called. Preflights an unreachable server and any
  `expected_sources[]` entry the context does not carry before any case
  runs — `expected_citations[]` is deliberately NOT preflighted, since a
  citation naming a source or paragraph the corpus lacks is exactly the
  failure the citation lane exists to detect. `evaluation.json` records
  lane/plan echoes and per-hit locators (never corpus body text),
  recall@k/MRR/nDCG and concept/label/association coverage (#274/#292),
  citation recall and locator validity as two measurements that are
  never merged into one score (#275; a `quote` mismatch records only the
  user's own declared quote and a match boolean, never the served text),
  latency distributions, a corpus-revision bracket
  (`revision_before`/`revision_after`/`stable`, plus `last_write_epoch`)
  that detects a write landing mid-run, and the resolved (masked —
  scheme/host/port only, never userinfo or a bearer) target. Runs to
  completion on a read-only API key. This build has no `--thresholds`:
  every run is report-only and exits 0 (a stderr line says so);
  configurable thresholds and exit 3 (#276), and `taguru evaluate
  compare` (#277), land as separate follow-up changes on top of this.
- New top-level verb `taguru benchmark search` (#260, implementing ADR
  0003 §11): builds one context per model (`PREFIX::MODEL_ID`) from a
  finished `taguru benchmark extract` results directory's own batch
  files, runs a shared `eval.jsonl` question set against every one of
  them over `POST /contexts/{name}/sources/search`, and writes
  `results/retrieval.json` — per-case/per-model hit counts, lane
  evidence, and model-pair hit-set overlap (empty-result rate, source
  diversity, `(source, paragraph)` Jaccard, mean rank difference),
  gold-data-free and judgment-free by construction like
  `differences.jsonl`. When a case carries `expected_sources` or
  `expected_concepts`, recall@k and MRR are also computed from them —
  the one thing this shares with #215's own quality gate; `eval.jsonl`
  is deliberately the shared dataset format between the two. Corpus
  import is idempotent (a per-source create-or-replace) and refuses to
  overwrite a context this run did not itself create. A server this
  run cannot reach is a hard failure; an older server's response
  missing `plan`/lane fields degrades those fields to `null` without
  failing the case. See `docs/benchmark.html#search` for the full
  schema and metric catalog.
- New top-level verb `taguru benchmark compare` (#257, implementing ADR
  0003 §9.3/§10): reads a finished `taguru benchmark extract` results
  directory — `manifest.json`, `runs/*.jsonl`, and the written
  `cells/**` batches — and derives `measurements.json`/
  `measurements.csv`, calling no model and touching no network. Covers
  latency/token/throughput distributions, attempt state and finish-
  reason rates, document outcome rates, and extraction-shape counts
  (association/alias/question volume, positive/negative weight split,
  distinct subjects/relations, paragraph attribution, relation reuse,
  and batch-writer health checks), each with a machine-readable
  `unit`/`statistic`/`source`/`caveat` embedded in the artifact's own
  `definitions` block. No single score or ranking is possible by
  construction: every per-model/per-cell map is key-ordered
  (`BTreeMap`), a unit test asserts the emitted key set never contains
  `rank`/`score`/`winner`/`best`/`recommended`/`overall`/`delta_vs_*`,
  and `measurements.csv` is one tidy file with `model_id` always a
  data column. Percentiles are nearest-rank, never interpolated. See
  `docs/benchmark.html#compare` for the full metric catalog.
- New top-level verb `taguru benchmark extract` (#256, implementing ADR
  0003 §5/§6/§8/§9.1/§9.2/§10, depending on #262): runs `taguru extract
  --diagnostics-out` across a matrix of models named by a `models.json`
  file, one subprocess per (model, run) cell, every cell over the same
  corpus under the same task settings — a `models.json` entry may
  describe only a provider's identity and capability, never a task
  setting, so the fairness invariant is enforced by construction. Every
  `TAGURU_EXTRACT_*` variable is scrubbed from the child's inherited
  environment and set explicitly per cell (including values left at
  their defaults); each cell gets a fresh `cells/<model_id>/run<NN>/`
  directory so `extract`'s own manifest/checkpoint skip logic can never
  cross-skip between cells, and `--force` is never used. Writes, under
  `--out`: `manifest.json` (run identity, resolved settings, the
  document/chunk dictionary, per-model provider-probe facts, per-cell
  outcomes; version range-accepted like `.ctx` images, since taguru
  both writes and re-reads it on resume), `models.lock.json`
  (`models.json` with defaults folded in, no secrets), and
  `runs/<model_id>.run<NN>.jsonl` per cell (header, document start/end,
  chunk, and attempt records — the diagnostics sidecar's own records
  carried through unmodified plus harness identity, joined by
  `(document_id, chunk_index)` rather than line position). Re-running
  the same `--out` resumes: a cell already recorded `complete` or
  `failed` is skipped outright; an `interrupted` cell is retried into
  its own directory, where `extract`'s own `.extract-manifest.json`/
  `.extract-checkpoints/` resume it at the document/chunk level: a
  changed `models.json`, corpus, or task setting refuses to resume with
  a usage error naming what drifted, rather than silently mixing two
  matrix definitions in one directory. See `docs/benchmark.html` for
  the full schema and `models.json`'s secrets/fairness rules.
- `taguru extract --diagnostics-out`'s JSONL sidecar gains two record
  kinds (#262, ADR 0003 §7): one `kind: "chunk"` record per chunk,
  written before that chunk's first attempt, carrying its
  `chunk_sha256`/`chunk_bytes` and the `paragraph_first`/
  `paragraph_last` range of the server's own canonical paragraph
  numbering that chunk covers (a paragraph-index range, never a byte
  offset — chunking runs on a relabeled document rendering, not the
  original bytes); and one `kind: "document"` record per document
  written, a structured counterpart to the existing human-readable
  summary line (`associations`/`concepts`/`labels`/`questions`/
  `duplicates`/`dropped`/`batch_path`). `AttemptRecord`'s own shape is
  unchanged — existing consumers keep working unmodified by filtering
  on `kind == "attempt"`, the same discriminator the sidecar already
  required. A new `pub(crate) chunk_plan` helper computes this
  provenance in-process, for the benchmark harness (#256) to call
  directly without a subprocess.
- `taguru import` gains `--url URL` (#247, implementing ADR 0002
  §6/§8/§9, depending on #243's shared `src/remote.rs` client):
  pointed at a running server instead of `TAGURU_DATA_DIR`, the input
  is split on batch boundaries only — never mid-batch, reusing the
  exact range `taguru route`'s own cross-shard import splitting
  already computes (`split_batches`, moved from `src/route.rs` into
  `src/ingest.rs` so both callers share it) — and packed into whole
  chunks under a byte budget starting at 4 MiB (half the 8 MiB default
  `TAGURU_MAX_BODY_BYTES`), each POSTed to `/import` in turn. A single
  batch that alone exceeds this fixed client-side budget is a hard
  error naming the source and the one real fix — split that source's
  content upstream of import — since raising the server's cap alone
  cannot help (the budget check happens before the server is ever
  asked), and splitting client-side would reimplement the
  retract-then-apply contract's atomicity boundary outside the
  server. A `413` on a chunk still oversized only because the cap is
  configured lower than assumed halves that chunk (never crossing a
  batch boundary) and resends, halving again on every further `413`
  until the chunk lands or hits a single batch — safe to automate
  since the server refuses a `413` before applying anything.
  `taguru_group`
  records ride after every batch chunk of the run, matching the local
  path's own group-after-every-batch order. `--dry-run` sends every
  chunk as `?dry_run=true` to preview before applying for real. A lost
  connection reports which chunk landed and points at `--dry-run` to
  confirm before resuming — nothing past that point is retried
  automatically; import's retract-then-apply contract already makes
  any resend exact (ADR §8). Combining `--no-embed` with `--url` is a
  usage error: the server's own embedding configuration decides once
  the request lands there. The existing directory-lock refusal (a
  local server already running against the same data directory) gains
  one added line pointing at `taguru import --url` as the way in
  instead. Auth, the target-line print, and the version-skew preflight
  are the same `src/remote.rs` machinery `export --url`/`compact --url`
  use. Without `--url`, behavior is byte-for-byte unchanged (ADR
  §12.2).
- `taguru compact` gains `--url URL [--parallel N]` (#246, implementing
  ADR 0002 §6/§8, depending on #243's shared `src/remote.rs` client):
  pointed at a running server instead of `TAGURU_DATA_DIR`, CONTEXT
  arguments each call their own `POST /contexts/{name}/compact`; with
  none, it calls the server's `POST /maintenance/compact` sweep
  instead — the server picks its own candidates, worst dead ratio
  first, rather than the CLI enumerating every context first.
  `--parallel` parallelizes the per-context HTTP calls the same way it
  parallelizes the local path, reordering the report back to the
  sequential run's output byte for byte; it has no effect on the
  single-request sweep. Every remote, mutating invocation prints its
  target to stderr before sending anything (ADR §5), and a 503 (the
  heavy-op ceiling, or a sweep already in progress) is shown exactly
  as the server reported it, `Retry-After` included when present —
  never retried automatically, since compaction is safe to re-run
  (ADR §8). Auth and the version-skew preflight are the same
  `src/remote.rs` machinery `export --url` uses. Without `--url`,
  behavior is byte-for-byte unchanged (ADR §12.2).
- `taguru export` gains `--url URL` (#245, implementing ADR 0002 §6/§9,
  depending on #243's shared `src/remote.rs` client): pointed at a
  running server instead of `TAGURU_DATA_DIR`, it enumerates `GET
  /contexts` and `GET /groups` (keyset-paged) and fetches each item
  from `GET /contexts/{name}/export` / `GET /groups/{name}/export`,
  writing the exact files a local export would under `--out` — groups
  only on a full export, exactly as offline. Auth rides
  `TAGURU_API_TOKEN` (or the first `name:token` entry of
  `TAGURU_API_TOKENS`), the same variables the server itself reads; no
  `--token` flag is added, and a URL carrying `user:password@` is
  refused (ADR §7). Every run prints the version-skew warning #244
  prepared (`src/remote.rs`'s `warn_on_version_skew`, now wired up)
  and a stderr note that this is **not** a point-in-time snapshot
  across contexts — each context's own stream is internally
  consistent, but contexts are fetched one request at a time; an
  operator needing a whole-server point-in-time snapshot already has
  one in the replication bucket (`TAGURU_REPLICATE_URL`) plus `taguru
  restore`. Without `--url`, behavior is byte-for-byte unchanged
  (ADR §12.2).
- `taguru extract` gains an opt-in `--diagnostics-out FILE` /
  `TAGURU_EXTRACT_DIAGNOSTICS` JSONL sidecar (#200, implementing ADR
  0001 §10) — the follow-up to #188's motivating failure, where a
  malformed-JSON source could not be diagnosed as truncation versus a
  genuine model syntax error versus a thinking-mode empty answer
  because no per-attempt finish/usage metadata survived the run. One
  record per LLM attempt (written incrementally — a killed run keeps
  every record already flushed) carries `source`, `chunk_index`,
  `attempt`, `max_attempts`, `stage` (`"item"` vs the Stage 2
  `"cross_chunk"` correction), the ADR §7 terminal state
  (`stop_valid`/`stop_malformed`/`length_limited`/`empty`/`refusal`/
  `timeout`/`transport`, classified from provider metadata before any
  parsing), `length_limited`, `parse_error`, `validation_issues`,
  `elapsed_seconds`, and a nested `provider_metadata`
  (`finish_reason` as received, `input_tokens`/`output_tokens`/
  `total_tokens` when the backend reports usage) — field names mirror
  `langchain-taguru`'s `AttemptFailed`/`ProviderMetadata` events
  wherever the concept matches, parity-tested against the serialized
  shape on both sides. Four transport-layer retries inside one
  attempt are reported as the single attempt they are, not four.
  Metadata only by default; `TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES`
  opts into the model's raw answer text per record, byte-capped at
  capture like `TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES` — chain-of-
  thought is never captured under any setting. The manifest stays
  exactly what it has always been, a skip-index of successes, with no
  diagnostics mixed in. Off by default: without the flag, stdout/
  stderr are unchanged byte for byte.
- `langchain-taguru` (Python) replaces `TaguruIngester`'s silent
  merge-level item drop with lossless JSON repair and path-specific
  corrective retry (#180, implementing ADR 0001 §8 — the Python twin of
  #199's Rust behavior). Validation moves to a lenient walk
  (`interpret_model_output`) that reads the same shape the old
  `ModelOutput.model_validate()` did but collects a path-addressed issue
  for every departure instead of rejecting the whole answer over one
  wrong-typed field; a business-rule-invalid item now earns one targeted
  corrective turn naming its exact JSON path (e.g.
  `associations[0].weight: expected finite non-zero number, got string
  "strong"`) instead of a silent drop, and the source fails outright (no
  `/import` call) if it is still invalid afterward. Two new automatic,
  information-preserving JSON repairs (BOM stripping, unambiguous
  trailing-comma removal) are added on top of the existing fence-
  stripping/widest-braces tolerance. A `length`-terminated answer is
  treated as length-limited even when its content happens to parse
  cleanly — a valid prefix of a cut-off extraction is never imported — and
  a policy refusal (`content_filter`/`refusal`) is now terminal instead of
  spending a corrective turn on it. A dangling-canonical or shadowing
  alias — judgeable only against the full merged name set, never one
  chunk alone — is checked once per document across all chunks, right
  before merge, and spends at most one corrective turn per offending
  chunk the same way. `TaguruIngester(lossy=True)` restores the previous
  drop-and-proceed behavior exactly, reported through
  `IngestOutcome.invalid_dropped`; `IngestOutcome` also gains
  `lossless_repairs` and `correction_attempts`, and `AttemptFailed`/
  `AttemptStarted` gain `stage` (`"item"` vs `"cross_chunk"`) and
  `AttemptFailed` gains `validation_issues`. Validates against the same
  shared fixture corpus `tests/fixtures/model_output/repaired/` the Rust
  twin (#199) uses.
- `langchain-taguru` (TypeScript) ports the same lossless JSON repair and
  path-specific corrective retry to `TaguruIngester` (#181, the TS twin of
  #180/#199). `interpretModelOutput` replaces `coerceOutput`'s
  all-or-nothing scalar coercion with the same lenient, path-addressed
  walk: a wrong-typed or business-rule-invalid item earns one targeted
  corrective turn instead of throwing over the whole answer or being
  silently dropped by `merge()`. BOM stripping and unambiguous
  trailing-comma removal join fence-stripping/widest-braces slicing as
  automatic lossless repairs. A `length`-terminated answer is never
  imported even when it happens to parse, and a policy refusal
  (`content_filter`/`refusal`) is now terminal. `crossOutputIssues` checks
  dangling/shadowing aliases once per document across all chunks, each
  spending at most one corrective turn. `TaguruIngester({ lossy: true })`
  restores the previous drop-and-proceed behavior, reported through
  `IngestOutcome.invalid_dropped`; `IngestOutcome` also gains
  `lossless_repairs` and `correction_attempts`. One deliberate behavior
  change beyond the never-silent-drop default: `parseModelOutput` and
  `TaguruIngester` (lossy or strict) no longer coerce a numeric string or
  boolean into a number for `weight`/`paragraph` — the pydantic-lax-mode
  parity `coerceFloat`/`coerceInt` provided is dropped in favor of the
  Rust producer's stricter, cross-language-consistent parsing (ADR 0001
  §11). Validates against the same shared fixture corpus
  `tests/fixtures/model_output/repaired/` the Rust and Python twins use.
- `add_associations`, `store_passages`, and `POST /import` (including
  MCP's `import` tool) return structured, path-addressed validation
  detail on ingestion refusals (#182, implementing ADR 0001 §8/§11's
  MCP consistency obligation). **Response shape change**: the JSON
  error body gains four additive fields, present only where they apply
  (absent from every success and from every other error) —
  `issues` (up to 20, `{path, kind, expected, actual}` per rejected
  field; `kind` one of `missing`/`type`/`empty`/`too_long`/`range`/
  `over_limit`/`unknown_reference`/`conflict`), `integrity`
  (`"nothing_written"`, or a multi-batch `import` stream's
  `"durable_prefix"` with `durable_batches` naming exactly how many
  earlier batches already landed — never implying any part of the
  REJECTED batch itself landed), and `retryable_after_correction`
  (`true` when a corrected, complete resend can resolve the rejection).
  `add_associations` and `store_passages` now collect every item's
  issues in one pass instead of stopping at the first bad one, so a
  rejection names every offending path at once; both also read a raw
  JSON body now, so a wrong-typed field (e.g. `weight: "strong"`) is
  reported as a path-addressed `invalid_argument` issue instead of a
  generic `malformed_request` — a deliberate, additive shape and
  error-classification change for these two endpoints (existing
  consumers reading only `status`/`code`/`error`/`time` are
  unaffected). Over MCP, a rejected tool call's structured detail rides
  again as `structuredContent` (MCP 2025-06-18+) alongside the
  unchanged prose in `content[0].text`. `taguru import`'s predicted
  alias rejection (`ApplyRefusal::Rejected`) is now a structured
  `AliasRejection` internally; its prose `text()` is byte-for-byte
  unchanged. Tool descriptions and `GET /protocol`'s "Errors and
  limits"/"Ingest loop" sections document the new fields, the `kind`
  vocabulary, the atomicity boundary between `add_associations`/
  `store_passages` and `import`, and the correction discipline
  (preserve every item, correct only the listed paths, resend the
  complete write, never delete-as-repair, add no unsupported fact).
  MCP never retries the extracting LLM itself — correction and
  resubmission are entirely the calling host's responsibility.
- `taguru extract` replaces merge-level silent item drop with
  path-addressed corrective retry (#199, implementing ADR 0001 §8) — a
  business-rule-invalid item (a wrong-typed or zero/non-finite/over-cap
  weight, an empty or oversized name, a dangling or shadowing alias, an
  unknown alias kind, an out-of-range or malformed question) earns one
  targeted corrective turn naming its exact JSON path (e.g.
  `associations[1].weight: expected finite non-zero number, got string
  "strong"`) instead of a silent drop, bounded by the existing
  `TAGURU_EXTRACT_MAX_ATTEMPTS`/`TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES`
  controls. A dangling-canonical or shadowing alias — judgeable only
  against the full merged name set, never one chunk alone — is checked
  once per document, right before merge, and spends at most one
  corrective turn the same way. `--lossy` / `TAGURU_EXTRACT_LOSSY`
  restores the previous drop-and-proceed behavior exactly, and the
  report line marks a lossy run's drops explicitly (`N item(s) dropped
  (--lossy)`) so they are never confused with a policy trim (a
  `--questions` cap overflow, a volunteered question nobody asked for).
  `--lossy` is a manifest compute input like every other extraction
  control (old manifests default to strict correction and keep matching
  all-defaults runs). The shared fixture corpus
  `tests/fixtures/model_output/repaired/` names each (rules, answer,
  issues, corrected) tuple that #180/#181 (the Python/TypeScript twins)
  are expected to validate against identically.
- `taguru extract` puts structured output on the wire and gains an explicit
  output budget with deterministic length escalation (#198, implementing
  ADR 0001 §4.1/§4.2/§7) — `--structured-output` /
  `TAGURU_EXTRACT_STRUCTURED_OUTPUT` selects a rung of the capability
  ladder (`json-schema` sends the canonical `ModelOutput` schema as
  `response_format` with `strict` requested, `json-object` sends JSON
  mode, `auto` probes the endpoint once at startup and keeps the
  strongest rung the answer verifies, falling back to today's bare
  prompted JSON); `--max-output-tokens` /
  `TAGURU_EXTRACT_MAX_OUTPUT_TOKENS` sends `max_tokens` explicitly, and a
  completion that still ends `finish_reason: length` is never re-asked
  under the same limit — the budget escalates once (a neutral resend,
  the truncated answer discarded, its valid-looking prefix never
  imported), then the chunk splits with paragraph labels preserved, then
  the source fails with a named diagnosis. Engaging either control also
  makes policy refusals (`content_filter`) terminal instead of burning a
  corrective turn, and reports a schema-constrained answer that fails
  validation as provider non-conformance on stderr. Both controls are
  manifest compute inputs (changing them re-extracts; old manifests
  still match all-defaults runs). Defaults are unchanged: with neither
  control set, the request body and the whole retry behavior — including
  the truncation-aware "answer SHORTER" correction, now the
  no-budget-control fallback — are byte-for-byte today's.
- The extractors gain a canonical `ModelOutput` JSON Schema (#185) — the
  `{associations, aliases, questions}` shape `taguru extract`'s parser, the
  Python SDK's pydantic model, and the TypeScript SDK's `parseModelOutput()`
  already all accept, hand-written once (never derived from the parser
  types) and exported as `model_output_json_schema()` (Rust, the source of
  truth) and `MODEL_OUTPUT_JSON_SCHEMA` (Python and TypeScript, hand-
  mirrored). All three copies are tested against the same shared
  accept/reject payload fixtures. `TaguruIngester` (Python and TypeScript)
  gains an opt-in `structured_output` flag, default `False`/`false`, that
  asks the chat model for schema-constrained generation via
  `with_structured_output()` / `withStructuredOutput()` instead of parsing
  free text — provider/model dependent, so a chat model that cannot bind
  tools raises immediately at construction rather than per attempt; either
  way the result still goes through the same `ModelOutput` revalidation and
  merge()'s business-rule checks.
- `taguru extract` and `TaguruIngester` (Python and TypeScript) gain
  bounded structured-output controls and a configurable retry strategy
  (#178, #184) — `--fact-budget N` / `fact_budget` asks the model to keep
  a chunk's answer to at most N associations, folded into the system
  prompt; `TAGURU_EXTRACT_MAX_ATTEMPTS` / `max_attempts` raises the total
  attempts at valid JSON per chunk from the default of 2 up to 10 (or
  down to 1, skipping the corrective turn entirely);
  `TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES` / `corrective_context_bytes`
  caps how much of a malformed answer gets replayed back on the next
  attempt (`0` omits it behind a placeholder), default unchanged at
  replaying it in full; and once a provider's own finish reason
  (`"length"`, or Anthropic's `"max_tokens"`; read from
  `AIMessage.response_metadata` in the SDKs) says a bad answer was cut
  off at its output-length cap, the corrective ask itself switches from
  "try again" to "try again shorter," naming the fact budget when one is
  set — the fix for a local model stalling for minutes replaying and
  re-requesting the same oversized malformed answer. Defaults are
  unchanged in every case.
- `TaguruIngester` gains an optional `on_event` progress callback (#177) —
  synchronous, typed events (`document_started`, `chunk_started`,
  `attempt_started`, `attempt_failed`, `chunk_completed`, `import_started`,
  `import_completed`, `embedding_refresh_started`/`completed`/`warning`)
  fire from both `ingest_text` and `aingest_text`, so a caller can show
  live progress and see *why* a corrective attempt fired (parse error,
  provider finish reason, token usage when the model reports them)
  without copying the private extraction helpers. Callback exceptions are
  caught and reported via `warnings.warn` rather than failing the ingest.
- `langchain-taguru` (TypeScript) gains the same optional `on_event`
  progress callback (#201, the TypeScript twin of #177/#183) —
  synchronous, typed events (`document_started`, `chunk_started`,
  `attempt_started`, `attempt_failed`, `chunk_completed`,
  `import_started`, `import_completed`,
  `embedding_refresh_started`/`completed`/`warning`) fire from
  `ingestText`, field-for-field with the Python twin, so a caller can
  show live progress and see *why* a corrective attempt fired (parse
  error, provider finish reason, token usage when the model reports
  them) without copying the private extraction helpers.
  `extractFinishReason` folds into a shared `providerMetadata` reader
  that also reads `usage_metadata`, feeding `AttemptFailed`'s
  `provider_metadata`. Callback exceptions are caught and reported via
  `console.warn` rather than failing the ingest. `events.ts` and its ten
  event shapes are exported from the package index like the Python
  package exports its events.
- `taguru extract` gains durable per-chunk checkpoints, cooperative
  stop, and resume (#179) — a long document no longer loses every chunk
  already extracted when a run is interrupted (Ctrl+C, a preemptible
  instance reclaimed, a later document's panic). `--out` gains a
  `.extract-checkpoints/` directory, one JSON file per document, holding
  every chunk unit landed so far; a rerun over the same `--out` skips
  re-sending them to the model. Each unit is keyed by the hash of its
  own text rather than by chunk index, so the ADR 0001 §7 length
  ladder's split rung — which can divide an oversized chunk differently
  across two runs — still resumes correctly: a completed sub-piece is
  recognized regardless of how either run split around it. Checkpoint
  files carry the same compute-input fingerprint the manifest checks;
  any mismatch invalidates the whole file rather than risking a false
  reuse, and `--force` now discards existing checkpoints too. A
  document's checkpoint file is deleted once its batch lands and kept
  when the document ultimately fails, since the chunks it did complete
  are still good work. `--dry-run` folds a "reusable" count from
  checkpoints into its existing per-document line. Ctrl+C/`SIGTERM` is
  cooperative: the first signal finishes the in-flight chunk, lets it
  checkpoint, and stops before the next chunk (sequential mode) or the
  next document (`--parallel`, whose concurrent dispatch is not
  interrupted mid-flight); the process exits `130` and a rerun resumes
  from the checkpoints. A second signal forces an immediate exit, also
  `130`, for when the graceful path itself is stuck. No new Cargo
  dependency — cooperative stop reuses the `tokio` `signal` feature
  already in the workspace for the server's own shutdown handling.
  Python and TypeScript `TaguruIngester` checkpoint/resume ports are
  tracked as separate follow-up issues; #179 stays open until they
  land.
- `langchain-taguru` (Python) `TaguruIngester` gains durable per-chunk
  checkpoints, cooperative stop, and resume (#211, the Python twin of
  #210's Rust behavior for #179) — an interruption mid-document
  (a killed process, a reclaimed spot instance) no longer loses every
  chunk already extracted for it. A new optional `checkpoint_store`
  constructor argument accepts anything implementing the three-method
  `CheckpointStore` protocol (`load`/`save`/`delete`, keyed by source
  id — the SDK, not the store, owns the fingerprint gate and the
  content-hash unit keying, so a custom backend cannot cause a false
  reuse even in principle); `FilesystemCheckpointStore` is the
  batteries-included default, one JSON file per document under a given
  directory, written atomically (temp file, fsync, rename, parent-dir
  fsync). Each chunk unit is keyed by the hash of its own (labeled)
  text rather than its index, matching the Rust side's #179 amendment
  and leaving room for a future Python split rung to resume correctly
  the same way. The checkpoint fingerprint mirrors the Rust side's
  `CheckpointFingerprint` (content hash, model identity, prompt
  version, and every output-shaping setting) minus `max_attempts`/
  `chunk_bytes` — deliberately, matching Rust: a validated chunk output
  does not depend on retry budget, and a chunk-size change just makes
  today's pieces hash differently, a safe cache miss rather than a
  correctness hazard. `checkpoint_model_id` is required when the chat
  model exposes none of `model`/`model_name`/`model_id`/
  `deployment_name` for the fingerprint to key on — checked at
  construction time, not first use. A reused chunk emits the same
  `ChunkStarted`/`ChunkCompleted` event pair as a fresh one (with
  `llm_calls=0` and a new `ChunkCompleted.reused=True`), and still
  participates in Stage 2 cross-chunk correction exactly like a
  freshly-extracted chunk, replaying its own stored conversation. A
  new `should_stop` argument (a zero-argument callable, or a
  `threading.Event`) to `ingest_text`/`aingest_text`/
  `ingest_documents`/`aingest_documents` stops cooperatively between
  chunks (and between documents); `IngestOutcome` gains `interrupted`
  (true when a stop was honored — not a failure, and bypasses
  `raise_on_error` in the batch APIs) and `chunks_reused`. A
  document's checkpoint is deleted once its batch actually lands in
  `/import` and kept when the document ultimately fails; `dry_run=True`
  records checkpoints (real model calls happen) but never deletes them,
  since no import ever lands. The async twin's store calls run via
  `asyncio.to_thread` so a synchronous backend (a filesystem write, a
  blocking object-storage SDK call) cannot stall the event loop. The
  TypeScript port remains tracked separately; #179 stays open until it
  lands too.
- `langchain-taguru` (TypeScript) `TaguruIngester` gains the same durable
  per-chunk checkpoints, cooperative stop, and resume (#212, the
  TypeScript twin of #211's Python behavior and #210's Rust behavior for
  #179 — the last outstanding port; #179 can now close). Semantics are
  identical to the Python entry above (unit-hash keying, the fingerprint
  gate, atomic writes, delete-on-success/keep-on-failure, `dry_run`
  recording but never deleting); a few fields are TypeScript-idiomatic
  rather than byte-for-byte ports: `CheckpointStore`'s three methods are
  async and exchange `Uint8Array` (`taguru_langchain`'s `bytes` twin);
  `should_stop` accepts a zero-argument function or an `AbortSignal`
  (the JS analogue of `threading.Event`); hashing uses the Web Crypto
  global (`crypto.subtle`) so the module needs no top-level `node:`
  import, and `FilesystemCheckpointStore`'s `node:fs`/`node:path` use is
  behind dynamic imports (the same pattern the core SDK's
  `Context.exportToFile` already uses), which makes
  `FilesystemCheckpointStore.pathFor` async; `checkpoint_model_id` is
  required when the chat model exposes none of `model`/`modelName`/
  `modelId`/`deploymentName`; `ChunkCompleted.reused` is a required
  (not optional) boolean.

### Changed
- **Response-shape change** (#244, implementing ADR 0002 §10): `GET
  /health`'s success body is JSON now — `{"status": "ok", "version":
  "<the server's own version>"}` (`Content-Type: application/json`) —
  instead of the bare text `ok`; `taguru route`'s own `/health` gains
  the same `version` field beside its existing `router`/`shards` keys.
  The `503` failure bodies (`unhealthy`, `maintenance`) are unchanged,
  and `/health` stays auth-exempt. `taguru health` is compatible by
  construction — it judges the status code and prints the body
  without parsing it, so it now prints the JSON line — but anything
  scripted against the literal body `ok` should read the `status`
  field instead. This is the version-discovery half of ADR 0002 §10; the
  `--url` forms of `import`/`export`/`compact` (#245-#247) will read
  the field once per run and print a one-line stderr warning — never
  blocking — when the server's major.minor differs from the CLI's own;
  the shared mechanism lands in `src/remote.rs` now, ready for those
  verbs to call.
- **Behavior change** (#199, ADR 0001 §12.2 — approved by the ADR
  itself, not a later regression): by default, `taguru extract` no
  longer silently drops a business-rule-invalid item while reporting
  the extraction as fully successful. An invalid item now earns a
  corrective turn; if the corrected answer is still invalid, **the
  source fails and no batch file is written** — a run that previously
  wrote a batch quietly missing that one item now reports failure and
  writes nothing for that source instead. Pass `--lossy` /
  `TAGURU_EXTRACT_LOSSY` to keep the previous drop-and-proceed behavior
  exactly.

### Fixed
- **On-disk format change**: `taguru extract`'s and `langchain-taguru`
  (Python)'s chunk checkpoint file names now always carry a
  16-hex-character content-hash suffix, closing a short-source-id
  collision (#227). The flatten step (`/`, `\`, `:` → `__`) is not
  injective — `"a/b"`, `"a:b"`, and `"a__b"` all flattened to the same
  string, and since none was anywhere near the old 120-byte
  hash-suffix threshold, they collided on one checkpoint file and could
  silently share (and overwrite) each other's checkpoint progress.
  `langchain-taguru` (TypeScript) already suffixed unconditionally
  (#226/#212); all three ports now agree. Every source's checkpoint
  file name changes, not just previously-colliding ones; a
  `.extract-checkpoints`/checkpoint-store directory from before this
  fix is not read under the new names and degrades to a one-time cold
  start for any in-flight interrupted document — safe, not a false
  reuse.

## [0.4.0] - 2026-07-20

### Added
- Source metadata and pre-lane search filters (#167) — the two B-1
  entries #148 deferred as a data-model change. Every stored source
  now carries a server-stamped `stored_at` (epoch seconds, stamped
  once as the WAL op is built so replay never re-stamps), an optional
  user-supplied document `date`, and `tags` — accepted by
  `POST /contexts/{name}/sources` (`tags`/`dates` per-source maps;
  MCP: `store_passages`) and the import batch format (riding the
  `passage` line; export writes them back, and import preserves an
  exported `stored_at` so restores don't re-date the corpus), durable
  through the passage WAL, a new S5 snapshot generation (S1–S4 read
  forever), compaction, WAL shipping, and restore-on-start. Passage
  search (`POST /contexts/{name}/sources/search`, `POST
  /sources/search`; MCP: `search_passages`; SDKs:
  `search_passages`/`searchPassages`) takes `tags` (any-of) and a
  half-open `[since, until)` window over each source's `date ??
  stored_at`, resolved to an eligibility set BEFORE the BM25/vector
  lanes run — BM25 statistics stay corpus-global (the filter gates,
  never re-weights), the ANN probe widens until its oversample target
  is met among eligible rows, and absent metadata never matches a
  filter (pinned by tests). The plan gains a per-context
  `filter: {eligible_sources, total_sources}` block,
  `search/explain` gains the same filter params and a `filtered_out`
  verdict, `GET /contexts/{name}/sources` lists metadata back under
  `entries`, and retrieval/semantic cache keys carry the filter on
  both search variants (same query, different filter: never one
  entry).
- Community analysis as an offline derived index (#166) — the
  corpus-overview surface GraphRAG's global search answers and taguru
  had no verb for. `GET /contexts/{name}/communities` detects
  communities on the concept graph server-side (hand-rolled
  deterministic Louvain with a component-split pass, `louvain-cc/1`;
  hierarchical via aggregation; heavy-ops gated) and streams them with
  the revision snapshot the analysis was cut at. `taguru communities`
  — an HTTP client to a running server, like calibrate — turns that
  into an artifact: an ordinary context (default `{name}::communities`)
  holding one summary passage per community, membership and hierarchy
  as associations, and a manifest recording the source revision.
  Summaries come from the extract provider and are incremental by
  content fingerprint: an unchanged graph re-runs with zero LLM calls,
  and only changed communities re-summarize.
  `POST /contexts/{name}/communities/search` (MCP: `search_communities`;
  SDKs: `search_communities`/`searchCommunities`) ranks the summaries
  with the same two-lane passage search — plan, floors and all — and
  answers with an honest staleness verdict (`stale: true` = the source
  graph moved since derivation); a missing artifact is a refusal
  naming the build command, never an empty result. Because the
  artifact is an ordinary context, quotas, retrieval caching (a new
  cache op keyed on the artifact AND the source's current graph
  revision), export/import, groups, and router-mode routing all apply
  unchanged.
- Image supply chain, made verifiable end to end (#138). The release
  pipeline already signed the multi-arch manifest (Sigstore keyless)
  and attached BuildKit SBOM + SLSA provenance since 0.2.0 — but the
  SBOM was vacuous: a scratch image gives the scanner nothing, so two
  releases shipped an SPDX document cataloguing zero crates. The
  binary is now built with `cargo auditable`, which embeds the
  Cargo.lock crate list in the executable — the same attached SBOM
  now carries every crate with its version, and a bare binary is
  auditable outside any container (`cargo audit bin`). A new `verify`
  job gates every release from a consumer's seat — blank runner, no
  registry login, no OIDC — on three checks: the signature binds this
  repository's exact workflow identity (unanchored identity patterns
  match look-alike repos; the docs pin it), the SBOM actually lists
  crates (a ≥100 floor, against the vacuous-SBOM failure that shipped
  silently), and the provenance names this repository's CI run.
  Verification is documented for consumers in SECURITY.md ("Verifying
  a release"), and deploy/README.md now spells out digest pinning —
  a tag pin is a convention, `tag@sha256:…` is a guarantee — with the
  kustomize `digest:` override alongside.
- Hot-reload for the auth table (#134): `TAGURU_API_TOKEN`,
  `TAGURU_API_TOKENS`, and `TAGURU_KEY_SCOPES` — and nothing else —
  now reload on a running server, so key rotation no longer costs the
  restart-outage a single-writer boot implies. Two triggers, one
  swap: SIGHUP (unix; previously the unhandled default disposition
  made SIGHUP *kill* the server), and a ~5s watch on the `--config`
  file when one was given — which is what makes the Kubernetes
  secret-volume flow hands-free (mount the Secret as a file, `kubectl
  apply` a rotation, the kubelet's atomic symlink swap reaches the
  watch; no exec, no signal) and gives non-unix platforms, which have
  no SIGHUP, the same rotation. Reload sources mirror boot precedence
  exactly — a shell-set variable keeps winning over the file at every
  reload (which is also why an env-only deployment reloads as an
  explicit no-op: a live process's environment cannot change).
  Fail closed on every path: an unreadable or malformed source keeps
  the previous table armed with a loud error line, and the one
  transition a reload must never perform — "tokens configured" → "no
  tokens", which would silently reopen the server unauthenticated —
  is refused outright (arming keys on an open dev server stays
  allowed; that direction closes). Every gate reads the ring through
  one `SharedKeyring` handle and resolves authentication AND scope
  from a single per-request snapshot (the bearer gate now stamps the
  resolved `KeyScope` onto the request, `enforce_authorization`
  judges from it keyring-free, and `/mcp` stamps it through to
  dispatched tool calls), so in-flight requests see the old table or
  the new one — never a torn one, and never a removed key falling
  through `scope_of` to the unscoped admin default mid-request.
  OAuth delegations minted from a removed key die with it (the
  per-request `recognizes` check reads the swapped ring); per-key
  rate-limit buckets need no hook (new names start at full capacity,
  removed names' idle buckets fall to the existing prune). Each
  reload leaves one `taguru::audit` line — keys added / removed /
  rotated (same name, new bytes — the k8s case a name diff can't
  see) / rescoped, names only, never token bytes, with an explicit
  "no change" line so a SIGHUP is never silent — plus two counters:
  `taguru_keyring_reloads_total` and the alertable
  `taguru_keyring_reload_refusals_total` ("the rotation you think
  you performed didn't take"). `taguru route` is untouched: the
  router holds no keyring by design.
- Per-context quotas (#136), declared as one JSON env in the
  `TAGURU_KEY_SCOPES` mold — `TAGURU_CONTEXT_QUOTAS='{"name":
  {"storage_bytes": …, "cache_bytes": …}}'`, each field optional but
  never both absent; a broken declaration refuses boot, like broken
  credentials. `storage_bytes` caps the context's whole on-disk family
  (image, both WAL lanes, passages snapshot, sidecars — the same sum
  `taguru_context_disk_bytes` serves, read from the live WAL
  bookkeeping plus the flush-refreshed #137 snapshot, which now stays
  on whenever a storage quota is declared even with the gauges off):
  at or over the ceiling, growth writes refuse with the
  already-documented 507 `storage_full` across every entrance — graph
  batches (associations/aliases, via the one `logged_write`
  chokepoint, gated only when a batch carries growth ops),
  `store_passages`, and `/import`, which stops before the first capped
  batch as a resumable prefix exactly like a spent deadline (dry runs
  stay advisory). Retract, unalias, `DELETE`, and compaction stay open
  at the ceiling — shrinking is how a tenant gets back under, the line
  the passage store's own cap already draws. `cache_bytes` is the
  ceiling side of the pinning floor: no reservation while there is
  slack, but under cache pressure a context past its declared share is
  evicted before any compliant one, so the eviction damage one
  saturating context can inflict on the rest is bounded by its ceiling
  (pinning still wins — a pinned context never enters the sweep).
  Declared ceilings surface as
  `taguru_context_quota_bytes{context,resource="storage"|"cache"}`
  beside the #137 usage families (same knob, same top-N cut), and
  refusals count on `taguru_storage_quota_refusals_total`. Offline
  commands (`taguru import`/`compact`/…) run as the operator, outside
  the policy; a replica refuses writes before any gate and honors only
  the eviction ordering.
- Ratio-triggered auto-compaction for contexts (#135), default on:
  each flusher tick rebuilds at most the one worst context whose dead
  ratio (dead edges / total edges — the bookkeeping the maintenance
  sweep already reads, live for hot contexts, sidecar stats for cold)
  strictly exceeds `TAGURU_AUTO_COMPACT_RATIO` (default 0.5, i.e. dead
  weight outgrew live content — the graph-side restatement of the
  passages store's own self-compaction ratio). The compaction takes a
  permit from the same `TAGURU_MAX_CONCURRENT_HEAVY_OPS` pool manual
  calls contend on (no free slot: the candidate waits for a later
  tick), runs under a 60-second budget so one oversized rebuild cannot
  stall the loop that persists every other context (a context that
  blows the budget is set aside for the process's lifetime with a
  pointer at `POST /maintenance/compact` / offline `taguru compact` —
  retrying a rebuild that cannot finish would burn the budget every
  tick), reuses `compact_context` verbatim (same crash guarantee: the
  fresh image carries the old WAL watermark), leaves the same
  `taguru::audit` "context compacted" line with `trigger="auto"` and
  the measured ratio, and shows on `/metrics` as
  `taguru_auto_compactions_total{outcome}`,
  `taguru_auto_compact_reclaimed_bytes_total`, and
  `taguru_auto_compact_last_success_timestamp_seconds`.
  `TAGURU_AUTO_COMPACT=0` restores manual-only compaction for
  scheduled quiet-window sweeps; replicas never auto-compact (they run
  no flusher — their images belong to the primary).
- Per-context capacity gauges on `/metrics` (#137), behind
  `TAGURU_METRICS_PER_CONTEXT` (default off; `1`/`all` = every
  context, `N ≥ 2` = the top-N by total disk bytes): the
  `taguru_context_*` families — on-disk bytes by file family (image,
  graph WAL, passages snapshot, passages WAL, sidecars), modeled
  resident bytes, the pinned flag, and concept/association/label/
  source counts. Sizes come from a stat sweep at each flush tick,
  `POST /flush`, and boot — never from the scrape, which reads only
  registry state; counts and residency are live for hot contexts and
  the last-saved snapshot for cold ones, the same semantics
  `GET /contexts` serves. The WAL series reuse the existing live
  bookkeeping, so they sum exactly to `taguru_wal_bytes` and friends.
- `taguru calibrate` (#131): measures the semantic-floor bands of a
  running server's embedding model instead of prescribing the manual
  ritual. `--context NAME --probes FILE` (TSV `cue<TAB>expected`
  pairs) drives the resolve/explain machinery: the expected name's own
  gloss cosine feeds the upper band (measured floor-independently),
  the best other semantic candidate a 0.05-floor resolve surfaces
  feeds the lower, and the report prints both distributions, the gap,
  and a suggested `TAGURU_SEMANTIC_FLOOR` mid-gap — `--json` for
  automation, one provider embed per probe (the cue cache covers the
  second call). Probes whose cue lexically resolves — the step humans
  get wrong; the semantic tier never scores them — are excluded
  loudly, each with its own diagnosis; overlapping bands earn a
  warning verdict, never a fabricated number. Auth and URL resolution
  ride the same variables the server reads (`TAGURU_API_TOKEN`/
  `TAGURU_API_TOKENS`, `TAGURU_ADDR`, `--config`), and the run is
  read-only, so a replica serves it.
- `GET /contexts/{name}/embeddings` (#131): the embedding identity in
  one read — the configured provider model beside the (model, width)
  each vector sidecar was actually built with (#133's recorded
  identity, now exposed), plus row counts. What `taguru calibrate`
  stamps its report with, and the state to check after a model switch
  without provoking a search. Read role, replica-safe.
- Execution plans on every search response (#151): recall/query
  (single-context and cross) gain a `plan` object beside
  `total`/`matches` — additive — carrying `contexts`, the list of
  contexts actually consulted in effective order. For the cross
  variants that is the resolved target list (groups expanded, the
  key's grants applied), which the tagged matches alone cannot
  reconstruct when a target comes back empty; it names nothing a
  caller cannot already see through `GET /groups`. The passage-search
  half of #151 is a response-shape change and lives under **Changed**
  below. The MCP `retrieve` tool and both SDKs' composed `retrieve()`
  forward the fallback search's plan as `search_plan` (null/absent
  when no fallback ran).
- Semantic retrieval cache for passage search (#153): with
  `TAGURU_SEMANTIC_CACHE_THRESHOLD` set (off by default), a
  paraphrased `sources/search` (single-context or cross, MCP
  included) can answer from an equivalent earlier query's exact-cache
  entry. Equivalence requires the query-vs-query embedding cosine
  (through the same cue cache the search uses — no extra provider
  calls on the fresh path) to clear the threshold AND a
  negation/number/entity guard to find no mismatch between the two
  query texts, so "does it X" never serves "does it not X", a changed
  number, or a swapped name at high cosine. The tier stores only
  equivalence claims — no payloads, no invalidation machinery: a
  match rewrites the request's exact key to the canonical query's
  params under current revision fingerprints, so every #150 freshness
  guarantee (revision lanes, identity nonce, delete-recreate, replica
  lineage) applies unchanged, and a post-write serve becomes a
  `stale` fall-through that re-canonicalizes the cluster. Outcomes in
  `taguru_semantic_cache_total{outcome=hit|stale|guarded|miss}` plus
  a `taguru_semantic_cache_entries` gauge; semantic serves add
  `similarity` and `matched` to the opt-in search log line. Guard
  blind spots (spelled-out and kanji numerals, sentence-initial
  English entities, entities in unsegmented scripts) are documented
  in `src/registry/semantic_cache.rs` — the threshold is the primary
  filter and the default posture is off. No request or response shape
  changes.
- Exact-match retrieval cache (#150): an identical
  recall/query/passage-search request (single-context and cross
  variants, MCP tool calls included) against an unchanged corpus now
  answers from the stored response bytes without re-running the
  search. Invalidation is the key itself: each entry is keyed on the
  resolved target list plus, per target, the #149 revision lanes that
  surface depends on (recall/query: graph+passages; passage search:
  passages+config) and a per-incarnation identity — read before the
  search runs — so a write simply makes stale entries unreachable
  (delete-recreate and replica lineage switches included), with no
  TTL and no purge hooks. Scoped keys share an entry exactly when
  their grants resolve a request identically. Byte-budgeted LRU:
  `TAGURU_RETRIEVAL_CACHE_BYTES` (default 32 MiB, `0` disables).
  Hit/miss per op lands in `taguru_retrieval_cache_total` with
  entry/byte gauges beside it, while `taguru_searches_total`, the
  passage lane contributions, and per-context usage counters keep
  counting served responses — dashboards read continuously. Cache
  hits emit the opt-in search log line with `cached=true`. No
  request or response shape changes.
- Context revision counters (#149): every directory row (`GET
  /contexts`, `GET /contexts/{name}`, and the MCP
  `list_contexts`/`get_context` pass-throughs) now carries
  `revision: {graph, passages, config}` — applied graph writes, the
  passage log watermark, and config/embedding changes respectively —
  the change token the upcoming retrieval caches key on. Group rows
  gain a `fingerprint` hashing the scope-visible transitive members'
  counters, so a group-level cache invalidates exactly when a
  relevant member changed. Compare for equality only: within one
  process the counters are live and strictly monotonic; across a
  crash a cold context can lag until its first load, and a cache
  that outlives the process must treat a restart (or a
  delete-recreate) as invalidation. Response-shape addition only;
  the `.meta.json` sidecar gains a `revision` field older sidecars
  simply lack (they read as zeros until the first flush).
- Passage search takes a per-request `semantic_floor` (#148): a
  one-call override of the vector lane's cosine floor — request beats
  the context setting beats the server default, the same chain
  resolve's override walks — on `/contexts/{name}/sources/search`,
  cross-context `/sources/search`, and `…/sources/search/explain`
  (the explanation reports the floor it actually ran under). Exposed
  through the MCP tools and both SDKs. It floors only the vector
  lane: BM25-only hits still return, and the fused score stays rank
  arithmetic.
- Embedding tier resilience (#132): the provider now sits behind a
  small circuit breaker — three consecutive failed attempts open it,
  every embedding call then fails fast (the lanes behind it degrade
  exactly as they do today: `sources/search` serves its lexical lane,
  resolve serves lexical candidates or answers the existing
  `embeddings_failed`) instead of each paying the provider timeout,
  and after a 30s cooldown a single probe decides whether to close.
  Breaker state, opens, and short-circuit counts land on `/metrics`
  as `taguru_embedding_breaker_state` /
  `_consecutive_failures` / `_opened_total` / `_short_circuits_total`
  (present only when a provider is configured).

### Changed
- **On-disk format**: the gloss vector sidecar (`{stem}.vectors.bin`)
  now records its vector width beside the model — header `TAGURUV2` →
  `TAGURUV3` (#133). Existing V2 sidecars load exactly as before (the
  width is taken from the rows, which loads now verify are uniform)
  and are stamped to V3 by their next save, so the upgrade costs no
  provider spend; a binary older than this release, however, reads a
  V3 file as corrupt and re-embeds on its next refresh. Loads of
  either header refuse rows that mix widths, and a V3 header
  disagreeing with its rows, the same way they refuse other
  corruption: discard, warn, re-embed — never serve. The passage
  sidecar (`TAGURUP2`) already recorded (model, dim) and is unchanged.
- **Response shape**: `POST /contexts/{name}/sources/search` and cross
  `POST /sources/search` now answer `{plan, hits}` instead of a bare
  hit array (#151) — the hits themselves are unchanged, moved under
  `hits`; `plan.contexts` carries, per context actually searched,
  whether each lane ran (`bm25`/`vector`, mirroring the per-hit
  `lanes` shape), the reason when one was skipped — embeddings off,
  nothing embedded yet, model changed, provider refused, in the same
  prose `sources/search/explain` uses — and the vector lane's
  effective cosine floor when it swept (the override → context
  setting → server default chain, resolved per context). This is what
  makes "did the semantic lane actually run" visible without a
  diagnostic call; a zero-hit page under a skipped lane no longer
  reads as "nothing matched". Breaking for clients that parse the
  search result as an array: both SDKs (`search_passages` /
  `searchPassages` now return the page object with `.hits` and
  `.plan`), the langchain retrievers, the MCP `retrieve` tool, and
  `taguru route`'s shard merge move in lockstep in this release —
  upgrade router and shards together, and pin older SDKs to older
  servers (the pre-1.0 posture `GET /protocol` and this file already
  declare; older Python SDKs fail loudly on the new shape rather than
  mis-reading it). The plan rides inside the cached result bytes, so
  #150 exact hits and #153 semantic serves replay it unchanged — every
  event that could alter a plan (corpus write, vector publish, floor
  change) already moves the cache key.
- The approximate passage index now activates at 10 000 vector rows
  instead of 50 000 (#148). The old threshold sat above the default
  `TAGURU_PASSAGE_VECTOR_LIMIT` (20 000), so no default-config
  deployment could ever reach the index — every semantic sweep was
  the linear scan. Calibrated by benchmark (in the PR): at 10k rows
  the exact sweep costs 6–14 ms of read-fenced CPU per query against
  <1 ms via the index at ~100% measured recall@10, and the lazy
  one-time build (0.6–1.3 s) fits inside a request budget, which the
  7–15 s build at 50k did not reliably do. A compile-time assertion
  pins the threshold at or below the default vector limit, and boot
  says so once when an operator-lowered limit puts the index out of
  reach.
- Embedding provider calls are deadline-aware and stop-signal-aware:
  each attempt's HTTP timeout is the smaller of
  `TAGURU_EMBED_TIMEOUT_SECS` and the request's remaining budget (a
  slow provider is cut at the budget and that request's lane
  degrades, instead of holding the request past its own timeout and
  answering 408 after the work was done), and SIGTERM/SIGINT abandons
  in-flight provider waits, so a graceful drain no longer waits out
  the timeout ladder (up to 180s at the defaults — measured down to
  under a second). The deploy manifests resize accordingly:
  `terminationGracePeriodSeconds` / `stop_grace_period` drop from
  200(s) to 60(s), now sized by the request budget plus the final
  flush.

### Fixed
- A provider changing vector width behind an unchanged model name (a
  `dimensions` setting is a request-time parameter on Titan V2 and
  Matryoshka-style models) no longer produces silently empty semantic
  results in the window before the next refresh (#133). The refresh
  side already detected the change and re-embedded both stores
  wholesale; the serve side now refuses to score across the mismatch
  — every cosine would be `similarity`'s 0.0 width-mismatch sentinel
  — and names it instead: search plans and `sources/search/explain`
  report stored vs current width as a `ran: false` reason (the plan
  previously claimed `ran: true` over that all-zero sweep), and
  `resolve/explain`'s semantic report does the same where it
  previously presented the 0.0 sentinel as a measured cosine and
  prescribed lowering a floor no value could satisfy. `resolve`
  itself keeps folding to empty, exactly like a model change. The
  width-triggered wipe is now counted
  (`taguru_embedding_width_rebuilds_total{store="gloss"|"passages"}`)
  beside its existing warn line, and the Bedrock page's "pick a
  dimension once and never change it" instruction became "detected
  and rebuilt — still pick one width; rebuilds cost provider spend".
- A passage search whose query embedding was refused by the provider
  (a transient failure — the one vector-lane state that recovers with
  no revision bump) is no longer filled into the retrieval caches
  (#151): previously the degraded BM25-only page was cached like any
  other result and kept serving — and could canonicalize semantic-tier
  paraphrases onto itself — until the next unrelated corpus or config
  write, silently outliving the provider's recovery. The degraded page
  is still served (and its plan now confesses the failure); it is just
  never pinned. Stable skip states (embeddings off, nothing embedded,
  model changed) stay cacheable — config changes do move the key.
- Hydration against a LIVE lineage no longer mistakes the writer's own
  progress for rot: a replica (or stateless writer) booting while the
  writer keeps shipping could fetch an object the writer had just
  replaced — newer bytes than the manifest snapshot — and refuse to
  start with "downloaded bytes do not match the manifest". A
  verification mismatch now re-reads the generation's manifest and
  retries against whatever it currently says (a few paced rounds,
  every fetched shape: published files, sidecar meta, log lanes with
  reset series); true rot — bytes disagreeing with a manifest that is
  NOT moving — still refuses exactly as before. Missing bucket objects
  also keep their NotFound kind through the download wrapper, so a
  lane whose old series aged out heals the same way.

### Added
- Kustomize packaging for the Kubernetes manifests (#139):
  `deploy/kustomize/` serves `kubectl apply -k` over the reference
  manifests — a base (the single-writer PVC model) and overlays for
  the stateless, writer+replicas, and sharded-router variants; the
  router overlay is the two-shard fleet worked out (writer shards via
  nameSuffix + selector labels, the route-map as a content-hashed
  generated ConfigMap so a map edit rolls the routers on apply, and
  the front-door Deployment/Service). Kustomize over a Helm chart,
  with the rationale recorded in `deploy/kustomize/README.md`: the
  reference manifests stay the documentation (comments intact,
  consumed verbatim), `apply -k` needs no extra tool, and the retuned
  knobs are patches, not templates. `deploy/kustomize/verify.sh` —
  run by a new CI workflow on every PR touching `deploy/` — keeps the
  in-tree manifest copies byte-identical to the reference files,
  schema-validates every rendered configuration (kubeconform), and
  asserts the base renders equivalent to
  `kubectl apply -f kubernetes.yaml`.
- Router mode (#130): `taguru route` is a stateless scatter-gather
  router over sharded instances — `TAGURU_ROUTE_MAP` names a
  `context = shard-url` map file (optional `* = shard-url` fallback),
  context verbs proxy byte-for-byte to the owning shard, and the
  cross-context searches (`POST /recall`, `/query`,
  `/sources/search`), the directories, and groups span every shard
  with the exact single-instance merge semantics — `after` cursors
  forward verbatim (they anchor on the last match, not on a
  per-instance position), and equivalence with a single instance
  holding the same contexts is pinned by an integration test. Groups
  exist on every shard with member lists projected by the map;
  `/import` splits its batch stream by context and dry-run-preflights
  batch chunks and projected group records alike, so a stream one
  instance would refuse with nothing applied is refused the same way; `POST /mcp` works
  unchanged (bearer re-attached to each dispatched call). Auth is
  pass-through — the router holds no key store (setting
  `TAGURU_API_TOKEN(S)`/`TAGURU_KEY_SCOPES`/`TAGURU_PUBLIC_URL` on it
  refuses to boot), shards keep enforcing keys, scopes, and rate
  limits. Failure honesty: a shard that answers an error fails the
  request whole; an unREACHABLE shard degrades fan-out reads to
  labeled partials — the envelope gains an `unreached` array
  (response-shape note: new optional top-level field, absent when
  every shard answered) — and routed verbs answer the new
  `502 shard_unreachable` error code. Router-shaped `/metrics`
  (`taguru_router_*`). Moving a context is a documented runbook:
  quiesce → export → delete from the old shard through the router →
  map edit + rolling router restart → re-import through the router.
- Read replicas (#129): `serve --replica` / `TAGURU_REPLICA=1` serves
  the replication bucket's lineage read-only and keeps tailing it —
  issue #128's hydration running continuously. Every retrieval verb
  (resolve, describe, recall, query, activate, explore,
  sources/search, the listings and exports) serves from the replica's
  own hydrated copy, so reads scale horizontally with the pool; every
  mutating verb — raw HTTP and the MCP write tools alike — answers
  `403 read_only_replica` naming the writer (`TAGURU_WRITER_URL`,
  plus the bucket's fence holder), and neither SDK retries it. A
  replica never claims a generation and never ships; deletions and
  new contexts propagate; a new writer's generation is followed live,
  no restart. Consistency, stated honestly: per context at that
  context's applied watermark, cross-context skew possible, staleness
  bounded by shipping lag + `TAGURU_REPLICATE_INTERVAL_MS`; a bucket
  outage freezes the replica at its last watermark and it keeps
  serving. `deploy/kubernetes-replicas.yaml` is the read-pool
  manifest.
- Replica lag on `/metrics` (#129): `taguru_replica_applied_seq` vs
  `taguru_replica_shipped_seq` per context and lane,
  `taguru_replica_behind_seconds`, the followed generation, and the
  manifest/poll freshness timestamps — the promotion-time RPO on
  display, which is what the manual promotion runbook (now in the
  architecture docs, rehearsed end-to-end by an integration test)
  reads before flipping a replica's directory into the next writer.
  Availability with a replica pool is promotion time, not restore
  time; what the bucket never received — the deposed writer's
  un-shipped tail — remains the async-replication RPO.
- Boot from the bucket (#128): with `TAGURU_REPLICATE_URL` set, a
  server started on an **empty** data directory materializes itself
  from the bucket's newest complete generation instead of starting
  blank — the volume demotes to a cache of the bucket lineage, and
  recovery becomes "start anywhere". Hydration is lazy and
  priority-ordered: shared files (groups, the grant store, every
  context's sidecar meta) land before boot, pinned contexts hydrate in
  parallel before the port opens, and everything else hydrates on
  first touch or via a background fill; local files whose bytes
  already match the manifest are reused without a download, so warm
  restarts of a cache-mode volume stay cheap. The successor's own
  generation is not marked `complete` until every family has settled
  locally, so a restore can never land on a hollow lineage.
  `deploy/kubernetes-stateless.yaml` is the emptyDir variant this
  enables.
- The takeover guard (#128): starting a writer against a bucket IS the
  takeover/promotion act, so while the bucket's newest generation
  still looks alive — a heartbeat object refreshed every minute, no
  clean-shutdown marker, within a 300s grace — booting a different
  writer against it refuses unless the operator states the intent with
  `serve --take-over` / `TAGURU_TAKEOVER=1`. A cleanly stopped writer
  retires its generation on the way out and never trips the guard; a
  crashed one ages out of it. Ergonomics only: epoch fencing (#127)
  remains the sole arbiter, and a writer past the guard still deposes
  its predecessor cleanly and loudly. The deposed writer's un-shipped
  tail exists only on its own volume — a successor hydrating elsewhere
  serves the lineage without it.
- The `complete` marker now carries a manifest — every shipped file's
  and lane's exact extent (length + CRC-32C), refreshed after each
  batch of uploads. `taguru restore` verifies every downloaded object
  against it (a swapped or rotted object is a refusal, not a quiet
  divergence); pre-manifest buckets keep restoring through the listing
  fallback, unverified as before.
- Continuous replication to object storage (#127): set
  `TAGURU_REPLICATE_URL` (`s3://` / `gs://` / `az://` / `file://`,
  credentials via each cloud's default chain) and a background shipper
  continuously copies every context's complete file family to the
  bucket — both log lanes (the graph WAL and the passage log) tailed
  record-by-record with the same CRC-32C verification replay runs,
  published files (images, meta, sources, passage snapshots, derived
  sidecars, groups, the OAuth grant store, crash markers) whole on
  change. Durability becomes two honest tiers: a local crash still
  loses nothing (unchanged), and losing the machine or volume now
  costs at most the shipping lag — seconds, exported per lane as
  `taguru_replication_lag_records` / `_lag_seconds` in `/metrics`,
  beside upload/error counters and a last-success timestamp. Shipping
  polls; the acknowledge path gains no work, no latency, and no new
  failure modes (a dead bucket degrades replication only).
  `TAGURU_REPLICATE_INTERVAL_MS` (default 1000) is the cadence.
- `taguru restore --out DIR [URL]`: materializes a data directory from
  the bucket's newest complete generation — published files verbatim,
  each log lane reassembled from its shipped segments and re-verified
  record-by-record — refusing gapped segment runs and non-empty
  targets. Verify the result with `taguru inspect`; the derived
  sidecars ride along but remain rebuildable, so a restore tolerates
  their absence.
- Epoch fencing on the replication bucket (#127): each writer claims a
  monotonic generation with a conditional create and ships only into
  its own `gen-N/` namespace, so two live writers behind one URL (a
  botched restore, a doubled deployment) can never interleave one
  lineage. A deposed writer's shipper fail-stops permanently and
  loudly (`taguru_replication_fenced` latches, plus a `taguru::audit`
  line) while its serve path keeps answering from local truth.
  Deliberately no TTL, heartbeats, or automatic failover — the fence
  is lease-compatible (a permanent lease with TTL 0) for any future
  automation layer.

## [0.3.0] - 2026-07-18

### Added
- `TAGURU_EMBED_PARALLEL` (default 1, the prior sequential behavior):
  gloss refresh and passage refresh now dispatch each 128-item
  embedding chunk to the provider on up to `N` worker threads instead
  of one chunk at a time (#65). Both lanes persist whatever subset
  landed even when a later chunk fails, and under parallelism that
  subset is no longer necessarily a prefix of the original order — it's
  whatever completed before the first failure was recorded; the refresh
  still returns the error, so the rows it skipped stay stale for the
  next refresh to retry. Raise to match the provider's rate limit, not
  the machine's core count.
- `TAGURU_MAX_CONCURRENT_HEAVY_OPS` (default 2; `0` disables): one
  shared, non-queuing semaphore around `audit_vocabulary` and
  `compact_context`, over both raw HTTP and MCP dispatch. Once full,
  another heavy call is shed immediately as 503 `overloaded` with
  `Retry-After: 1`, leaving worker capacity available for ordinary
  requests while admitted sweeps run to their individual deadlines.
- Context groups: `/groups` bundles contexts (many-to-many) and may
  nest child groups — a shallow DAG, at most 3 groups tall, cycles
  refused — as the addressing unit cross-context retrieval will
  build on. `GET /groups` (keyset-paged directory), `PUT/GET/PATCH/
  DELETE /groups/{name}`; each row is `{name, description, contexts,
  groups}`, membership updates are deltas (`add_contexts`/
  `remove_contexts`, `add_groups`/`remove_groups`), and the same four
  operations ride MCP as `list_groups`/`create_group`/`update_group`/
  `delete_group`. A group bundles at most 1,000 member contexts and
  1,000 child groups: the delta lists were already capped per request,
  and the RESULT now is too (`over_limit`; removals apply first, so
  one request can trade members within the cap — past it, split into
  nested child groups). Referential integrity is strict: adding a
  member requires the context (`no_context`) or child group
  (`no_group`) to exist, deleting a context or a group drops it from
  every group, and boot reconciles any dangling member — or
  hand-written over-cap set, cycle, or over-deep chain — a crash or an
  edited data directory could leave behind. A group file that reads
  but does not parse keeps its name with empty content, its bytes set
  aside as `{name}.group.corrupt` and a fresh empty record written in
  their place; an UNREADABLE group file refuses the boot outright —
  registering it empty would let the next write silently overwrite
  membership that was never loaded. Nesting refusals answer
  `invalid_argument` (cycle) or
  `over_limit` (depth). Group reads/creates/updates need read/write;
  deletion is admin, like contexts. A context-scoped key sees every
  group row (child names included — labels, not content) but only the
  members its grant allows, and a group write touching any context
  beyond the grant — counted through nested children — is refused
  whole. Each group persists as one `{name}.group` JSON file beside
  the context files; one new error code, `no_group` (404). Groups
  ride export/import: a `taguru_group` record — one JSON line, the
  group's complete truth — travels the same stream batches do and
  restores AFTER every batch of a run as a create-or-replace of the
  whole record, so re-importing is idempotent and the files re-apply
  in any order. The set is validated whole (existence, caps, nesting;
  a child may be a name the same run brings) and a violation refuses
  every group record with the batches already durable; `POST /import`
  answers restored records under a new `groups: [...]` field (absent
  when the stream carried none — the old shape is untouched). A full
  `taguru export` writes each group as `{group}.group.jsonl`; a live
  server serves one at `GET /groups/{name}/export` (a context-scoped
  key exports its grant's slice, exactly the row it can read).
  `taguru inspect` verifies `.group` files too: unreadable or
  unparseable ones fail the check — a boot would refuse, or reset the
  record — and dangling references, over-cap sets, and ill-shaped
  nesting warn with exactly what boot's reconciliation would drop.
  Known limitations this iteration: `taguru compact` leaves group
  files alone (they hold nothing to compact), and a
  `DELETE /groups/{name}` whose unlink fails can resurface the group
  at the next restart (the error message says so).
- `taguru_groups_registered` gauge on `/metrics`.
- Cross-context search: `POST /recall`, `POST /query`, and
  `POST /sources/search` run one search across several contexts at
  once — `contexts: [full names]` beside the usual arguments. Every
  match carries the `context` it came from; recall/query merge on
  |weight| (weights share one scale — evidence mass) and passage hits
  interleave by per-context rank, since passage scores are
  corpus-local. The target list is vetted up front: a name beyond a
  scoped key's grant refuses the request whole (checked before
  existence, so grants cannot probe names), a missing name is
  `no_context`, an empty list is `invalid_argument`, and the list caps
  at the usual 1,000 items. The MCP search tools (`recall`, `query`,
  `search_passages`) take `contexts` as an alternative to `context`.
- The cross-context searches also take `groups: [group names]`, alone
  or beside `contexts`: each group searches every context it reaches,
  nested children included, and overlaps — with `contexts`, between
  groups, or through nesting — dedupe silently, so a context is
  searched once however many ways it was named. Directly named
  contexts lead the merge's tie order in request order; group-resolved
  members follow in name order. A name that is not a group is
  `no_group`, and the list shares the 1,000-item cap. For a scoped
  key, a group resolves to just the members the grant covers — the
  same slice group listings show it — rather than refusing, which
  would leak out-of-grant membership; directly named contexts keep the
  whole-request refusal. The MCP search tools take `groups` beside
  `contexts`.
- SDKs: groups and cross-context search on both clients. A `groups`
  resource (`client.groups`) mirrors `contexts` — `list`/`iter`/`get`/
  `exists`/`create`/`update` (deltas)/`delete`/`export` — and the
  top-level searches ride the client root: `client.recall(cue,
  contexts=…, groups=…)`, `client.query(…)`, `client.search_passages
  (…)` answer `CrossMatchPage`/`CrossPassageHit` rows, each match
  tagged with the `context` it came from. New models `GroupEntry`,
  `GroupPage`, `CrossAssociation`, `CrossMatchPage`, `CrossPassageHit`
  in both languages; surface parity is spec-checked as always.
- `TaguruRetriever` (both LangChain packages) now addresses `contexts`
  and `groups` beside the single `context` (at least one required —
  the field is no longer mandatory alone). Across several contexts the
  graph lane runs per context (concurrently, in the async clients) and
  interleaves by per-context rank, the
  text lane rides the server's cross-context search, and every
  Document's metadata gains a `context` key naming where it came from
  (single-context retrievers too — additive).

- Single-association retraction:
  `POST /contexts/{name}/associations/retract` `{subject, label,
  object}` withdraws ONE association outright — every source's
  contribution to that edge, unsourced weight included — where
  `sources/retract` withdraws a whole document's. The surgical
  correction for a fact that should never have been asserted; a fact
  that is merely contested still wants a negative-weight assertion,
  which preserves the dispute. Names resolve through aliases; the
  answer is `{retracted, attributions_removed}` with
  `retracted: false` for a triple naming no live edge (nothing
  changed, found-nothing honesty like `sources/retract`). The edge row
  stays visible at weight 0 / count 0 until compaction sheds it
  (`activate` already skips it), re-asserting the triple later just
  works, and the write is WAL-staged like every other (write role;
  one `taguru::audit` line, since the triple lives in the body). Rides
  MCP as `retract_association` and the SDKs as
  `Context.retract_association(subject, label, object)` →
  `RetractAssociationOutcome`. On-disk note: the WAL grew a
  `retract_association` op — a binary predating it refuses a log
  holding one as corruption (the documented forward-only WAL posture).
- `GET /protocol` documents the correction split: retract what should
  never have been asserted, contest with negative weight what the
  world disputes.
- MCP parity for the backup verbs: `flush` (admin, answers the flushed
  names), `export_context` (the import batch stream as one text
  block), and `export_group` (the group's `taguru_group` line) ride
  the tool surface, mapping onto `POST /flush` and the export routes
  with their roles intact — an agent can run the documented
  flush-then-export discipline without leaving MCP. Very large
  contexts should still export over plain HTTP or `taguru export`
  offline; the tool descriptions say so.

- Integrity checksums in every on-disk format that holds acknowledged
  data (#59). The context image gains a whole-file CRC-32C footer
  (format v5 → v6), verified before anything else is trusted on load;
  the passage snapshot does the same (`TAGURUS3` → `TAGURUS4`); and
  every WAL record — graph and passages — now carries a `crc` field
  verified on replay. Structural validation alone accepts silent
  corruption that happens to keep the invariants (a flipped byte
  inside a stored name loads, serves, and flushes back as truth);
  the checksums close exactly that gap, and `taguru inspect` now says
  what was *verified* versus merely parsed (image/snapshot generation
  in each ok-line, a NOTE counting pre-checksum WAL records).
  On-disk notes: older images and snapshots keep loading forever,
  unverified, and writing always produces the checksummed formats — so
  after the first flush a DOWNGRADED binary refuses the image as an
  unsupported version (roll back onto a pre-upgrade backup, or through
  export/import). The WAL change is additive in both directions: a
  pre-checksum binary ignores the field, and pre-checksum records
  replay unchecked.
- Torn-import detection (#59): one import batch applies as four
  separately durable steps (retract the source → store the passage →
  add associations → add aliases), each store individually consistent,
  so a crash — or an unrepaired mid-batch refusal — used to leave the
  source half-applied with nothing able to say so. Now a per-source
  batch-open marker (`{stem}.{source-hash}.importing`, the pair named
  in its content) is written before the first step and removed only
  after the last: the server's next boot warns for every surviving
  marker whose context still exists (and removes moot ones), and
  `taguru inspect` reports the same tear with its repair. Both
  documented repairs clear it — re-importing the batch file (offline
  or `POST /import`; retract-then-apply keeps the retry exact) or
  retracting the source. Deleting or recreating the context sweeps its
  markers. Cross-store atomicity is deliberately not attempted:
  per-source idempotency already makes the repair exact, so detection
  was the whole remaining gap.
- Search explainability (#75): every retrieval lane can now say why an
  expected result did not appear, in one read-only call instead of
  orchestrating four endpoints with varied thresholds by hand.
  `POST /contexts/{name}/sources/search/explain` takes `{query, source,
  paragraph?, limit?}` and answers the first verdict that applies —
  `not_stored` (never stored here, or retracted; the store keeps no
  tombstone history to tell which), `paragraph_out_of_range`,
  `no_query_terms`, `no_term_overlap` (both sides' terms rendered AS
  STRINGS, so a 酒蔵-vs-酒造 spelling fork is visible on the table),
  `below_cutoff` (the actual rank, the cutoff score at the requested
  limit, and a `limit_to_reach` VERIFIED by rerunning the real serve
  computation, pool caps included), or `served` — with per-term BM25
  evidence (tf, df, idf, contribution: bit-for-bit the addends search
  summed) and the vector lane's cosine, or the named reason that lane
  never ran (off, no provider, query embedding failed, nothing
  embedded, model changed). `POST /contexts/{name}/resolve/explain`
  and `resolve_label/explain` take `{cue, expected}` plus the same
  one-call overrides resolve honors, and answer `not_in_vocabulary`
  (nearest stored spellings attached, lexical and semantic — the
  register-an-alias repair is one step away), `cue_resolved_exactly`
  (the cue IS another stored spelling; the exact tier answers alone,
  which no floor tweak can fix), `below_floor` (the actual Dice score
  vs the floor in effect — only the fuzzy tier is floor-gated, and
  the verdict honors that), `below_cutoff`, `semantic_not_run` /
  `semantic_below_floor` (whether the fallback tier joined, its gloss
  cosine vs the semantic floor, or which precondition failed), or
  `served`. All three ride MCP as `explain_search` /
  `explain_resolve` / `explain_resolve_label`. No new persistence, no
  new counters; explain shares the live scoring code paths (one term
  walker, one BM25 addend, one fusion/trim), so it cannot disagree
  with the search it explains.
- Match pagination past the 1,000-row cap (#60): `recall`/`query`/
  `unreachable_from` (single- and cross-context) and `explore` used to
  hard-truncate at `limit` (max 1000) with no way to reach whatever
  sat past it — a corpus with 5,000 matches for a cue permanently hid
  4,000 of them from every response. Each now accepts `after`, a
  keyset cursor copied verbatim from the last row of the previous
  page: `{weight, subject, label, object}` for `recall`/`query`/
  `unreachable_from`, the same plus `context` for their cross-context
  forms `POST /recall`/`POST /query` (two different target contexts
  can independently hold an edge at the identical triple, so `context`
  is the tiebreak they can't share on their own), and `{distance,
  subject, label, object}` for `explore`. `total` stays constant
  across pages — it's the population before the cursor and before
  truncation, the same convention `aliases`/`labels`/`/contexts`
  already use — so a client pages until `matches` comes back empty,
  never until `total` changes. The server never mints an opaque
  cursor; the client always derives the next `after` from the last
  item of the page it just received. Rides MCP (`recall`/`query`/
  `explore`/`audit_coverage`) and both SDKs as `MatchCursor`/
  `CrossMatchCursor`/`ExploreCursor`. Wire-visible ordering note:
  these endpoints now always sort their results (by weight or hop
  distance, then lexicographically on `(subject, label, object)` as
  the tiebreak) instead of only sorting when truncation kicked in —
  keyset pagination requires one deterministic order on every page
  whether or not a cursor is present, so a caller relying on the old
  insertion-order tiebreak under the limit will see a different order
  now.
- List filters (#62): `GET /contexts` takes `pinned: bool`;
  `GET /contexts/{name}/sources`, `/labels`, and `/aliases` take
  `prefix: string` — all filtered before `total` is computed,
  consistent with the search endpoints' "total describes the filtered
  population" contract. Chasing this down uncovered a real MCP bug:
  `query_string` silently dropped any boolean argument (`Value::Bool`
  had no case), so `pinned` on `create_context`/`update_context` was
  unreachable over MCP until now.
- MCP passthrough (#62): `compact` (admin), `get_context`, and
  `get_group` — mirroring their HTTP routes exactly — plus a new
  `import` tool (admin) that takes a `stream` (NDJSON, capped at
  32MiB by a constant local to `mcp.rs`, since `taguru-mcp` does not
  link `ingest.rs`) and an optional `dry_run`. Wiring up `stream` also
  fixed a latent MCP transport bug: a tool argument that is already a
  plain string was still being JSON-string-encoded before it reached
  the HTTP body, which escaped the NDJSON's newlines and broke the
  parser.
- `POST /import?dry_run=true` (#62): previews an import batch —
  creates, retractions, dropped passages/questions/associations —
  without writing anything, reusing the real apply path's counting
  logic wherever that is already read-only. Association/alias counts
  are the one place dry-run is necessarily optimistic: a capacity or
  version conflict cannot surface without actually writing, which the
  response and the tool description both say plainly. `taguru_group`
  records go through a separate restore path and are skipped in
  dry-run mode (omitted from the response's `groups` field).
- `retrieve` MCP tool (#62): runs the SDKs' resolve → describe →
  query/activate → cite_passage → search_passages walk server-side in
  one call, so an MCP-only client gets the same one-shot retrieval the
  Python/TypeScript SDKs already had over HTTP. Citations come back as
  `[{source, paragraph, citation}]`, since a JSON object cannot key on
  a `(source, paragraph)` tuple the way the SDKs' in-memory dicts do.
- Rename (#62): `POST /contexts/{name}/rename` and
  `POST /groups/{name}/rename` (admin role, body `{"to": "..."}`),
  plus `rename_context`/`rename_group` on both SDKs and as MCP tools.
  A context rename moves its whole file family and rewrites every
  group naming it; a group rename moves its one file and rewrites
  every OTHER group naming it as a child. Both are crash-safe the same
  way delete is: a durable marker (`.renaming` / `.grouprenaming`) is
  written before anything moves, and a boot that finds one resumes the
  file move AND the group-membership rewrite before the usual
  dangling-reference reconciliation runs — the ordering matters, since
  reconciliation has no notion of a rename in flight and would
  otherwise prune the old name as dangling instead of carrying it to
  the new one. A context rename's `from`/`to` are reserved against a
  concurrent create or another rename; a group rename runs entirely
  under the group table's single write lock instead, so no extra
  reservation is needed there.
- Compaction dead-weight visibility (#60): three counters — live dead
  edges (count fallen to zero), attributions unlinked from every chain
  but not yet reclaimed, and arena slack (bytes behind removed aliases)
  — answer "how much would `compact` reclaim right now" without
  actually running it. Tracked incrementally (retraction, alias
  removal) and seeded once at load by piggybacking on the existing
  attribution/name-table walks, so there is no extra full walk and a
  freshly compacted context starts all three back at zero. Exposed via
  `GET /contexts/{name}` and the directory (new `ContextStats` fields),
  `/metrics` (`taguru_dead_edges`, `taguru_dead_attributions`,
  `taguru_arena_slack_bytes` — server-wide sums; a context name is
  unbounded user data, so no per-context label), `taguru inspect`'s
  stats line (the same three plus the dead ratio), and `taguru
  estimate` (a new "maintenance window" line pricing a compaction's
  transient double footprint).
- `POST /maintenance/compact`: the operational counterpart to the
  visibility above — closes the server to ordinary traffic just long
  enough to drain in-flight requests and rebuild every context whose
  live dead ratio clears an optional `min_dead_ratio` (default: any
  dead weight at all), worst ratio first, then reopens. `/health`
  answers 503 `maintenance` for the duration (distinct from an actual
  fault) and new work is shed early; only one sweep runs at a time; a
  second call while one is running answers 409 rather than queuing.
  Server-wide like `/flush`: a context-scoped key is refused outright.
  Admin-only, via the existing catch-all default rather than a new
  authorization rule.
- `taguru compact --parallel N`: compacts up to `N` contexts at once
  (default 1, the prior sequential behavior), reusing the worker-pool
  pattern boot-time preload already uses. Output is reordered to the
  original argument order before printing, so stdout is byte-for-byte
  identical to `--parallel 1` regardless of `N` or thread scheduling.
- `taguru estimate` now prices passage-related memory — the passage
  store, the BM25 index, and (with `--embedding-dims`) passage vectors
  — into its `TAGURU_CACHE_BYTES` budget; before, `--passage-bytes`
  only ever showed up in the disk section. The paragraph count the
  vector estimate multiplies is capped at the same
  `DEFAULT_PASSAGE_VECTOR_LIMIT` the server enforces per context.
- `taguru extract --parallel N` (or `TAGURU_EXTRACT_PARALLEL`, the flag
  wins when both are set): runs up to `N` of one document's chunk
  completions concurrently instead of one at a time (default 1, the
  prior sequential behavior) (#64). Chunks still merge in their
  original index order, so output is byte-for-byte identical to
  `--parallel 1` regardless of `N` or thread scheduling — only
  wall-clock changes. The first chunk to fail, by index rather than by
  which thread finishes first, still fails the whole document: no
  worker claims a new chunk past the failure once it is recorded, but
  a chunk already claimed and in flight at that moment still runs to
  completion — its result is simply discarded. Parallelism never
  crosses documents — each document's
  relation-label vocabulary feeds the next document's prompt, so
  documents themselves keep extracting one at a time.
- Drift audit (#63): `unsourced weight` — an association's weight left
  over once every named source's contribution is subtracted, the same
  bucket export/re-import round trips already tag with the reserved
  source id `export:unsourced` — now surfaces at three layers instead
  of only inside exported batches. `GET /contexts/{name}` and `taguru
  inspect`'s stats line add `unsourced_edges`/`unsourced_weight`;
  `/metrics` adds `taguru_unsourced_edges`/`taguru_unsourced_weight`
  gauges (server-wide sums, same reasoning as `taguru_dead_edges`).
  `POST /contexts/{name}/drift/audit` (MCP `audit_drift`, Role::Read)
  is the new read-only verb, answering three things at once: edges
  whose unsourced weight clears an optional `unsourced_floor` (default:
  any at all), worst-first and cursor-paginated like
  `unreachable_from`; aliases whose canonical concept or label has
  gone dead (zero live edges); and, opt-in via `include_twins`, the
  same lexical/semantic fork candidates `vocabulary/audit` already
  finds, at the same `dice_floor`/`cosine_floor` defaults (0.6/0.6) —
  one shared implementation, not a second copy. `taguru estimate` is
  unaffected: synthesized associations always carry a generated
  source, so unsourced weight cannot arise there.
- Registering an alias now leaves its own `"aliases registered"`
  `taguru::audit` line (context, concept/label counts, applied count —
  never the spellings themselves), symmetric with the existing one for
  removal. Reconstructing a bad alias's live window — see the new
  "Recovering from a bad alias" note under Running in production — no
  longer depends on the removal side alone.

### Changed
- doc2query `questions` now index into their paragraph's BM25 postings
  (terms and length both — the doc2query move itself), so a
  question-shaped search lands lexically on every server; before, a
  deployment without `TAGURU_EMBED_PASSAGES` stored questions and
  ignored them for retrieval. Passage scores shift only where
  questions are attached. On-disk note: the BM25 sidecar format bumped
  (`TAGURUB1` → `TAGURUB2`, slots carry a question fold for the drift
  digest) — a derived structure, so an old sidecar rebuilds itself on
  the residency's first search, in either upgrade direction; no
  action needed.
- `aliases`/`labels`/`/contexts`'s directory paging no longer
  re-collects and re-sorts the entire namespace on every page request
  (O(n·log n) per page, O(n²·log n) to walk the whole thing) (#60):
  each context now keeps a `BTreeMap`/`BTreeSet` index alongside its
  existing storage, and the server registry does the same for
  `/contexts`, so a page is a true keyset seek — O(log n + k),
  independent of table size, for an unscoped key. A context-scoped
  key's allow-list has no relation to name order, so `/contexts`
  still sorts that (typically small, operator-configured) allow-list
  per request rather than seeking the registry — `aliases`/`labels`
  are unaffected, since they page one context's own namespace, not
  the registry. Cross-context search (`POST /recall`/
  `POST /query`, and cross-context `sources/search`) also no longer
  fans out to its target contexts with a sequential `for` loop: every
  target is now fetched concurrently, bounded by
  `TAGURU_CROSS_SEARCH_CONCURRENCY` (default 4), so one slow or cold
  context no longer blocks every context listed after it. Results and
  `total` are unchanged in both cases — only the cost/wall-clock
  improves.
- Passage vector search grows an approximate nearest-neighbor index
  past 50,000 rows in one context (#60): a hand-rolled IVF index
  (deterministic farthest-point clustering — no RNG, no external ANN
  crate, matching every other binary index in this codebase), built
  lazily on the first search past the threshold and cached for the
  store's lifetime. Below the threshold, and for any call asking for
  every row (`explain_passage_search`'s exact-ranking contract, and any
  deadline too tight to build the index), the full linear sweep still
  runs unchanged — approximation is strictly an optimization here,
  never a behavior callers need to account for.
- Three more scalability improvements (#60): the in-memory cue-
  embedding cache now evicts least-recently-used instead of FIFO;
  boot-time directory scanning parallelizes each context's disk I/O
  (sidecar read plus WAL stats) across a worker pool instead of a
  sequential loop, with results merged into the same sorted map
  regardless of arrival order; and the passage log size `/metrics`
  reports for a cold context is now cached at eviction time instead of
  re-`stat`ed on every scrape.
- `taguru extract`'s retry policy replaces the previous fixed-2-second
  sleep, 2-attempt retry with exponential backoff and full jitter (1s
  base, doubling toward a 30s ceiling; 4 attempts total — 1 initial
  plus 3 retries) (#64). A 429 response's `Retry-After` header, when
  present as delta-seconds (HTTP-date values are not recognized), is
  honored verbatim (clamped to the same 30s ceiling) instead of the
  computed backoff, since the server's own instruction beats a guess;
  other statuses are unaffected. A non-retryable 4xx
  still fails immediately, spending none of the retry budget. The
  final error message now reports how many attempts were made.

### Fixed
- `estimate`'s synthesis walked labels by round (`round % labels`), so
  the number of distinct labels actually materialized in the measured
  context was capped at the round count (`associations / concepts`),
  not at the requested `--labels`. The default shape
  (`concepts = associations / 2`) has exactly 2 rounds, so every
  default-shape run measured a context holding only 2 labels while the
  header printed the planned 50 — and an explicit `--labels N` for a
  label-rich workload was silently capped the same way, with no
  warning. The label index is now offset by the subject index
  (`(round + subject_index) % labels`), so every requested label
  appears from round 0 whenever `concepts >= labels` (every realistic
  shape, including the default); a new warning also fires on the
  residual case where `labels > concepts` and rounds are too few to
  cover them all.
- Gloss embedding refresh now prunes the vectors of concepts and labels
  the graph no longer holds. `refresh_embeddings` extended the loaded
  vector store in place, so a name dropped by compaction kept its row
  in the `.vectors.bin` sidecar forever, and `resolve`, the vocabulary
  audit's twin suggestions, and `explain` kept surfacing it. The
  sidecar is now rebuilt against the live concept and label names each
  refresh, the way the passage refresh already was.

## [0.2.0] - 2026-07-12

### Added
- Machine-readable error codes: every JSON error now carries a stable
  `code` beside the human `error` text —
  `{"status": "error", "code": "<kind>", "error": "...", "time": ...}`.
  The vocabulary (documented in `GET /protocol`): `malformed_request`,
  `invalid_argument`, `over_limit`, `unauthorized`, `forbidden`,
  `no_context`, `no_source`, `no_paragraph`, `unknown_path`,
  `method_not_allowed`, `timeout`, `already_exists`, `conflict`,
  `payload_too_large`, `rate_limited`, `internal`,
  `embeddings_unconfigured`, `embeddings_failed`, `overloaded`,
  `unhealthy`, `storage_full`. Branch on the code (or the status),
  never on message wording. The SDKs surface it as `.code` on every
  error.
- Client SDKs under `sdk/`: `taguru` for Python (sync + async, httpx)
  and TypeScript/JavaScript (fetch, zero dependencies) with one
  identical surface — typed models for every endpoint,
  idempotency-aware retry (`add_associations` never retries after an
  ambiguous transport failure; 429/503 always retry), keyset
  auto-pagination, chunked batch writes, export/import helpers, and a
  `retrieve()` implementation of the protocol's retrieval loop.
  Cross-language parity is machine-checked against
  `sdk/spec/surface.yaml` in CI; integration suites spawn the real
  server binary. Plus `langchain-taguru` for both ecosystems:
  `TaguruRetriever` (graph lane + text lane, RRF-merged, verbatim
  citations) and `TaguruIngester` (the LangChain twin of
  `taguru extract` — same prompt discipline, same merge validation,
  applied via `POST /import`'s per-source replace). The packages
  version in lockstep with the server: each `v*` release tag
  publishes all four to PyPI and npm alongside the crate.
- SDK use-case examples under `examples/langchain/` — RAG QA with
  citations, governed document ingestion (dry-run review → apply →
  per-source replace → retract), and conversational long-term memory
  with correction by negative weight — one directory per use case,
  each as a Python and a TypeScript program mirrored line for line.
  All run offline (a real server binary is spawned per run;
  deterministic fake chat models stand in for the LLM) and the SDK CI
  workflow executes every one of them.
- `taguru export` and `GET /contexts/{name}/export`: every context
  renders as the same JSONL batch stream `taguru import` and
  `POST /import` apply — the portable, version-independent backup.
  Both import entrances now read multi-batch streams (each
  `taguru_batch` header opens the next batch); a multi-batch
  `POST /import` answers `{batches: [...]}` per batch.
- `taguru compact` and `POST /contexts/{name}/compact` (admin):
  rebuild a context's image from live content alone, shedding
  retracted edges, unlinked attributions, and arena slack. Content,
  counts, and paragraph locators survive; the outcome reports the
  shrink.
- `TAGURU_KEY_SCOPES`: per-key authorization — roles
  read ⊂ write ⊂ admin plus optional per-context grants, enforced
  identically over raw HTTP and MCP tool dispatch. Keys the variable
  does not name keep the historical full grant.
- `TAGURU_MAX_CONCURRENT_REQUESTS` (default 256): an in-flight
  ceiling that sheds excess load with 503 + Retry-After before auth
  runs; probes stay exempt. New `taguru_inflight_requests` gauge and
  `taguru_requests_shed_total` counter.
- `GET /live`: unconditional liveness probe. `/health` keeps the
  readiness (write-path) signal; wire orchestrator probes
  accordingly.
- Audit trail: the access log now names the context each request
  addressed, and destructive operations (context delete, source
  retract, alias removal, import batches, compaction) each leave one
  self-contained `taguru::audit` event naming the key and the object.
- Embedding resilience: transient provider failures (transport, 429,
  5xx) retry twice with backoff; `TAGURU_EMBED_TIMEOUT_SECS` makes
  the per-attempt ceiling a knob; a new
  `taguru_embedding_duration_seconds` histogram times every round
  trip; boot warns when the request timeout sits under the provider
  ceiling.
- Load quarantine: a context (or passage store) whose load keeps
  failing answers its remembered refusal for 30s instead of
  re-reading the broken files on every request; restoring the files
  heals it on the next retry.
- Pinned contexts preload in parallel at boot.
- Deployment examples under `deploy/` (Kubernetes, docker-compose)
  matching the documented single-writer model.
- `examples/http_benchmark`: concurrent load against a running server
  — throughput and p50/p95/p99 per phase (seed writes, reads, a 90/10
  mix), the capacity-planning companion to the library benchmark.
- The protocol doc states the compatibility policy: no `/v1` (the
  protocol travels with the server), additive responses parsed
  tolerantly, pre-1.0 shape changes announced here.
- Documentation site at <https://t0k0sh1.github.io/taguru/>: getting
  started, concepts, the import/extract references, per-platform
  deployment guides (Docker Compose, Kubernetes, Amazon Bedrock), the
  internal architecture, and a captured MCP retrieval walkthrough. The
  README slims down to a user-facing overview that points there.

### Fixed
- The protocol document and README now list `/live` among the
  auth-exempt probes — the code always exempted it alongside
  `/health` and `/metrics`; only the docs omitted it.
- A failed `DELETE /contexts/{name}` unlink could leak the context's
  sidecar files forever — or, if `.ctx` itself survived, resurrect
  the context at the next boot. Deletion now writes a durable
  `.deleted` marker first, boot resumes any deletion it finds a marker
  for, and recreating a context clears a stale marker so a failed
  delete followed by a same-name create cannot be undone at the next
  boot.
- Export is now a true fixed point: a context with sourceless weight
  exports a reserved `export:unsourced` batch, and re-exporting the
  restored context (which carries a real attribution to that id) folds
  it back instead of refusing — the round trip export exists for no
  longer breaks on its own output.
- `taguru export` writes each stream atomically (stage + fsync +
  rename), so a crash while refreshing a backup no longer truncates
  the previous good copy.
- `/flush` refuses a context-scoped key (it is server-wide and names
  every flushed context); authorization now wraps the `/mcp` and OAuth
  routes it previously missed; and `@` in a key name is refused at
  boot (it collided with the OAuth-delegation scope fallback).
- A context compaction racing a background flush's stage-then-publish
  window could have the flush win the race and republish
  pre-compaction bytes over the compacted image, silently reverting
  the associations the compaction had just discarded. A per-entry
  generation counter now detects a Hot-to-Hot swap mid-flush and backs
  the stale republish off instead.
- `POST /import`'s multi-batch apply loop, `create`/`update` (a pin
  toggle can also load a context from disk) and `delete` on
  `/contexts/{name}`, and passage lookup/citation/listing on a
  context's first cold load all ran their synchronous, fsync-bearing
  I/O directly on the async runtime with no `block_in_place` — a large
  import alone could stall it for seconds, delaying every other
  in-flight request. All now wrap their blocking calls, matching the
  rest of the write and passage-search paths.
- A non-numeric component in an embedding provider's response silently
  became `0.0` instead of refusing the response like every other
  malformed shape (missing vector, wrong width, bad index) — a
  corrupted vector could then rank as a plausible neighbor in
  similarity search. It now refuses and names the offending index.
- `activate`'s decay and every `dice_floor` entry point
  (`resolve_with_floor`, `set_dice_floor`,
  `similar_concepts`/`similar_labels`) clamped into `[0, 1]` with a
  bare `.clamp`, which passes a NaN input straight through instead of
  clamping it — flipping some fail-closed filters open. Each now maps
  NaN onto the safe extreme instead.

### Changed
- **Response shape** (pre-1.0 break): `POST /import` now answers
  `{batches: [...]}` for a single-batch body too (was: that batch's
  bare outcome) — one shape for every import, no client-side
  branching on stream length.
- A request body over `TAGURU_MAX_BODY_BYTES` now answers 413 in the
  same JSON error shape as every other axis (was: axum's plain-text
  rejection).
- `add_associations`' partial-write arm keeps the capacity/conflict
  status split (507 vs 409) every other batch write reports —
  previously it answered 507 unconditionally. Unobservable today
  (association writes only fail on capacity), pinned for uniformity.
- **Response shapes** (pre-1.0 break): `GET /contexts/{name}/labels`,
  `.../aliases`, and `.../sources` now page like the directory —
  `?limit=1000&after=...` in, `{total, ...}` out. The alias cursor
  spans both namespaces (`after=concept:<alias>` or `label:<alias>`).
  MCP tool schemas carry the same parameters.
- Embedding failures no longer embed the provider URL in client-facing
  502 bodies; messages name the status code or transport error kind.
- Boot warns when listening beyond loopback with the per-key rate
  limit off.

### Security
- The OAuth grant store (`oauth.json`) is created owner-only (0600) at
  open time — born with the mode, not chmod'd after, so no readable
  window exists between create and the secret write.
- The OAuth consent page carries `X-Frame-Options: DENY`, a
  locked-down `Content-Security-Policy`, and
  `Referrer-Policy: no-referrer`.
- Dynamic client registration accepted a `redirect_uri` by
  string-prefix-matching `"https://"`, so
  `https://trusted-app.example.com@evil.attacker.com/callback`
  registered without error — the host an approved code actually
  reaches is the attacker's domain after the `@`, not the
  trusted-looking name before it. Registration now parses the URI
  structurally and refuses any userinfo component outright.

## [0.1.0] - 2026-07-05

Initial release: the association-graph library (flat-buffer images,
WAL-backed durability), the HTTP server (auth, rate limits, metrics,
OTLP tracing, OAuth for remote MCP), the MCP stdio bridge, and the
offline tooling (`import`, `extract`, `inspect`, `estimate`).
Published to crates.io and GHCR.

[Unreleased]: https://github.com/t0k0sh1/taguru/compare/v0.9.5...HEAD
[0.9.5]: https://github.com/t0k0sh1/taguru/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/t0k0sh1/taguru/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/t0k0sh1/taguru/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/t0k0sh1/taguru/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/t0k0sh1/taguru/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/t0k0sh1/taguru/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/t0k0sh1/taguru/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/t0k0sh1/taguru/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/t0k0sh1/taguru/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/t0k0sh1/taguru/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/t0k0sh1/taguru/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/t0k0sh1/taguru/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/t0k0sh1/taguru/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/t0k0sh1/taguru/releases/tag/v0.1.0
