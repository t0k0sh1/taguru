# 0031. A recorded completion can stand in for a live model call

- **Status**: Accepted
- **Date**: 2026-08-26
- **Issue**: #815
- **Related**: ADR 0030 (the step vocabulary this ADR resumes into —
  `--resume-from`'s names come from there), ADR 0023 (`run_id`,
  `attempt_seq`, `piece_id` — the identifiers this ADR's matching and
  bookkeeping are built on, never duplicated), ADR 0025 (the attempts
  log this ADR reads and appends to), ADR 0021 (the `auto`
  structured-output probe — deliberately not replayed, §3.1), ADR
  0001 §10 (the diagnostics sidecar's metadata-only ruling — this ADR
  does not touch the sidecar), #179 (checkpoints — why replay reads
  neither them nor the manifest skip, §3.5), ADR 0003 §2.2 (the
  manifest's computation-input reasoning — this ADR rules replay is
  not one, §3.5), #781 (the parent issue; ADR 0030 is this ADR's
  sibling, not its prerequisite in implementation order, though both
  gate #781's later children), #782 (context generation adds a
  model-called step before `steer`, §3.6 — the vocabulary problem this
  ADR solves for `steer` recurs there)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

Whether and how `taguru extract` can satisfy a completion from the
attempts log (ADR 0025) instead of calling the model: where the seam
sits, what identifies "the same request", what happens when nothing
matches, how the log stays safe to read while a run also writes to
it, how this interacts with the manifest and checkpoints, how the
cross-document label vocabulary is handled, and the shape of the CLI
surface. Out of scope: the actual code (#781's later children,
#816–#822), the step names replay resumes into (ADR 0030), and the
exact prose of the `docs/extract.html` sections this decision requires
(written alongside the implementation that ships them).

## 2. Context

ADR 0023–0029 (#784) made every prompt and every answer durable —
`--out/.extract-trace/<batch stem>.attempts.jsonl` holds every
completion's full conversation and full answer, on by default, kept
even for a failed document. But ADR 0023 §1 was explicit that this is
an observation: *"persistence for resume ... the trace is an
observation, never something a rerun reads"* — and named this issue as
where resume would be decided.

#781's second and third payoffs — redoing only what changed downstream
of a completion, and moving the expensive step to a different machine
from the cheap ones — both need that observation to become an input.
The user's direction (2026-08-26): the mechanism is *replaying the
attempts log*, not a new storage layer — a request whose input changed
should fall through to a live call on its own, exactly because the log
already is content-addressed by what was actually sent.

Two problems make this harder than "diff the log": the attempts log's
system turn is recorded once per document and referenced by hash from
every `attempt` record (ADR 0025 §3.3), and the system prompt itself
is not a pure function of the document — `system_prompt`
(`src/extract/prompt.rs:42`) also depends on `Run.vocabulary`, a label
tally that accumulates **across documents in run order**
(`src/extract/run.rs:670-672`, folded back on a manifest skip by
`absorb_vocabulary`, `run.rs:1224`). Changing one document's mechanical
validation rule can change its label counts, which changes the next
document's system prompt, which changes every completion after it. A
naive "match the whole conversation" replay makes exactly the kind of
run #781 is for — "change one rule, keep everything else" — degrade
into "the second document onward calls the model anyway."

## 3. Decision

### 3.1 The seam is a new value, `Completions` — not inside `ChatClient::complete`

`src/extract/chat_client.rs:16-51,227-242` currently puts `run_id`
(minted once per process) and `attempts` (an `AtomicU64` counter for
`next_attempt()`) on `ChatClient` itself, and takes a completion's
number at exactly three call sites, immediately before
`ChatClient::complete` — `src/extract/chunking.rs:137` (the legacy
path), `chunking.rs:696` (the ladder's `extract_round`), and
`run.rs:927` (Stage 2 cross-chunk correction). ADR 0021's `auto` probe
and `taguru communities`/`consolidation` call `complete` directly and
take no number — they are not extraction completions.

Replay reuses that exact split. A new value, `Completions`, holds
`run_id`/`attempts`/`next_attempt()` (moved off `ChatClient`) plus
`client: Option<&ChatClient>` and `replay: Option<&ReplayIndex>`
(§3.2). Its `complete()` tries the index first; on a hit it
reconstructs the response from the record with no HTTP call; on a
miss it falls through to `client` (or fails, under `--replay strict`,
§3.3). The three call sites above take a `&Completions` instead of a
`&ChatClient`; `ChatClient::complete` itself is untouched, so the
probe and `communities`/`consolidation` are unaffected without being
told anything about replay.

A trait (`CompletionSource` with a live and a replay implementation)
was considered and rejected: replay and live are not two
interchangeable implementations of the same thing, they are a
fallback chain (replay, then live) — the miss-falls-to-live behavior
*is* the design, and a trait would scatter it into whichever
implementation happens to run second instead of keeping it in one
`complete()` body.

### 3.2 The matching key: a normalized conversation digest, not an identifier

The key a request is looked up by — each field length-prefixed (a
fixed-width byte count ahead of its bytes) rather than delimited, so
no content, including a stray NUL byte, can ever be mistaken for a
boundary between fields or between messages:

```text
sha256( for each message in order:
          len(role) || role || len(field) || field )
  ++ requested_max_tokens

where field = system_sha256 when role == "system", else content
```

The system turn is hashed by reference to ADR 0025's own `system_sha256`
(rather than requiring today's `system_prompt()` output to be hashed
and compared byte-for-byte first) — this is what makes §3.6's pin
decidable without inventing a second notion of "the same system
prompt." `requested_max_tokens` is part of the key, not the
conversation, because ADR 0019's escalated resend repeats the exact
same messages at a larger cap (`chunking.rs:508-528`) — folding it
into the conversation hash would make the base attempt and the
escalated resend collide.

**Rejected: keying by `(piece_id, stage, attempt)`.** These identify
*a slot in the ladder*, not *what was asked*. A different validation
rule, a different vocabulary offer, or a different `--candidates`
setting all leave the slot's coordinates unchanged while changing what
should have been sent — keying on them would silently hand a stale
answer to a new question. The whole point of the user's chosen
mechanism is that a changed input falls through to a live call **on
its own**; identifier-keying defeats that by construction.

Multiple records can share one key — an ADR 0021 demote restarts the
same piece at the same messages and cap after a failure. These are
resolved as a **FIFO queue, per key, in the file's own line order**:
not sorted by `(run_id, attempt_seq)`, because one attempts log can
span several runs (a resumed document appends, ADR 0025 §3.2) and line
order is the only thing that still reflects real chronology across
that boundary. A corrective round, an escalated resend, and a split
sub-piece all naturally get distinct keys (their messages or
`requested_max_tokens` differ), so the FIFO queue in practice only
ever holds more than one entry for the demote-and-restart case.

The ladder's `rung` is deliberately **not** part of the key: under
`--structured-output auto` with `--parallel > 1`, which rung a piece
demotes to interleaves non-deterministically across pieces run in the
same process, so keying on it would make replay non-deterministically
miss. `rung` is instead added to the `attempt` record as a plain field
(#781's later children) so an operator pinning `--structured-output`
for a replay run can read what rung the record was taken under.

**Diagnosability**: the index is organized `piece_id → (key →
queue)`, not a single flat map, for two reasons — it makes the
`--parallel` determinism argument below structural rather than a
comment, and it lets a miss be reported as *"this piece has N recorded
attempts, none match"* with the first turn that differs, rather than a
bare "not found." `piece_id` is free: every completion's `messages`
already carries the piece text in its user turn, so grouping by
`piece_id` costs nothing extra to compute.

**`--parallel` determinism**: a key collision (the demote-restart
case) only ever occurs *within one piece's own ladder*, which one
worker runs sequentially end to end (`extract_piece`'s recursion,
`chunking.rs:481-487`); different pieces always key differently
(different user-turn text). So FIFO consumption never races across
`--parallel` workers, and replay is deterministic regardless of
`--parallel`'s value.

### 3.3 Miss policy is a mode, not a boolean

`--replay auto`: a request with no matching record falls through to a
live call. `--replay strict`: it fails the document instead, with the
piece-scoped diagnostic from §3.2. `off` (default) never consults the
index. Two named modes rather than a bare flag, because the difference
— "spend money" vs. "fail loudly" — is exactly the choice between
"I'm iterating locally and don't mind topping up" and "I'm running
somewhere with no model at all and want to know the instant something
doesn't match" (§3.8); burying that behind an implicit default would
get one of the two audiences wrong silently.

### 3.4 File safety: snapshot before open, append — never truncate — on replay

`ReplayIndex::load` reads the whole attempts log into memory
**before** `AttemptLog::open` runs for this run (i.e. between
`load_checkpoints`, `run.rs:477`, and `open_attempt_log`, `run.rs:492`).
The index is then immune to whatever the run's own log-opening does
next.

`AttemptLog::open` (`attempts.rs:60-74`) today truncates
(`fs::File::create`) whenever the checkpoint holds no units — which,
for a document whose batch already landed, is always, since
`checkpoints.clear()` runs after a successful write (`run.rs:669`).
Truncating the very file a replay run is about to read from would
destroy the input mid-read if the two operations were not already
ordered by the snapshot above, and would also destroy it for any
*later* run against the same `--out` that wanted to replay from it.
**A replay run therefore always opens its own attempts log in append
mode, never truncate — including when `--force` is set.** This
narrows ADR 0025 §3.2's "fresh start truncates, `--force` starts
fresh" rule for the one case ADR 0023 §1 left to this issue: replay
was never something the pre-#781 rule was written with in mind, and a
truncate here would be actively hostile to the feature's own reason
for existing.

A replay run's own completions — live or replayed — are appended to
the *same* file under a new `run_id`; ADR 0025's format already
tolerates one file spanning several runs (joins are `(run_id,
attempt_seq)`, never `attempt_seq` alone), so this needs no new
convention. A separate `--replay-from`-only output file was considered
and rejected: it would need a new file-discovery rule for tools that
already know to look beside the batch, for no benefit over the join
rule that already exists.

Two new record kinds mark what happened: `kind: "replay"` (first,
right after `document`) names the mode and the source directory;
`kind: "replay_summary"` (last) counts replayed vs. live vs. missed.
A stderr line per document — `replayed N/M completions (K live)` —
gives the same count without opening the log.

### 3.5 Replay reads neither the manifest skip nor the checkpoint

**The manifest skip is bypassed under replay.** The skip
(`run.rs:407-409`, `self.manifest.matches(...)`) exists to avoid
paying for model calls on an unchanged document; under replay a
completion is free whether or not it is a hit, so the skip's reason
for existing is gone. Requiring `--force` to get around it was
considered and rejected: `--force` also empties the checkpoint
(`checkpoint.rs:164`), which starts the attempts log fresh
(§3.4) — forcing exactly the file replay needs to read.

**Replay is never a manifest or checkpoint computation input.** A
replayed completion is byte-identical to its recorded original, so the
batch it produces is the same function of the same inputs a live run
would have produced; adding a `replay: bool` field would make a
replay-built batch and a live-built batch of the same document look
like different computations, and the next live run would silently
re-extract everything a replay run had already written. No field is
added to `ComputationInputs` (`manifest.rs:234-253`) or
`CheckpointFingerprint` (`checkpoint.rs:15-59`).

**A replay run does not read the checkpoint store.** `checkpointed_unit`
(`chunking.rs:322-334`) short-circuits *before* a completion is even
attempted, returning a `ModelOutput` that was parsed and mechanically
validated **under whatever code ran the original extraction** — for
the exact use case this issue's own body names first ("change the
validation rule, replay the rest"), consulting the checkpoint would
silently hand back an answer validated under the *old* rule while
believing it had honored the new one. `load_checkpoints` is skipped
outright under replay (extending the existing `if self.force` branch,
`run.rs:310-321`, to `if self.force || replaying`), and no unit is
recorded during a replay run either — nothing would read it back, and
a replay run that dies partway restarts for free (every completion it
already replayed is free to redo).

§3.9 explains the alternative this section forecloses — building the
checkpoint store itself from the log instead — and why it was rejected
despite looking cheaper.

### 3.6 The cross-document vocabulary problem: pin the recorded system prompt

Three options were weighed for the problem in §2 (the label vocabulary
that changes document-to-document changes every later document's
system prompt, so a naive full-conversation key breaks replay's most
basic multi-document use case — validate the rule change on document
one, replay documents two onward for free):

- **(a) Drop the system turn from the key entirely**, matching only on
  the user turn and `requested_max_tokens`. Rejected: this hands the
  model's answer to a *different* prompt than the one that produced it
  — different candidates, different vocabulary offer, a different
  `--schema` — as if it were the answer to today's prompt. That is not
  "replay tolerating drift", it is silently discarding the one thing
  content-addressing exists to guarantee.
- **(b) Keep the whole-conversation key and accept the miss rate.**
  Rejected on the evidence: `Run.vocabulary` starts empty for a
  document run alone (`taguru extract --replay strict doc3.md` has no
  prior documents to build it from), so even a *single*-document
  replay of a document that was originally extracted after others in
  the same run would miss on its system turn immediately — the
  fallback everyone will reach for first doesn't work under option
  (b).
- **(c) Pin the recorded system prompt when the index names exactly
  one.** `ReplayIndex` groups its `system` records by hash; if there
  is exactly one, its content is used verbatim as the system turn for
  this document — `system_prompt(...)` is never called — at all three
  call sites in `run.rs` (`:741` sequential, `:811` `--parallel`,
  `:885` Stage 2). If more than one distinct system record exists (a
  resumed document whose checkpoint-append spans a run where the
  vocabulary differed), nothing is pinned and matching falls back to
  computing `system_prompt()` fresh — ambiguity is never resolved by
  guessing. A mismatch between the pinned system and what this run's
  settings would have produced is reported once, on stderr, and
  recorded on the trace's `steering` record (`system_sha256` and
  `pinned_from: {run_id}`, ADR 0027) so a reader can see the prompt
  actually differed from what today's flags imply.

**(c) is the decision.** It is the only option under which "replay
documents two onward" and "replay one document in isolation" both work
without operator intervention, and it costs nothing new to identify —
ADR 0025 already writes the system record by hash once per document.

**What is not pinned, and why that is a real limit, not an oversight.**
The user turn is a pure function of the document text and
`--chunk-bytes` (`prompt.rs:217-227`) and correctly falls through to a
live call when either changes — pinning it would defeat replay's
entire purpose. The corrective ask's user turn
(`corrective_validation_message`, built from *today's* validation
issues) is likewise never pinned: it is not steering, it is the
question this run is actually asking, and if the validation rule
changed, it is a genuinely new question the model has to answer. **A
document whose original extraction needed a corrective round, replayed
under `--replay strict` after a validation-rule change, can therefore
fail** — the corrective turn has no recorded answer to a question that
was never asked. This is documented as a known limit (implementation
docs, #823), not solved here: solving it would mean fabricating an
answer to a question the model never saw, which is worse than failing
loudly.

### 3.7 CLI surface

`--replay {auto|strict|off}` (default `off`) + `--replay-from DIR`
(default `OUT/.extract-trace`) + `--resume-from STEP` (accepts ADR
0030's step names; "everything before STEP is satisfied from records
where they exist, decided per request exactly as replay always is —
this flag changes nothing about *matching*, only which steps are
*attempted* live at all"). `TAGURU_EXTRACT_REPLAY` /
`TAGURU_EXTRACT_REPLAY_FROM` are the matching environment defaults.

`--replay strict` does not require `TAGURU_EXTRACT_URL` (§3.8);
`TAGURU_EXTRACT_MODEL` stays required even then, because
`ComputationInputs.model` (`run.rs:389`) is written into the manifest
— an empty model name there would be a computation input that lies
about what was actually asked, and would let a later live run believe
a document already matches its settings when it does not.
`--structured-output auto` combined with no client configured is a
usage error: the ADR 0021 probe requires a live call by construction
and is never something replay can stand in for (§3.1).

### 3.8 Operating across machines

The scenario #781 names third: extract on a machine with model access,
replay everything downstream of `call`/`reconcile` on one that has
none. What has to travel: the document files themselves (the attempts
log's `messages` hold piece text, not the whole document or paragraph
offsets — reconstructing coverage or passage rendering from the log
alone is not attempted), `.extract-trace/*.attempts.jsonl`, and the
`--schema` document if one was used. The batch, the manifest, and the
checkpoint directory do not need to travel — replay produces its own
batch and reads neither of the other two (§3.5). Because the matching
key is exactly what was sent, a setting left unmatched between the two
machines is self-diagnosing: it changes the key, which either falls
through to a live call (impossible without a URL — `--replay strict`
fails loudly) or is reported as a mismatch (§3.6) — there is no silent
"looks right but used the wrong prompt" failure mode to guard against
operationally.

### 3.9 Rejected: replay by pre-populating the checkpoint store

An alternative considered in depth: instead of matching conversations,
read the attempts log, pick each piece's accepted (`state:
"stop_valid"`) answer by `piece_id` (already the checkpoint's own key,
ADR 0023 §3.2), and inject it into `CheckpointStore` before the normal
pipeline runs — `checkpointed_unit` (`chunking.rs:322-334`) would then
short-circuit every piece with no other code change.

This is materially smaller to build, gets fingerprint-based
invalidation for free from `CheckpointFingerprint`
(`checkpoint.rs:15-59`), and sidesteps §3.6's vocabulary problem
entirely (it never compares prompts). It was rejected on three points,
each fatal on its own:

- **It cannot serve §3.8's use case for a document that needed a
  split, a Stage 2 correction, or (the paradigm case this issue exists
  for) a validation-rule change the recorded answer no longer passes.**
  The checkpoint only ever holds a piece's *final accepted* answer —
  never a mid-ladder failure, never Stage 2's result (Stage 2 is never
  written back to the checkpoint even today, `run.rs:1096`). Every one
  of those cases falls straight through to a live call, which a
  URL-less machine cannot make. §3.8's scenario would only work for
  documents that happened to need none of the ladder's machinery —
  not a bound #781 can ship as its headline capability.
- **It would leave a run's own attempts log empty.** A checkpoint hit
  returns before `observers.emit` is ever called (`chunking.rs:485-487`),
  so a replay run built this way would record nothing about what it
  did — directly undermining #784's premise that a run's own record is
  a complete account of it.
- **It collapses four of ADR 0030's steps (`call`, `parse`, `validate`,
  a corrective `correct`) into one opaque "reused" unit**, leaving
  `--resume-from` with nowhere meaningful to distinguish them — exactly
  the ambiguity ADR 0030 was written to prevent.

Two things from this design are kept, because they are correct on
their own merits independent of the rejection: the `piece_id`-grouped
index shape (§3.2), and adding a `settings` record (a
`CheckpointFingerprint`-shaped snapshot: model, `chunk_bytes`,
`structured_output`, etc., §3.2's diagnosability point) to the
attempts log purely as **a diagnostic, never a gate** — replay's
matching is still decided by the conversation, but a settings mismatch
can be named to an operator in one line instead of discovered as N
silent misses.

## 4. Consequences

- `--replay auto` on an unmodified run replays every completion; the
  same run after a validation-rule change replays every piece whose
  conversation is unaffected and calls the model only for the pieces
  whose corrective ask actually changed — the cost profile #781 was
  written to get.
- `--replay strict` with no `TAGURU_EXTRACT_URL` is what makes #3.8's
  two-machine operation possible; it fails loudly, per document, the
  moment a setting drifts between the two machines.
- `Run.vocabulary`'s accumulation across documents stops affecting
  replayed prompts once §3.6's pin engages — it is still *computed*
  (nothing here changes `run.rs:670-672`) but a pinned document's
  system turn no longer depends on it. A document whose original
  extraction spans a checkpoint-resumed run with two distinct system
  prompts cannot be pinned and falls back to live matching (§3.6).
- A document whose original extraction needed a corrective round fails
  `--replay strict` if the validation rule that caused the correction
  changed — a documented limit (§3.6), not a bug.
- `--out/.extract-trace/` gains two record kinds (`replay`,
  `replay_summary`) and the `attempt` record gains `rung`; the
  attempts log gains a `settings` record. None of this touches the
  batch, the manifest, or the diagnostics sidecar's existing shape.
- The manifest skip and the checkpoint store are both inert under
  replay (§3.5) — a replay run always processes every document it is
  given and never reads or writes `.extract-checkpoints/`.
- Cost: three call sites (`chunking.rs:137,696`, `run.rs:927`) and
  three system-prompt assembly sites (`run.rs:741,811,885`) all thread
  a new value through; this is #781's next child (#816) to do, as a
  behavior-preserving refactor before replay itself lands.
