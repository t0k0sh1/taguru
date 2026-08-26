# 0030. The extract pipeline is a named, ordered, open list of steps

- **Status**: Accepted
- **Date**: 2026-08-26
- **Issue**: #814
- **Related**: ADR 0023 (the identifiers every step's records are keyed
  by — this ADR adds names to those coordinates; ADR 0023 §1 put
  resume out of its own scope and named #781, which this and its
  sibling ADRs carry forward), ADR 0003 §2.2/§7 (the manifest's
  compute-input reasoning `read` inherits, and the paragraph
  provenance `plan`'s chunk record mirrors), ADR 0014/ADR 0015/ADR
  0009 §11.1 (what `steer` assembles: candidates, vocabulary, schema),
  ADR 0013 (`validate`'s mechanical pass), ADR 0022 and #758
  (`reconcile`'s Stage 2 prunes), ADR 0016 and ADR 0026 (`verify`'s
  coverage), ADR 0001 §7/§8/§10 (the length ladder, "a batch is
  complete or nothing", one completion is one attempt — why the
  ladder's moves are not steps), ADR 0028/ADR 0029 (`corrects`/`move`
  — why corrective rounds and ladder moves are not steps either), #179
  (checkpoints — today's unnamed partial step store, §4), ADR
  0024–0027 (the records this ADR indexes by step in the accompanying
  docs change's "by step" table), #784 (the umbrella those records
  came from), #782 (the steps it is expected to add, §3.6), #781 (the
  parent issue; ADR 0031 continues where this ADR's Scope stops)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The names, order, scope, and produced artifact of every step
`taguru extract` takes a document through, as a public contract:
where each name may appear, and the procedure for adding a step. Out
of scope: the on-disk storage format for a step's input/output (ADR
0031), fingerprint invalidation and `--resume-from`'s actual
implementation (ADR 0031 and #781's later children), and any change to
the pipeline's behavior — this ADR changes no code and no byte the
binary produces. Keeping storage out of scope is deliberate: it lets
this ADR reach Accepted, and its names become load-bearing, before the
storage design is settled, so #781's later work and #782's new steps
both build on a fixed vocabulary rather than a moving one.

## 2. Context

Today the only description of extract's shape is the module doc atop
`src/extract.rs`, a paragraph of prose naming the submodules in
roughly the order they run. `docs/extract.html` explains extract by
*feature* — Chunking, Structured output, The manifest, Chunk
checkpoints, Diagnostics sidecar, Trace, The attempts log — and none
of those sections says "extract is made of the following steps." #784
(ADR 0023–0029) gave every produced item a coordinate back to the
completion that made it, but each record is named for what it *is*
(a `chunk`, a `piece`, an `attempt`), not for the *work* it reports on
— there is no word a user can type for "the mechanical-validation
step" today.

#781 needs that word twice: `--resume-from <step>` needs step names to
accept, and the storage design (ADR 0031) needs step names to key
records by. #781's own body proposes eight steps, read off the module
doc: chunk planning, prompt assembly, model call, parse result and
issues, corrective turns, mechanical-validation removals, chunk
aggregation, rendering. Reading the actual control flow in
`Run::extract_document` (`src/extract/run.rs:327-701`) against that
list surfaces two problems: **one entry is placed wrong** (mechanical
validation is not a stage *after* corrective turns; it is what
*decides whether* a corrective turn happens), and **four are missing**
(reading the document, assembling the steering block, the Stage 2
cross-chunk reconciliation, and coverage verification all produce
artifacts a user can point at and all have a real "just redo this
part" story, but none appear in the eight). This is the evidence that
a step list must be read off the control flow, not off a module
doc's prose, and that it must be published rather than re-derived by
each reader.

#782 will add steps for document-structure analysis and context
generation (a synopsis-so-far, referenced-section summaries) between
today's steps. A list frozen at eight, or one that leans on position
to mean anything, breaks the first time #782 lands.

## 3. Decision

### 3.1 What counts as a step: two tests

A step must pass both:

- **The read test**: does it produce one artifact with a name a user
  can point at ("I want to see this step's output")?
- **The resume test**: is there a real change — a different flag, a
  different rule — after which everything up to here is still good
  and only this point on should be redone?

Passing only one disqualifies an entry as a step; it may still be
recorded (§3.1.1), but `--resume-from` will never name it and the
step table will not list it.

#### 3.1.1 What is not a step, and which test it fails

| Actual work | Why it is not a step |
|---|---|
| Batch-name / source-id collision check | No artifact (fails the read test) |
| The manifest skip decision | No artifact — it is the gate *above* the pipeline, not a step in it; `--resume-from manifest` would be meaningless |
| `crate::ingest::parse_batch` self-validation | A postcondition of `render`, not a step of its own — a "batch written but not yet validated" state must never be observable |
| `storage::write_atomic` | Part of `render` — the string and the file are the same artifact |
| Writing the trace / attempts log / manifest record / clearing the checkpoint | Bookkeeping *about* a step, not a step itself — `--resume-from trace` would be incoherent |
| Absorbing vocabulary and claimed names into the run | Run-scoped carry-over into the next document's `steer`, not a per-document step |
| `--questions` (doc2query) | Rides the same `call` as everything else; a feature is not a step |

### 3.2 The steps

| Step | Scope | Model call | Produces |
|---|---|---|---|
| `read` | document | — | the document text as sent, `document_sha256`, the canonical paragraph numbering |
| `plan` | document | — | the chunk plan: `chunk_index`/`chunk_sha256`/`chunk_bytes`/paragraph range per chunk |
| `steer` | document | — | what taguru puts into the prompt on its own: candidates, vocabulary, context names, schema block (ADR 0027, 1:1) |
| `prompt` | piece | — | the messages actually sent, every turn, including a corrective round's replay |
| `call` | attempt | **yes** | the raw answer, `finish_reason`, usage, `transport_retries`, the classified `state` |
| `parse` | attempt | — | the `ModelOutput` and any parse issue (`--lossy`'s parse-time drops included) |
| `validate` | attempt | — | the items kept after ADR 0013's mechanical pass, and its `Removal` records |
| `reconcile` | document | **yes** | the piece outputs after Stage 2 cross-chunk correction, and what it prunes (ADR 0022, #758) |
| `merge` | document | — | the `Extraction` — duplicates folded, contract enforced, one document's worth |
| `render` | document | — | the batch file (self-validated, atomically written) |
| `verify` | document | — | paragraph coverage (always) and uncovered candidate pairs (`--coverage`) |

Repetition and branching inside a step are not separate steps (§3.4);
order is total (§3.3); the list is not closed (§3.6).

### 3.3 Order is total; scope is data

Scope is a nested vocabulary: `run ⊃ document ⊃ chunk ⊃ piece ⊃
attempt`. No new identifier is introduced — every step's records are
keyed by exactly the coordinates ADR 0023 §3.2 already minted
(`run_id`, `source`+`document_sha256`, `chunk_index`+`chunk_sha256`,
`piece_id`, `attempt_seq`).

The table's order is total, enforced by one barrier rule: **a
coarser-scoped step runs only after every unit of the finer-scoped
steps before it has finished.** This is why `validate` (attempt scope)
can precede `reconcile` (document scope) without contradiction, and it
is what makes `--resume-from <step>` well-defined regardless of which
scope `<step>` names: resuming at a piece-scoped step means "for every
piece, from here"; resuming at a document-scoped step means "once."

Two rejected alternatives: a table per scope (readers would have to
interleave two tables to know what happens when, and `--resume-from`
would need to say which table it means) and a DAG (`--resume-from`
needs a total order, and every real branch in the code is a
repetition, not a fork — see §3.4).

### 3.4 Repetition is not a step

A Stage 1 corrective round, the ADR 0019 escalate resend, an ADR 0021
demote-and-restart, and an ADR 0001 §7 split all repeat the same four
steps (`prompt` → `call` → `parse` → `validate`) rather than doing new
kinds of work. None of them is a step:

- Making them steps would break §3.3's total order — the table would
  no longer be a line, and "resume from the corrective step" would not
  say which round.
- ADR 0028's `corrects` (`{run_id, attempt_seq}` on a corrective
  `attempt`) and ADR 0029's `move` records already track exactly this
  — "which attempt corrected which" and "why the ladder moved" are
  answered by joining existing records, not by adding steps.

### 3.5 Names are the identity; the table carries one version

1. **A position number is never part of the contract.** It may appear
   in prose for a reader's orientation, but never in a name, a file
   name, or a record field. #782 inserts into the *middle* of the
   table; a number baked into an identifier would either lie or force
   every stored artifact and script to renumber.
2. **A name is never reused or repurposed.** Once an Accepted ADR
   names a step, that name means the same thing forever. When a step
   is split, the artifact's successor keeps the original name and the
   new work gets a new name. A retired step's name is retired with it
   — never reassigned.
3. **The table carries exactly one version number**, `pipeline_version`
   (starting at `1`), bumped when a row is added, removed, or its
   scope/artifact changes. This governs future step-scoped storage
   only (#781's later children, ADR 0031 onward) — today's trace,
   attempts log, and diagnostics sidecar keep their shape unchanged
   (§4) and carry no such field. Where a step-scoped record does
   exist, it carries `step` (the name) and `pipeline_version` (which
   edition of the table it was produced under) — not a per-step
   version; order is a property of the table, not of any one row, so
   two rows cannot disagree about it.
4. **`pipeline_version` is never a manifest computation input.** A
   table edition changing the *description* of the pipeline does not
   change a batch's *content*; a document extracted under table
   edition 1 must not spuriously re-extract under edition 2.

### 3.6 Adding a step: procedure, and #782's worked example

Adding a step means: a new ADR names the row (position, scope,
artifact, and its answer to both §3.1 tests), the same change updates
`docs/extract.html`'s "The pipeline" table, and `pipeline_version`
bumps. Only the storage
for steps at or after the insertion point is invalidated — everything
before it is still good, which is the entire point of naming steps in
the first place.

#782 is expected to add (illustrative, not decided here):

| Step (working name) | Scope | Model call | Inserted |
|---|---|---|---|
| `structure` | document | — | after `read`, before `plan` |
| `context` | document | **yes** | after `plan`, before `steer` |
| `annotate` | chunk | — | after `context`, before `steer` |

Note `plan` is placed *before* `steer` in §3.2 even though today's
code computes candidate names (part of `steer`) before cutting the
chunk plan — candidates are a whole-document computation independent
of the plan, so ordering `plan` first costs nothing today and gives
`structure` and `context` a natural point to insert without touching
`read` or `plan` at all. Inserting `context` invalidates every stored
artifact for `steer` onward (the prompt genuinely changes); `read` and
`plan`'s stored artifacts remain valid. That — not "nothing is ever
invalidated" — is what "the list tolerates insertion" means.

### 3.7 One spelling, everywhere

A step's name must appear identically in: this table, any stored
record's `step` field, any file or directory name keyed by step, the
values `--resume-from` accepts, and the binary's own dump of its step
table (a future `--steps`, deferred to #781's implementation issues —
this ADR only commits to the binary being the ultimate source once
that lands, so the docs table is never authoritative over a running
binary).

### 3.8 Reader rules

A record naming a step this reader does not know is kept, not
discarded, and its position is never guessed. A run's own table (or,
once it exists, its own `--steps` dump) is the only authority for that
run. `--resume-from <unknown name>` is a usage error (exit 2) listing
the binary's known step names.

## 4. Consequences

- Extract gains a vocabulary: "the model's answer was right but
  `validate` removed it" now has a subject. #794's observation→knob
  table can be indexed by step.
- #782 adds rows without a format change — see §3.6.
- #781's body's eight-step list is superseded by the table in §3.2.
- **`parse`'s artifact has no record of its own today** — no trace or
  attempts-log line carries the `ModelOutput` independently of
  `validate`'s outcome; a rejected answer's parse is never recorded at
  all. An *accepted* piece's parsed-and-validated output is not
  entirely absent, though: it is held in a checkpoint unit until the
  batch lands (next bullet) — just not as a first-class, addressable
  step record. Closing that gap is #781's storage design (ADR 0031)
  to do.
- **The checkpoint file is already an unnamed partial step store.**
  `CheckpointUnit` (`src/extract/checkpoint.rs:69-97`) holds one
  piece's `prompt` (`user`), `call` (`answer`), and `parse`+`validate`
  (`output`/`removed`/`unparsed`) keyed by `piece_id`, and is reused on
  resume. ADR 0031's storage design will either generalize this or
  subsume it.
- No existing record (`trace`, attempts log, diagnostics sidecar)
  changes shape. The step↔record correspondence is given by
  `docs/extract.html`'s "by step" table alone — no `step` field is
  added to any of them (rationale in the accompanying docs change:
  most record kinds already map 1:1 to a step by name; the ones that
  do not, like `loss`, are already disambiguated by an existing field
  (`reason`) that a `step` field would only risk contradicting).
- One known inaccuracy is left in place rather than fixed here: the
  `loss` record's `reason: "removed"` covers both `validate`'s Stage 1
  mechanical pass and `reconcile`'s Stage 2 alias prunes; docs already
  say so in prose. Splitting `reason` further is future work, not part
  of this ADR.
- Cost: every future change to the flow needs an ADR and a docs update.
  §3.1's two tests are the check against step proliferation.
- Risk: naming something a step is a promise that `--resume-from` will
  eventually accept it. If a later ADR finds a named step cannot
  actually be resumed into, that is a contract change (a new ADR), not
  a silent docs edit — mitigated by requiring the resume test at entry.
