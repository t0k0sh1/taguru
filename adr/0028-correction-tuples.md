# 0028. Every corrective turn names the attempt it corrects

- **Status**: Accepted
- **Date**: 2026-08-24
- **Issue**: #790
- **Related**: #784 must-have (2) (the requirement: the recovery
  process auditable after the fact), ADR 0001 §8 / ADR 0013 (the
  corrective turns being recorded), ADR 0023 (ids), ADR 0024 (losses —
  where an uncorrected item ends up), ADR 0025 (the attempts log that
  already holds every ask and answer in full), #792 (the correction
  success rate computed from these records)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How a correction is recorded as a tuple — what was flagged, what was
asked, what came back, what was adopted, against which text — for
Stage 1 (per-answer corrective turns, syntax and validation, the
empty-answer corrective included) and Stage 2 (cross-chunk) alike.
Out of scope: judging whether a correction preserved meaning (#784
derivative c), and the rate arithmetic (#792).

## 2. Context

After ADR 0025 the raw material of every correction is on disk: the
failed attempt carries its `parse_error`/`validation_issues`, the
corrective attempt's `messages` carry the replayed bad answer and the
corrective ask verbatim, and its `answer` is the candidate the
mechanics then accept or reject. What was missing was the join: which
attempt a corrective attempt corrects. Ordering heuristics (same
piece, adjacent sequence) break exactly where auditing matters —
Stage 2 corrections happen long after the Stage 1 attempt they
correct, other pieces' attempts in between, and `--parallel`
interleaves sequences.

## 3. Decision

Attempt records — in the attempts log and the diagnostics sidecar
alike — gain `corrects: {run_id, attempt_seq}`, present exactly on
corrective attempts:

- Stage 1: the previous attempt of the same piece — the one whose
  answer this attempt's `messages` replay. Set for every corrective
  kind (malformed, invalid, empty).
- Stage 2 (`stage: cross_chunk`): the **accepted** attempt whose
  output is being corrected (the `ChunkOutput`'s own `attempt`).
  Absent only when that output came from a pre-0023 checkpoint,
  which recorded no attempt to name.

The field is additive; consumers of either stream are unaffected
(ADR 0023 §3.3's additivity rule for the sidecar).

The tuple is then a join, not a new record: **flagged** =
`corrects`-target's `parse_error`/`validation_issues` (path-addressed);
**asked** = the corrective attempt's own last `messages` turn;
**answered** = its `answer`, in full; **adopted** = the trace `piece`
record's `attempt` equals this attempt's id (its answer became the
piece's output — for Stage 2, replaced it), with any items the
mechanics still removed from it in the `loss` records (ADR 0024) by
`path` into that answer; **original text** = the piece via `piece_id`
(and a cited paragraph via the item's own locator). Nothing is stored
twice.

## 4. The correction success rate (for #792)

Per corrective attempt A with `corrects: B`: the flagged set is B's
`validation_issues` (or its one `parse_error`); A resolved it when A's
state is `stop_valid` (removals excepted — an item flagged on B and
mechanically removed from A's answer is ADR 0024's `removed` loss, the
"解消せず除去" bucket, joined by path); a Stage 2 alias issue still
standing after A is the ADR 0022 prune, likewise a `loss`. Rate =
resolved flagged issues ÷ flagged issues, rolled up piece → document
→ context → group → run, alongside the removal-instead ratio.

## 5. Consequences

- "何を指摘し、何に変わり、採用したか" is answerable for any
  correction by ids alone,原文と並べて — #784 must-have (2).
- A meaning-changing correction (the hallucination risk the current
  acceptance criteria cannot see) is findable after the fact by
  reading exactly the joined pair, which is the record #784
  derivative c would build prevention on.
- The link is recorded at the three call sites that build corrective
  conversations, nowhere else; no record shape is duplicated.
