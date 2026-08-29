# 0032. The cross-chunk correction escalates once, then its issues stand

- **Status**: Accepted
- **Date**: 2026-08-29
- **Issue**: #811
- **Related**: #780 (the baseline measurement that reproduced the
  failure), ADR 0001 §7/§8 (the ladder and the correction taxonomy
  this amends), ADR 0019 (the capped escalation this reuses), ADR 0020
  (the split rung this does NOT reuse), ADR 0022 (the "still standing
  → alias removed, anything else fatal" ruling this connects to), ADR
  0029 (the `escalate` move record this now also writes for Stage 2)
- **Supersedes**: ADR 0001 §8's "one targeted corrective turn, a
  length-limited reply fails the source" — for the Stage 2
  (cross-chunk) corrective turn only. Stage 1's per-answer corrective
  budget is untouched. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What `taguru extract` does when the Stage 2 cross-chunk corrective
answer ends with `finish_reason: length`. Out of scope: the Stage 1
ladder (ADR 0001 §7, 0019, 0020, 0021 — unchanged), what counts as a
Stage 2 issue (unchanged), the Stage 2 refusal/empty/malformed/
still-invalid outcomes (unchanged — fatal), and `--lossy`.

## 2. Context

The #780 baseline (v0.9.5, qwen3 30B-A3B) lost the same document —
the ripgrep README, English prose with many aliases — in two
consecutive runs by the same path:

```
chunk 1/1: the cross-chunk correction was cut off at the output limit —
failing the source rather than importing a truncated correction
```

Stage 1's `length` has had a ladder since ADR 0001 §7 (escalate once
at a capped budget, ADR 0019; then split, ADR 0020). Stage 2's
corrective turn — the one call that rebuilds an output's conversation
and asks the model to fix the alias issues the cross-output check
flagged — had none: a `length` there was fatal, with no resend. The
0.9.3 "an output cap loses the whole document" failure that ADR 0019
closed for Stage 1 was still open for Stage 2, and the checkpoint
resume ADR 0022 made visible does not help — the rerun asks the same
correction, at the same cap, and is cut off the same way (measured:
two runs, identical failure).

Two things distinguish Stage 2 from Stage 1:

- **There is no piece to split.** The corrective ask is about an
  output's whole alias set against the document's whole association
  set; halving the text halves nothing the issues are about. ADR 0020's
  split rung has no analogue here.
- **The correction is optional in a way the extraction is not.** ADR
  0022 already rules that an alias issue the corrective turn leaves
  standing is removed with accounting rather than failing the source
  (an alias is a spelling variant, never a fact), and that a standing
  non-alias issue (a schema domain/range violation) is content and
  stays fatal. A corrective answer that never arrived intact is
  exactly a turn that left its issues standing.

## 3. Decision

**A Stage 2 corrective answer that ends at the output cap is resent
once at ADR 0019's escalated budget (`TAGURU_EXTRACT_ESCALATION_FACTOR`
× `--max-output-tokens`), the cut-off answer discarded. If the resend
is cut off too — or no budget is configured, so there is no ladder to
climb — the correction is treated as having left its issues standing:
the output keeps its accepted Stage 1 answer, and ADR 0022's re-check
rules on what stands (an alias issue is removed with accounting, any
other issue fails the source). A truncated correction is never
imported, and Stage 2 never splits.**

1. **Escalate exactly as Stage 1 does.** The same corrective ask
   (same messages, same `corrects` attribution) resent neutrally at
   the escalated cap; the truncated answer is never replayed as a
   prior turn and never salvaged as a prefix. Only when a budget is
   configured — without one the ladder escalates nothing, for Stage 2
   as for Stage 1. At most one escalation per corrected output, so the
   call bound rises from one to two per flagged output.
2. **No split rung.** There is no sub-piece for a corrective ask, and
   inventing one (correct half the aliases at a time) would change
   what the model is asked in a way the checkpoint/manifest
   fingerprint does not record. The ladder's last rung for Stage 2 is
   "the issues stand".
3. **Cut off at the top = standing, not fatal.** The corrective turn
   was spent (ADR 0022 §3.1's requirement); what it could not deliver
   is what stands. The existing re-check then partitions by path
   exactly as ADR 0022 §3.2 says — alias issues removed highest-index
   first with the `… — still so after the corrective turn; removed`
   line, a non-alias issue fatal with the issues named. Nothing about
   the removal's accounting changes; the runs that used to fail here
   now land minus the aliases the model could not correct within the
   ladder.
4. **Recorded as before, plus the move.** Each corrective completion
   is one `stage: "cross_chunk"` attempt record (ADR 0025), so the
   escalated resend is a second record with its own
   `requested_max_tokens`; an `escalate` move record (ADR 0029, with
   `from_max_tokens`/`to_max_tokens` and a reason naming the
   cross-chunk correction) sits between them, joined by `piece_id`. A
   cut-off at the ladder's top is one stderr line naming the chunk and
   whether the escalated budget was tried, followed by ADR 0022's
   removal lines. No new record kinds, no on-disk format change.
5. **Refusal, empty, malformed, and still-invalid stay fatal.** Only
   `length` is a budget property the ladder can act on; the rest are
   the model's judgment, and ADR 0001 §8.2's ruling for them stands.

## 4. Consequences

- **Behavior change, named in the changelog**: a document that used to
  fail on a cut-off Stage 2 correction now costs one more model call
  and lands — with the corrected aliases when the resend fits, without
  the uncorrectable ones when it does not. A run's `removed` counts
  can rise by these; its `failed` counts fall.
- **Cost bound**: Stage 2 was one call per flagged output; it is now
  at most two, the second capped at the escalated budget (never
  uncapped unless the operator set factor `0`). The same looping-model
  argument ADR 0019 made bounds the wall-clock.
- **Manifest/checkpoint**: no new input. The escalation factor is
  already a fingerprint input when non-default (ADR 0019 §3.4), and
  the corrective ask is unchanged.
- **Tests**: end to end, a cut-off correction is resent once at the
  escalated cap with the same messages and lands the corrected alias
  (move record and both attempt records pinned); cut off twice, the
  document lands without the alias and with the removal named, the
  batch carrying nothing of the truncated answer; with no budget, one
  corrective call and no move.
