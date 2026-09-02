# 0035. Runaway output detection: an answer that outgrows its piece

- **Status**: Accepted
- **Date**: 2026-09-02
- **Issue**: #854
- **Related**: #852 (prompt-side suppression of the same phenomenon),
  #853 (output-side mechanical removal, not yet designed), #783 (the
  field run that observed it), ADR 0001 §7 (the ladder this amends),
  ADR 0019 (capped escalation), ADR 0020 (timeouts descend the
  ladder), ADR 0021 (rung demotion), ADR 0029 (move records)
- **Supersedes**: nothing wholesale. Amends ADR 0001 §7's ladder with
  one judgment: a `length` answer that has already outgrown its piece
  skips the escalation and split rungs. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The item-stage ladder only: what happens after a round ends at the
output cap when the truncated answer is already disproportionately
large for the piece that prompted it. Out of scope: the overview pass
(ADR 0034) and the Stage-2 cross-chunk correction (ADR 0032) — their
input/output economics differ (the cross-chunk correction's input is
the run's own prior answers, not a piece) and both keep their single
capped escalation; the *content* of an oversized answer that finishes
under the cap (that is #853's mechanical removal); and aborting a
completion mid-generation (the client is not streaming — the saving
here is the skipped rounds, not the round that detected it).

## 2. Context

The #783 field run (qwen3:30b via Ollama, minutes corpus) produced the
failure this guards against. A piece whose body was one heading line
(`[0] 衆議院 総務委員会 第2号 2024-12-12`, ~50 bytes beside a ~2.8 KiB
chunk-context block) answered with 21,745 bytes — 191 associations,
144 of them the same subject negating phrases that appear nowhere in
the input — and ended at the cap. The ladder read that as "the piece
is too big": it escalated (ADR 0019, 4000 → 8000 tokens), got `length`
again, and split (ADR 0020) — but the output was tracking nothing in
the input, so sub-pieces reproduced it and the document failed at the
split floor, after paying for every wasted round. The ladder's rungs
all assume output size is a function of input size; a fabricating
model breaks that assumption, and no rung can converge on it.

Measured over every attempts log the verification corpus holds (490
item/cross-chunk attempts across the 0.9.5 shakedown, the #780
baseline, and the #783 run): successful item-stage answers stay at or
under **5.5×** their user turn's bytes (p50 0.63, p90 2.05, p99 4.07,
max 5.43); the observed runaway attempts sit at **10–23×**. The two
populations do not overlap, and the runaway side is truncated by the
cap itself — untruncated it would be higher still.

The ratio is judged against the **piece body's bytes**, not the user
turn's: the chunk-context block (ADR 0033) can be many times a small
sub-piece's own text — in the observed case 2,833 of the user turn's
2,883 bytes — and it was precisely the block the fabrication fed on.
Against the piece body the observed runaway is ~400×; against the
user turn it is 7.5×, inside the legitimate range. Facts come from
the piece; the piece is "the input" the output must track.

## 3. Decision

**When a round under the item-stage ladder ends at the output cap and
the truncated answer's bytes exceed `TAGURU_EXTRACT_RUNAWAY_RATIO` ×
the piece's bytes (default 8), the ladder treats the piece as running
away: the escalated resend and the split rung are skipped — a bigger
budget or a smaller piece demonstrably does not converge on an output
that is not tracking the input — and only ADR 0021's rung demotion is
still tried before the source fails.**

1. **Default 8×, judged in bytes.** Above every legitimate ratio the
   corpus has produced (max 5.43×, p99 4.07×) with headroom for
   answer encodings that inflate bytes (a model that `\u`-escapes
   non-ASCII roughly doubles a Japanese answer's byte count), and
   below every observed runaway (10× and up). The threshold is a
   policy detail of the ladder, an environment knob like
   `TAGURU_EXTRACT_ESCALATION_FACTOR`, not a flag.
2. **Judged only at the cap.** A complete (`stop`) answer is never
   judged, however large — whether its content is warranted is #853's
   question, not the ladder's. A piece whose answers finish sees no
   behavior change at all.
3. **Demotion still runs first** (ADR 0021): under an `auto`-resolved
   constrained rung, a decoding loop is a plausible cause of exactly
   this signature, demotion is one cheap round, and it restarts the
   piece at the ladder's top where the detection applies again.
4. **The failure is immediate and named**: the source fails with the
   piece's bytes, the answer's bytes, and the configured ratio in the
   message, plus the knob's name — no escalation, no split, no
   further model calls for this piece.
5. **Recorded before acted on**: a `move` record (ADR 0029), action
   `"runaway"`, carrying `piece_bytes` and a new additive
   `answer_bytes` field, lands in the attempts log at detection —
   before the demotion or the failure it leads to. The attempt record
   already carries `length_limited` and the answer itself.
6. **`0` disables** the detection (every `length` keeps taking ADR
   0019/0020's rungs unjudged); anything non-integer is a usage
   error, read whether or not a budget is configured — the
   `TAGURU_EXTRACT_*` convention.
7. **Manifest and checkpoint input only when non-default**: recorded
   as `""` at the default and verbatim otherwise — ADR 0019 §3.4's
   precedent, so a manifest or checkpoint written before this ADR
   matches a default rerun, and a non-default ratio re-extracts like
   any other change to what the run accepts.

## 4. Consequences

- **Behavior change, named in the changelog**: a run that previously
  burned the escalated resend and a full split cascade on a runaway
  piece now fails that source after the first (or, mid-ladder, the
  current) capped round — same terminal state, minus the wasted
  rounds. A legitimately dense piece is unaffected unless its answer
  both hits the cap *and* exceeds 8× the piece's bytes, which no
  successful attempt in the corpus has ever done.
- **The detection is a stopgap for cost, not correctness**: #852's
  prompt rules aim to prevent fabrication and #853's removal will
  judge content; this ADR only stops the ladder from paying for it.
  If #853 lands a content-side judgment, this threshold stays — it is
  the ladder's own economics, not a content check.
- **Tests**: unit tests pin the ratio judgment (boundary, 0 = off,
  default engaged), the demote-before-fail order, the skipped
  escalation, the `runaway` move record, the manifest encoding, and a
  legacy manifest's continued match; an end-to-end test drives a stub
  server that answers a tiny piece with an oversized `length` answer
  and asserts the source fails without an escalated request.
