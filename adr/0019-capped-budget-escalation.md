# 0019. Capped budget escalation

- **Status**: Accepted
- **Date**: 2026-08-22
- **Issue**: #761
- **Related**: #760, #762, #763 (the same local-LLM verification
  series), ADR 0001 §7 (the ladder this amends), ADR 0013 (the
  corrective-turn demotion this ladder sits beside)
- **Supersedes**: ADR 0001 §7's escalation rung — "escalate once under a
  neutral regeneration ask" with the cap dropped entirely. The rest of
  ADR 0001 (the mechanism matrix, the split rung, the fail-the-source
  terminal, the integrity ruling) is untouched. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

One rung of `taguru extract`'s deterministic ladder: what `max_tokens`
the single escalated resend carries after an answer ends at
`--max-output-tokens`. Out of scope: when the ladder escalates at all
(unchanged — once per piece, only with a budget configured), the split
rung and its floor (unchanged), timeouts reaching the split rung
(#762), partial-success output (#763), and `--structured-output auto`'s
rung choice (#760).

## 2. Context

ADR 0001 §7 chose "escalate once, uncapped" from evidence gathered on
models that finish: a truncated answer is a budget property, and at a
large enough budget none of the measured cells truncated (§3.B/E). The
2026-08-21 local-LLM verification (qwen3 30B-A3B on Ollama, ten
document types) found the premise that does not hold: a model that
**loops** under constrained decoding never ends the uncapped resend
with `length`. It generates until `TAGURU_EXTRACT_TIMEOUT_SECS` (300 s)
expires; the client classifies that as a transport failure and retries
it up to `RETRY_ATTEMPTS` (4) times; the fourth failure is
`RoundOutcome::Failed`, which fails the source outright — the split
rung, the ladder's actual recovery for "this piece is too big", is
never reached. Measured cost: 110 s to the first `length`, then 4 ×
300 s, 10–25 minutes per chunk, and the source lost at the end of it.
It took novel 3/5, techdoc 3/5, paper 6/10, code 4/4, minutes 1/1.

The ladder's own termination guarantee (ADR 0001 G3) was argued from
a finite number of rungs; the uncapped rung made one of them
unbounded in wall-clock, and the timeout that bounds it exits the
ladder instead of descending it.

## 3. Decision

**The escalated resend is capped at `TAGURU_EXTRACT_ESCALATION_FACTOR`
× `--max-output-tokens`, default 2. `0` sends no cap (ADR 0001 §7's
original rung). Everything else about the rung stands: exactly one
escalation per piece, a neutral resend of the base ask, the cut-off
answer discarded and never replayed or salvaged.**

1. **Default 2×, not unlimited.** Twice the budget is enough headroom
   for an honestly-longer answer (the operator set the budget for the
   typical case; the resend is for the long tail), and small enough
   that a looping model hits `length` within about two budgets' worth
   of wall-clock — then the existing split rung takes over. The factor
   is an environment knob like `TAGURU_EXTRACT_MAX_ATTEMPTS`, not a
   flag: it is a policy detail of the ladder, not a per-run request
   parameter the way the budget itself is.
2. **`0` restores the uncapped resend** for operators whose provider is
   known not to loop and whose answers routinely need more than 2×.
   Opt-in, because the failure mode it re-enables is silent for
   minutes and loses the source; the capped default fails safe.
3. **A usage error for anything else**, read whether or not a budget
   is configured — the `TAGURU_EXTRACT_*` convention: a typo is never
   a silent fallback.
4. **Manifest and checkpoint input only when it can matter**: the
   factor is recorded as `""` when no budget is configured (the rung
   never runs) or when it is the default, and verbatim otherwise. A
   manifest or checkpoint written before this ADR matches a default
   rerun — the "new field defaults to the value that changes today's
   behavior least" precedent `structured_output`/`lossy`/`candidates`
   set — and a non-default factor re-extracts like any other change
   to what the model was asked.
5. **Diagnostics show the cap actually sent**: the resend attempt's
   `requested_max_tokens` carries the factored value (previously
   absent), so a run's ladder is reconstructible from the sidecar.

## 4. Consequences

- **Behavior change, named in the changelog**: a budgeted run whose
  escalated resend previously succeeded only because it was uncapped
  now splits instead when 2× is still not enough. That is the split
  rung doing its job; the operator who needs the old behavior sets
  the factor to 0 (or higher than 2).
- **Timeouts still do not split** (#762) — this ADR removes the path
  that turned a length problem into a timeout problem; it does not
  change what a genuine timeout does.
- **Tests**: the ladder's unit and end-to-end tests pin the resend at
  2× the budget and cover factor 3, factor 0, the usage error, the
  manifest encoding, and a legacy manifest's continued match.
