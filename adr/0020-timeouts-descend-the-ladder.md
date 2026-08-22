# 0020. Timeouts descend the ladder; the chunk cap is a knob

- **Status**: Accepted
- **Date**: 2026-08-22
- **Issue**: #762
- **Related**: #761 / ADR 0019 (the sibling finding from the same
  verification: the escalation rung's cap), #760, #763 (the rest of
  the series), ADR 0001 §7 (the ladder this amends) and §7 D (the
  "caller-selectable unit policy" this finally exposes), ADR 0003 §9.1
  (benchmark records the cap a cell ran under)
- **Supersedes**: ADR 0001 §7's `TIMEOUT` row — "existing transport
  retry policy, unchanged" — for the ladder only; and the fixed 24 KiB
  chunk cap. Everything else in ADR 0001 stands. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

Two things about `taguru extract`: what a completion timeout means to
the §7 ladder, and whether the chunk cap is an operator's choice. Out
of scope: the legacy path (no `--max-output-tokens`, no
`--structured-output` — byte-for-byte ADR 0001's "today's behavior"),
the transport retry discipline for anything other than a timeout,
partial-success output (#763), and `auto`'s rung choice (#760).

## 2. Context

ADR 0001 §7 classified `TIMEOUT` with `TRANSPORT`: transient trouble,
retried four times with jittered backoff, then the chunk fails. The
2026-08-21 local-LLM verification (qwen3 30B-A3B at ~36 tokens/s)
found timeouts that are not transient: a 16 KB statute chunk that
needs more than 120 s to answer needs it every time. Four attempts
cost 4 × 120 s and the source failed; raising the timeout to 300 s
passed. The chunk was simply too big for the hardware's time budget —
exactly what a `length` answer says about the token budget, and for
`length` the ladder already has the right next step: split. There was
no way to reach it from a timeout, and no way to start smaller: the
24 KiB cap was a constant, which ADR 0001 §7 D had already called "a
caller-selectable unit policy for documents known to be fact-dense"
without ever exposing it.

## 3. Decision

**Under the ladder, a completion timeout is a too-big piece: it
descends to the split rung exactly as `length` does — never retried at
the same size, never escalated — and at the split floor fails the
source after one attempt with the timeout named. The chunk cap is
`--chunk-bytes N` / `TAGURU_EXTRACT_CHUNK_BYTES` (default 24576, at
least the split floor of 512), a manifest and checkpoint computation
input when non-default.**

1. **The split is the retry.** `ChatClient::complete` gains a
   per-request `fail_fast_on_timeout`; the ladder's rounds set it, so
   the first timeout returns to the ladder at once instead of after
   `RETRY_ATTEMPTS` same-size attempts. Transport failures, 429 and
   5xx keep their four attempts on every path; Stage 2's cross-chunk
   corrective turn and the structured-output probes keep the full
   discipline too — neither is a piece the ladder could split.
2. **Never escalated.** A larger output cap cannot make a slow piece
   faster; a timeout in the budget round goes to the split rung
   without the ADR 0019 resend, and a timeout in the escalated round
   goes there from where it is.
3. **The floor names the cure.** A piece at the split floor that still
   times out fails the source with the client's timeout message and
   the two knobs that would have helped — `TAGURU_EXTRACT_TIMEOUT_SECS`
   and `--chunk-bytes` — after one attempt, not four.
4. **The chunk cap is a knob, at the floor the split rung already
   has.** Below 512 bytes a chunk could not split at all, so that is
   the minimum for both the flag and the variable; anything else is a
   usage error. Flag over variable, the `--max-output-tokens`
   convention.
5. **A computation input when it can matter.** The cap decides what
   every chunk the model is shown contains, so a non-default cap is
   recorded verbatim in the manifest and the checkpoint fingerprint
   and re-extracts like any other change to the ask; the default is
   recorded as `""` — the "new field defaults to the value that
   changes today's behavior least" precedent — so manifests and
   checkpoints written before the field existed keep matching a
   default run. `benchmark` already records the cap a cell ran under
   (ADR 0003 §9.1) and scrubs every `TAGURU_EXTRACT_*` variable before
   pinning its own, so its cells stay at the default cap unchanged.

## 4. Consequences

- **Behavior change, named in the changelog**: a ladder run whose
  chunk used to fail after four timeouts now splits it; a run that
  happened to succeed on a second same-size attempt (a genuinely
  transient stall) now splits instead of retrying — the split answers
  the same question with smaller asks, at most one timeout per split
  level instead of four per chunk.
- **`--dry-run`'s chunk count honors the cap** and remains a lower
  bound on calls, since timeouts as well as `length` can split at run
  time.
- **Tests**: the ladder's scripted endpoint learned to stall one
  connection; unit and end-to-end tests pin timeout → split with no
  same-size retry and no escalation, the one-attempt floor failure,
  the client's fail-fast switch against the unchanged four-attempt
  default, and the flag/variable's parsing, precedence, floor, dry-run
  chunk count, and manifest encoding including a pre-0020 manifest's
  continued match.
