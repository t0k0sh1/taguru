# 0021. Run-time demotion of an `auto`-resolved structured-output rung

- **Status**: Accepted
- **Date**: 2026-08-22
- **Issue**: #760
- **Related**: #761 / ADR 0019 and #762 / ADR 0020 (the sibling
  findings from the same verification, whose ladder steps this sits
  in front of), #763 (the rest of the series), ADR 0001 §6 (the probe
  this amends) and §7 (the ladder)
- **Supersedes**: ADR 0001 §6's "resolved once per run, never
  re-derived per chunk" — for `auto` only. Pinned modes, the probe
  itself, the mechanism matrix, and the split rung stand. /
  **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What `taguru extract --structured-output auto` does when the rung its
startup probe verified turns out not to hold on a real document. Out
of scope: the pinned modes (`json-schema`, `json-object` — the
operator's choice, unchanged), the probe's own design, and
re-promotion (a demoted run stays demoted).

## 2. Context

ADR 0001 §6 resolves the rung once: probe the endpoint with a tiny
ask carrying the exact extraction `response_format`, keep the
strongest rung the answer verifies, never re-derive per chunk. The
2026-08-21 local-LLM verification found the gap between "honored on
a 256-token probe" and "usable on a document": Ollama with qwen3
30B-A3B answered the json_schema probe in the canonical shape, then
on a 1.9 KB English abstract generated 14,500–15,000 tokens of
schema-shaped output without stopping (36 tokens/s, seven minutes,
the serving slot occupied and every later chunk stalled behind it).
The same chunk under `off` finished in 25 s with 19 associations.
Constrained decoding is what loops; the rung is the fault.

ADR 0019 and ADR 0020 make that loop *terminate* — capped escalation
ends it with `length`, a timeout descends to the split rung — but
termination alone is ruinous here: every sub-piece loops the same way,
so a 24 KiB chunk walks the split tree to the 512-byte floor, ~110
pieces each burning the full escalated budget or the full timeout.
When the rung is the fault, the only cheap recovery is to change the
rung.

## 3. Decision

**Under an `auto`-resolved rung, a piece that exhausts the ladder —
`length` at the budget and again at the escalated resend, or a
timeout — first demotes the run one rung (json_schema → json_object →
prompted JSON), reports it on stderr, and restarts that piece at the
ladder's top. Only a piece that exhausts the ladder with nothing left
to demote takes the split rung. The demotion is run-wide and never
reversed. Pinned modes never demote.**

1. **One rung at a time.** json_object is also constrained decoding
   and can loop too, but on a backend where it does not, it keeps
   JSON syntax guaranteed; skipping straight to prompted would throw
   that away on every demotion. Two demotions at most per run.
2. **Run-wide, not per piece.** The finding is about the backend's
   behavior on this model, not about one chunk: a rung that looped
   once will loop on the next chunk and the next document. Per-piece
   demotion would pay the full ladder on every piece before finding
   that out again. The rung therefore lives behind a mutex the
   `--parallel` workers share; a worker mid-round keeps the rung it
   started under and its next round reads the demoted one.
3. **Demotion is judged against the rung the piece failed under.**
   Two workers that each exhaust the ladder under json_schema demote
   the run once: the first moves it, the second finds the run already
   past where it failed and simply restarts under the new rung. The
   bookkeeping is a compare-and-demote, not a counter.
4. **Restart, do not split.** The piece was never shown to be too big
   — it was shown to loop. It re-enters the ladder whole, at the
   budget round, under the new rung; if it then exhausts the ladder
   under prompted JSON it splits exactly as ADR 0001 §7 / ADR 0020
   say, because at that point size is the only explanation left.
5. **Only `auto`.** A pinned rung is the operator's statement about
   the backend; a pinned rung that loops splits to the floor and
   fails the source with the named diagnosis, and the docs say to pin
   `json-object` or `off` for a model known to loop.
6. **Never reversed, always reported.** One stderr line per demotion
   names the source, the rung left, the rung entered, and why. The
   manifest keeps recording the REQUESTED mode (`auto`) — the
   computation input is what the operator asked for, as before; which
   rung carried which chunk is a run-time fact the log shows.
7. **Validation stays the authority.** A demotion gives up only the
   wire's shape guarantee; every answer still passes the full
   parse/merge contract (ADR 0001 §4), so correctness does not depend
   on the rung. What a false demotion costs is the constrained
   decoding's syntax-failure elimination on the rest of the run — an
   acceptable price against the alternative, which is the run not
   finishing.

## 4. Consequences

- **Behavior change, named in the changelog**: an `auto` run that
  would have split a looping chunk to the floor now demotes and
  finishes; an `auto` run whose chunk was genuinely too big for the
  json_schema rung's budget is also demoted, once, before it splits —
  the rest of that run extracts under json_object. The startup
  resolution line and the demotion line together tell an operator
  which rung carried which part of a run.
- **`LadderConfig` is no longer plain data**: the rung is read through
  `rung()`/`response_format()` and changed only through
  `demote_from`. The legacy path (no ladder) is untouched.
- **Tests**: unit tests pin the three-step demotion with the restart
  at the budget round, run-wide carry-over to the next piece, the
  pinned rung splitting instead, a timeout demoting before it splits,
  the bottom rung still splitting, and the rung table and
  compare-and-demote guards; an end-to-end test drives `auto` from a
  verified probe through a demotion and into the next document.
