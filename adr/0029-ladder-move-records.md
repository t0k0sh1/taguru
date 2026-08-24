# 0029. The ladder's moves are records, and transport retries are counted

- **Status**: Accepted
- **Date**: 2026-08-24
- **Issue**: #791
- **Related**: #784 axes 1/3 (frequency and cost per move → which
  parameter to adjust; #794's table stands on these), ADR 0001 §7
  (the ladder), ADR 0019 (escalation), ADR 0020 (timeout → split),
  ADR 0021 (rung demotion), ADR 0025 (the attempts log these records
  ride in), ADR 0001 §10 (transport retries folded into one attempt —
  kept, now counted), #792 (the aggregation)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How the retry machinery's actions are recorded: the ladder's moves
(escalate, demote, split — the timeout path included, since it takes
the split rung) and the transport-layer retries inside one attempt.
Out of scope: any change to what the ladder does or to stderr's
lines, and the aggregation itself (#792: counts, tokens, and seconds
per move and per document kind; the retry-lost time and tokens are the
failed attempts' own records, joined by `piece_id`).

## 2. Context

The ladder's decisions were stderr sentences; counting "how often do
law documents escalate, and what does each escalation cost" meant
parsing prose. Attempt records carry every attempt's state, tokens,
and seconds — what was missing is what the ladder DID next and why,
and how many transport-layer tries (429/5xx/transport, ADR 0001 §10's
"four retries are one record") hid inside an attempt.

## 3. Decision

### 3.1 `move` records in the attempts log

One `kind: "move"` record per ladder action, written to the
per-document attempts log (ADR 0025) where the attempts it joins
live; the diagnostics sidecar is unchanged. Common fields: `move`
(`escalate` | `demote` | `split`), `run_id`, `piece_id`,
`chunk_index`, `reason` (the stderr line's own words). Per move:

- `escalate` (ADR 0019): `from_max_tokens`, `to_max_tokens` — written
  between the `length`-ended round and the neutral resend.
- `demote` (ADR 0021): `from_rung`, `to_rung` — written when the
  run's rung is actually lowered (the piece then restarts; a pinned
  rung never writes one).
- `split` (ADR 0001 §7 / ADR 0020): `piece_bytes`, `split_cap`,
  `sub_pieces` — reason distinguishes the output cap from a timeout.
  A piece at the split floor writes no move: nothing happened; the
  failure is the attempt record and the source's error.

The ordinary corrective turn is already recorded (`corrects`, ADR
0028) and is not a move. Attempt `state`s are unchanged — a move
record says what followed.

### 3.2 `transport_retries` on attempt records

`ChatClient::complete` counts the failed tries behind each outcome
(429, 5xx, transport errors, torn bodies), and every attempt record —
sidecar and attempts log — carries `transport_retries` (always
present; 0 = clean first try). One attempt stays one record, exactly
ADR 0001 §10's ruling; the count makes the folded retries visible.
On a terminal transport failure the count is the retries spent before
giving up (fail-fast timeouts under the ladder: the tries before the
timeout was returned).

## 4. Consequences

- Per document kind, "how often does each move fire and what does it
  cost" is a fold over the attempts log: moves give the events,
  attempt records (by `piece_id`) the tokens and seconds around them,
  failed attempts the retry-lost cost.
- The observation → parameter table (#794) can cite record fields
  instead of stderr phrasing: escalations → `--max-output-tokens` /
  `--chunk-bytes`; demotions → `--structured-output`; timeout splits
  → `TAGURU_EXTRACT_TIMEOUT_SECS` / `--chunk-bytes`; high
  `transport_retries` → provider/parallelism.
- `attempt` records gain one always-present integer; `move` is a new
  kind, which both streams' consumers were told to filter on.
