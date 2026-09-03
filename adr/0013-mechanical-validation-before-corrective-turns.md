# 0013. Mechanical validation and removal before corrective turns

- **Status**: Accepted
- **Date**: 2026-08-08
- **Issue**: #496 (S1)
- **Related**: #199, #178, #180, #181, #464, #465, #466
- **Supersedes**: ADR 0001 §8's corrective-turn routing for
  mechanically-judgeable items (partial — everything else in ADR 0001
  stands) / **Superseded by**: ADR 0022 (§3.2–3.3's "shadowing and
  conflicting aliases stay corrective" — they still get their corrective
  turn, but one that leaves them standing now removes them with accounting
  instead of failing the source; everything else here stands), ADR 0036
  (§4's run rule for a name with no ideograph, kana, or hangul — whole
  words and stems count, character pairs do not; everything else here
  stands)

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru extract`'s strict (default) mode handles items a model answer
carries that could never import as answered. Out of scope: `--lossy` (its
contract stays byte-for-byte the pre-#199 drop-and-proceed), the prompt
text (`PROMPT_VERSION` unchanged), #496's other splits (S2 morphological
candidate constraints, S3 extract-time vocabulary resolve, S4 coverage
verification — each gets its own design when it lands), and the SDK
producers (§5 assigns follow-ups; until they land, parse-level issue
naming remains the only cross-producer parity surface).

## 2. Context

ADR 0001 §8 routed every content-level departure into bucket 2: a
path-addressed corrective turn, bounded by `max_attempts`, failing the
source if the model never corrects. That ruling bought integrity — no
silent drop — at a price measured on 2026-08-08 (issue #496, local bench,
3 models × 3 documents, 18 gold facts):

- A 7B model produced an answer with one empty `object` field and one
  self-referential alias. Five corrective turns re-listed the same two
  issues verbatim and changed nothing; the source failed after 53–63 s per
  document. The same config later succeeded on the same input —
  the corrective loop's outcome was nondeterministic where the defect
  itself was perfectly mechanical.
- The items in question carry nothing a correction could save: an
  association without an object asserts no fact; an alias that maps a
  spelling to itself maps nothing; `merge()` would drop both anyway (in
  lossy mode it silently does).
- Fabricated names — a subject or object appearing nowhere in the
  document — are the one departure the corrective turn is structurally
  wrong for: asking the model to "correct" an invention invites a second
  invention.

The observation: **bucket 2 conflated two kinds of invalidity.** A present
but wrong value (a string weight, an oversized name, an alias `kind` of
`"entity"`) is content the model can actually fix. An absent, self-negating,
or unattested item is not — its only correct disposition is removal, and
removal needs no model.

## 3. Decision

**Strict mode inserts a deterministic mechanical pass between parsing and
the corrective turn. Items the pass can judge are removed with explicit
accounting; the corrective turn is demoted to the last resort for issues
removal cannot judge.**

1. **Mechanically removed (Stage 1, per answer)** — `mechanical.rs`:
   - an association or alias element that is not a JSON object;
   - an association with `subject`/`label`/`object` missing, null, or
     empty after trimming; an alias likewise for
     `alias`/`canonical`/`kind`;
   - an alias whose `alias` equals its `canonical`;
   - an association whose `subject` or `object` does not occur in the
     document text (§4). Labels are never occurrence-checked: a relation
     label is vocabulary, often offered by the run's own prompt.
2. **Mechanically removed (Stage 2, per document)** — after any
   cross-chunk corrective turns, so corrective messages' item indices
   still match the replayed answers: an alias whose `canonical` names
   nothing any output's associations contain. Shadowing and conflicting
   aliases stay corrective — both carry real content whose resolution is
   a judgment, not a mechanic.
3. **Still corrective (the last resort)**: wrong-typed fields, oversized
   names, weight business rules, invalid alias kinds, question issues,
   top-level shape damage, shadowing/conflicting aliases, and schema
   domain/range violations (ADR 0009 §11.2). A corrective turn is spent
   only when at least one such issue survives the mechanical pass.
4. **Accounting, not silence** (ADR 0001 §8 bucket 3 restated): every
   removal is named path-first — one stderr line per item, a
   `removed (mechanical validation)` count on the report line, a
   `removed_items` list on the accepting attempt's diagnostics record,
   and a `removed` count on the document record. This is the load-bearing
   difference from `--lossy`, which validates nothing and reports only a
   count.
5. **A failed answer's removals are discarded**: if corrective issues
   remain, the whole answer goes back for correction and the accepted
   attempt's own mechanical pass is the one that counts — removal never
   creates a hybrid of two answers.

## 4. The occurrence check

Deterministic, dictionary-free, language-independent — deliberately far
short of S2's morphological analysis:

- Normalize both sides: drop every Unicode whitespace character,
  lowercase. Spacing is exactly what an extractor legitimately
  normalizes; `"CI テストランナー"` must pass against a document that says
  `"CI の テストランナー"`.
- Pass on verbatim containment. Names of ≤ 3 characters must pass this
  way — too short for partial coverage to mean anything.
- Otherwise, greedy left-to-right cover: at each position take the
  longest run of the name that appears in the document; runs of ≥ 2
  characters count. Pass at ≥ 3/4 coverage. This admits particle-dropped
  compounds (`プール最大接続数` from `プールの最大接続数`, covered 8/8 by two
  runs) and composed objects (`20→100` from a document stating both
  numbers) while rejecting fabrications that share only fragments
  (`MongoDB` against a PostgreSQL document covers 2/7).

Known limit, accepted deliberately: a legitimately *translated* or
*paraphrased* object (`日次` for a document that only says `毎日`) fails the
check and is removed. The extraction benchmark (ADR 0003) gates the
release: recall must not regress on the gold corpus. If measurement shows
this limit biting, the check's thresholds are constants in one module —
and S2/S3 are the principled fix, not a looser regex.

## 5. Consequences

- **Behavior change, named in the changelog**: a strict-mode source that
  previously failed after fruitless corrective turns on removable items
  now succeeds with those items removed and accounted. A source whose
  *only* defects are removable spends zero corrective turns (#496 S1's
  acceptance gate, held by `tests/fixtures/model_output/removed/`).
- **Fixtures**: `repaired/alias_dangling_canonical.json` moved to the new
  `removed/` corpus — in Rust a dangling canonical is no longer a
  corrective issue. The rest of `repaired/` stays the three-producer
  parse-level parity surface, exercised in Rust via
  `interpret_model_output` + `cross_output_issues` exactly as before.
- **SDK follow-ups**: the Python and TypeScript LangChain producers still
  implement ADR 0001 §8 bucket 2 unmodified. Each gets a follow-up issue
  to adopt this ADR's split (same removal classes, same accounting
  vocabulary, `removed/` fixtures shared); until then the producers
  legitimately diverge on what earns a corrective turn, and the parity
  tests assert only the parse-level surface both still share.
- **Checkpoints**: `CheckpointUnit` gains a `removed` list
  (`serde(default)`); a pre-0013 checkpoint file deserializes with it
  empty, which is also semantically right — its units validated fully
  under the old, stricter-or-equal rules. One narrow gap: a pre-0013
  cached unit was never occurrence-checked, and a reused unit is not
  re-checked. Checkpoint files live only until their document lands, so
  the gap closes itself; not worth a fingerprint break.
- **`--lossy` untouched**, byte for byte, including its report marker —
  the explicit opt-out ADR 0001 §12.2 promised stays meaningful.
