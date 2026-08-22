# 0022. An alias the corrective turn cannot fix is removed, not fatal

- **Status**: Accepted
- **Date**: 2026-08-22
- **Issue**: #763
- **Related**: #758 (the cross-document half of the same ruling), ADR
  0013 (the mechanical/corrective split this extends), ADR 0001 §8
  (the correction taxonomy and the integrity ruling this keeps), ADR
  0012 §4 (the consolidation audit that proposes what a removed alias
  would have asserted), #179 (checkpoints — the resume this makes
  visible)
- **Supersedes**: ADR 0013 §3.2–3.3's "shadowing and conflicting
  aliases stay corrective" and ADR 0001 §8.2's "if still invalid, fail
  the source" — for alias items after the Stage 2 corrective turn only.
  / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What `taguru extract` does with a Stage 2 (cross-chunk) alias issue
that its one corrective turn leaves standing, and what a failing
document tells the operator about its checkpoints. Out of scope:
Stage 1 per-answer issues (unchanged — a corrective budget, then the
source fails), association-level Stage 2 issues (schema domain/range,
unchanged — fatal), `--lossy` (unchanged), and any "write the chunks
that succeeded" mode (rejected below).

## 2. Context

The 2026-08-21 verification lost a Wikinews article to one alias:
`aliases[0].alias: names something the associations already contain`,
reproduced deterministically on every rerun, the corrective turn
answering with the same alias each time. ADR 0013 §3.3 kept shadowing
and conflicting aliases corrective because "both carry real content
whose resolution is a judgment" — true of the *turn*, which is still
spent — but after that turn fails, ADR 0001 §8.2's "fail the source"
throws away every association the document held to protect one
spelling hint. #758 already ruled, for the cross-document case, that
an alias is a spelling variant and never a fact, that removing one
loses nothing the consolidation audit cannot propose later, and that
the removal belongs to the mechanical pass with ADR 0013's accounting.
The in-document case after a failed correction is the same judgment,
one turn later.

The same verification also read a late-chunk failure as "the
successful chunks are lost" and reached for `--force`. They were not
lost — #179's checkpoints keep every completed unit and a plain rerun
resumes from them — but nothing on the failure line said so, and
`--force` is precisely the flag that discards them.

## 3. Decision

**After Stage 2's one corrective turn, an alias issue still standing
is removed with ADR 0013's accounting; the document proceeds to merge.
An issue still standing about anything other than an alias item fails
the source as before. A failing document's line names how many units
are checkpointed and that a rerun without `--force` resumes from
them. There is no partial-batch mode.**

1. **The corrective turn is still spent first.** ADR 0013's reasoning
   stands: a shadowing or conflicting alias can carry content the
   model can re-judge (rename the association, drop the alias, fix
   the mapping). Only what survives that judgment is removed.
2. **Alias items only.** The re-check after the turn partitions what
   still stands by the issue's path: `aliases[i]…` is removable,
   anything else — a schema domain/range violation on an association,
   the one non-alias issue Stage 2 can raise — is content and keeps
   ADR 0001 §8's ruling. An output with any non-alias issue fails the
   source with those issues named; its alias issues are not removed
   first, so the diagnosis is what the model was asked and refused.
3. **Removed highest-index first, re-checked until clean.** Each pass
   removes the flagged aliases of an output from the back so every
   recorded path still names the alias its issue did; the re-check
   then runs again, since removing an alias can only shrink the issue
   set, and stops when nothing stands.
4. **Accounted exactly as ADR 0013 §3.4**: one stderr line per removal
   (`…: removed: aliases[0].alias: names something the associations
   already contain — still so after the corrective turn; removed`),
   the report line's `removed (mechanical validation)` count, and the
   document record's `removed` count. Never silent.
5. **The failure line carries the resume.** When a document fails with
   units checkpointed, the message ends with `(N extracted unit(s) are
   checkpointed — a rerun without --force resumes from them)`. Nothing
   else about checkpoints changes.
6. **No partial batch.** ADR 0001 §8.3's integrity ruling — never
   import a subset of a source's knowledge-bearing items while
   reporting success — is kept whole. A batch is the source's
   complete, valid truth or nothing; the checkpoints already make the
   retry cost only the failed part, which is what a partial-batch mode
   would have bought, without the "partially done" manifest state and
   the batch rewrite it would need.

## 4. Consequences

- **Behavior change, named in the changelog**: a document that used
  to fail on an uncorrectable alias now lands without that alias, the
  removal reported. A run's `removed` counts can rise by these.
- **Stage 2's result is a removal list**, not `()`: the caller folds
  it into the same accounting the Stage 1 removals and the dangling
  prune use.
- **Tests**: the partition and ordering of the removal are pinned
  directly (alias vs non-alias issues, duplicate flags, out-of-range
  indices, chunk prefixes); end to end, a shadowing alias the
  corrective turn repeats lands the document with the removal named,
  and a late-chunk failure names its checkpoints and a rerun re-asks
  only the failed chunk.
