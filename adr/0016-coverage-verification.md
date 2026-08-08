# 0016. Coverage verification at extract time

- **Status**: Accepted
- **Date**: 2026-08-08
- **Issue**: #496 (S4)
- **Related**: ADR 0013 (S1 — the precision-side accounting this
  mirrors, and the occurrence machinery it reuses), ADR 0014 (S2 — the
  segmenter that defines a term), ADR 0015 (S3 — the steering whose
  interplay §5 names), ADR 0001 §8 (never-silent-drop, extended here
  from drops to misses)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru extract` detects and reports the sentences a model's
extraction never covered (#496 S4's "カバレッジ検証"). Out of scope:
automatic re-extraction of flagged sentences (§4), the SDK producers,
and any change to what gets written — this control changes reporting
only.

## 2. Context

The 2026-08-08 bench put a number on the systematic recall ceiling:
the mid-size model dropped the SAME fact (`頻度=日次`, a dense list
line) on every run — not noise, a ceiling. S1–S3 attack precision and
consistency; nothing yet makes a recall miss *visible*. Today a
dropped fact simply never exists: no report line, no diagnostic,
nothing for the operator or the benchmark to count. ADR 0001 §8's
never-silent-drop discipline covers items the model produced and the
pipeline removed; a fact the model never produced has so far been
silent by construction.

The ingredients to name it already exist: ADR 0014's segmenter says
deterministically which sentences are dense with names, and ADR
0013's occurrence machinery says deterministically whether a name is
present in a span of text. Composed, they answer "did this sentence
become a triple?" with no model in the loop.

## 3. Decision

**`--coverage` / `TAGURU_EXTRACT_COVERAGE` reports, per document,
every sentence that holds a candidate pair yet is covered by no
accepted association. Report-only, default-off, never a fingerprint
input.**

1. **Coverage unit — the candidate-pair sentence**: paragraphs (the
   server's own canonical split) divide into sentences at terminators
   (。!?, full- and half-width) and line breaks; an ASCII period is
   deliberately not a boundary (it lives inside identifiers —
   `file.rs`, `qwen2.5:14b`), and the cost of missing one is only a
   larger unit, which can only under-flag. A sentence owes coverage
   when ADR 0014's segmenter (`seg1`) finds at least two distinct
   candidate terms in it — one name anchors nothing relational; two is
   the smallest shape a triple can land on. Hiragana-heavy prose and
   lone identifiers are structurally exempt, biasing the check toward
   exactly the dense technical lines the bench saw dropped.
2. **Covered means two of three parts land**: an association covers a
   sentence when at least two of its subject, label, and object occur
   in the sentence (ADR 0013's `name_occurs`, sentence as haystack).
   Two, not three, on purpose: a fact's subject often lives in an
   earlier sentence — a heading, a list introducer — while the
   sentence carries label and object; demanding all three would flag
   precisely the extractions the make-implicit-membership-explicit
   rule produces. Judged against the merged, validated extraction, so
   the answer describes what will actually be written.
3. **Accounting rides ADR 0013's channels**: one stderr line per gap,
   path-first (`uncovered: [paragraph N] <sentence quote, byte-capped>`),
   a count on the document's report line
   (`N sentence(s) uncovered (coverage)`), and an `uncovered` count in
   the diagnostics `document` record. A gap is never a failure: the
   batch is written whole, exactly as without the flag.
4. **Report-only is why it is not a computation input**: the batch
   bytes are identical with and without the flag, so no manifest or
   checkpoint fingerprint carries it — and a manifest-skipped document
   is judged too, from the batch it already has on disk. Coverage of
   any past run is therefore measurable for free: rerun with
   `--coverage` and everything unchanged skips, reports, and calls no
   model.

## 4. Staged on purpose

Re-extracting the flagged sentences (the issue's parenthetical "該当文
だけ再抽出") is the planned upgrade, not part of this decision. It
buys recall recovery at real costs a report does not have: extra model
calls on every false gap, duplicate-assertion weight inflation unless
deduplicated, and a second corrective surface. It is bought the way
ADR 0014 §3.4 buys a heavier segmenter: against a measured corpus
where the gap report demonstrably names recoverable facts at an
acceptable false-gap rate — evidence this control is what produces.
Until then the operator's loop is the remedy: read the gap lines, fix
the document or the model choice, re-run.

## 5. Known limits, accepted deliberately

- **A gap is a lead, not a verdict**: a heading dense with names, a
  table row, or a sentence whose fact the model legitimately attached
  to neighboring sentences' spellings can flag without a real miss.
  The check is calibrated to under-flag (pair threshold, two-of-three,
  period non-boundary), and a false gap costs one stderr line.
- **ADR 0015 interplay**: a `--vocabulary`-steered spelling can differ
  enough from the sentence's own surface form to miss `name_occurs`'s
  coverage threshold, flagging a sentence whose fact WAS captured
  under the context's spelling. The occurrence fallback (3/4 character
  coverage) absorbs the common steering shapes (`山科さん`→`山科`);
  the residue is visible noise, never a wrong batch.
- **Cross-sentence facts**: an association joining entities from two
  sentences covers neither unless two of its parts land in one of
  them. The two-of-three rule absorbs the common anaphora shape; a
  fully distributed fact can still flag its sentences.

## 6. Consequences

- S1 and S4 are the two halves of one accounting: removed (asserted
  but unsupported) and uncovered (supported but unasserted) — both
  named per item, both counts on the same report line, both in the
  diagnostics document record.
- The gap count is a zero-gold recall proxy: comparable across models
  and settings on any corpus, with no annotated eval set — input for
  the default-on decisions S2/S3 still owe and for extract-model
  selection.
- Rust-only, like the other three controls; SDK follow-ups inherit
  S1–S4 together.
- `benchmark extract` does not forward the flag in v1: report-only
  output changes no computation, so fairness and resume gates are
  untouched; a benchmark-side gap metric can ride the diagnostics
  sidecar later without touching `extraction_settings`.
