# 0036. Occurrence by words and stems for sparse-script names

- **Status**: Accepted
- **Date**: 2026-09-04
- **Issue**: #853
- **Related**: #852 (prompt-side suppression of the same phenomenon),
  #854 / ADR 0035 (ladder-side detection of it), #783 (the field run
  that observed it), ADR 0013 §4 (the occurrence check this amends),
  ADR 0015 (the `--vocabulary` allowlist, unchanged), ADR 0016
  (coverage, which reuses the check), ADR 0033 §3.6 rule 2 (the
  context block is part of the haystack), ADR 0024 (loss records)
- **Supersedes**: ADR 0013 §4's covering-run rule for a name that
  holds no dense-script character (partial — the rest of ADR 0013,
  and §4 for every other name, stands). / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru extract`'s occurrence check judges a subject or object
written entirely in a sparse script — Latin, Cyrillic, Greek, and any
other script where the unit of meaning is the space-separated word,
not the character. Out of scope: relation labels (ADR 0013 §3.1's
ruling stands, §3.4 below says why); names holding an ideograph, kana,
or hangul (unchanged); an oversized object (`MAX_NAME_BYTES` is a
corrective issue, not this check's); the prompt (#852) and the ladder
(ADR 0035), the other two guards against the same phenomenon.

## 2. Context

The #783 field run (qwen3:30b, minutes corpus) answered a one-line
piece with 191 associations, 144 of them the same attested subject
(`会計年度任用職員`) negating objects the input never contains —
`a permanent position`, `a fixed-term role`, thirteen spellings from
the product {permanent, fixed-term, regular} × {position, role, job}.
ADR 0035 now stops the ladder from paying for that answer; nothing yet
judged its *content*, and a fabricating answer that finishes under the
cap reaches the mechanical pass as any other.

ADR 0013 §4's check should be that judge: a subject or object that does
not occur in the document is removed. Its covering rule counts every
run of **two or more** shared characters. That unit was calibrated on
ideographs, where one character is roughly one morpheme and two
adjacent shared characters are real evidence. In a Latin-script text
two shared letters are a bigram that any paragraph of the language
supplies, so the rule barely constrains anything — and the haystack is
Latin whenever the document is English, and often partly Latin when a
Japanese document carries code, identifiers, or a context block.

Measured on the verification corpus (every accepted item-stage answer
across the 0.9.5 shakedown, the #780 baseline, and #783's local
reruns: 269 answers, 14,117 subject/object names, 183 distinct
haystacks of which 55 are majority Latin):

- Sixteen fabricated English phrases (#783's objects, plus phrases
  such as `the quick brown fox jumps`, `Kubernetes cluster
  autoscaler`, `MongoDB`) judged against each of the 55 Latin
  haystacks: **788 of 880 pass** under the pair rule (90%). Against
  one arXiv abstract, `a permanent position` covers 18 of 18
  characters.
- The same phrases against the Japanese minutes chunk alone: all
  fail — the check works where it was calibrated. The hole is Latin
  haystacks, not the check's shape.

## 3. Decision

**For a name with no dense-script character, a run of letters counts
toward ADR 0013 §4's 3/4 coverage only as a whole word of the name of
three or more letters, or as a stem of five or more letters; a run
holding a digit or symbol keeps the two-character rule. Names holding
an ideograph, kana, or hangul are judged exactly as before. Nothing
else in ADR 0013 changes.**

1. **Dense and sparse.** A dense-script character is an alphabetic
   character in the CJK block range (ideographs, kana, bopomofo,
   compatibility jamo), the hangul syllables and conjoining jamo (a
   decomposed spelling's, both extension blocks included), halfwidth
   katakana, the CJK compatibility ideographs, or the ideograph
   extension planes. A name
   with at least one such character is a dense or mixed name — `fn定義`,
   `implブロック` — and its Latin fragments are identifiers, not
   function words, so ADR 0013 §4's rule stands for it unchanged. A
   name with none is a sparse-script name.
2. **Word boundaries come from the original name**, read before
   whitespace is dropped: a word starts at the name's edge or after a
   non-alphanumeric character (whitespace, `-`, `_`, `::`) and ends
   likewise. Normalization is unchanged — the check's normal form is
   still whitespace-blind and lowercased, so `--vocabulary`'s
   allowlist keys (ADR 0015) match as before. Only the name's own
   word boundaries are consulted; the document side is still judged
   by containment, so `exit` counts against a document that says
   `exits`. A run of letters never crosses a word boundary of the
   name: the longest document run is cut after the first word end
   inside it, so the cover takes the longest run *within* the word
   (`prediction` counts when the document continues with `heads` and
   the name with `head insertion`) and two adjacent short words never
   assemble a stem neither has (`ab cd ef` against `abcde`). A run
   holding a digit or symbol is left whole.
3. **Why three and five.** Two-letter whole words are the language's
   function words (`of`, `at`, `is`, `to`) — present in any text, so
   they attest nothing; three admits the short identifiers that carry
   real evidence (`std`, `cwd`, `API`, `MIT`). Five letters is the
   shortest stem that survives inflection and pluralization
   (`selects`/`selection`, `recommendation`/`recommendations`,
   `computationally`/`computational`) while no longer being a fragment
   any paragraph supplies. Measured on the same corpus:
   - fabricated phrases passing: **788 → 7** of 880 (the seven are
     compositions of words the document does use — §4);
   - accepted names newly removed: **120 of 14,117** (0.85%, 56
     distinct), every one an English rendering of a Japanese
     document's term (`Copy trait`, `drop function`, `unused memory`
     — ADR 0013 §4's declared translation limit, which #852's prompt
     also forbids), the prompt's own instruction words used as an
     object (`procedure`, `membership`, `next-step`, `first-step`), a
     bare `true`/`false`, or a paraphrase sentence (`code works as
     intended`). No name a document actually spells out was lost.
4. **Labels stay unchecked** (ADR 0013 §3.1). A relation label is
   normalized vocabulary — `内包する`, `is a component of`, `uses` —
   that legitimately never appears in the text, and there is no
   dictionary-free tell separating a fabricated label from a
   normalized one. The predicate is judged through the object: every
   one of #783's 144 carried an object the input never contains, and
   the object check removes the association whole. A fabricated label
   on an attested object is content — the schema's domain/range rules
   (ADR 0009) and the corrective turn keep it.
5. **Removal, no failure threshold.** The issue asked whether a run
   whose removals exceed some share should fail the source. It does
   not: a surviving association is attested by construction, failing
   the source would discard attested facts to punish the removed
   ones, and the cost of a fabricating answer is ADR 0035's concern,
   not this check's. Accounting is ADR 0013 §3.4's, unchanged: one
   stderr line per removal naming the object (`associations[0]:
   object "a permanent position" does not appear in the document
   text`), the report line's `removed (mechanical validation)` count,
   `removed_items` on the diagnostics record, and the trace's loss
   record (ADR 0024) with the item verbatim.
6. **Coverage inherits the rule** (ADR 0016 reuses `name_occurs` with
   a sentence as haystack). An English label or object that "covered"
   a sentence only through shared bigrams no longer does, so English
   documents may show more uncovered sentences. Report-only, so no
   batch changes.
7. **Not a fingerprint input.** As with ADR 0013 itself and ADR 0022,
   the rule is the build's code, not a setting: `PROMPT_VERSION` is
   untouched, manifests and checkpoints do not change, an already
   extracted document is not re-extracted, and a cached checkpoint
   unit is not re-judged (ADR 0013 §5's same gap, closing itself when
   the document lands).

## 4. Known limits, accepted

- **A composition of the document's own words passes.** `adaptive
  exit selection` passes against a document that says `adaptive`,
  `exits`, and `selects`; so does an enumeration object joining such
  phrases with `/`. That is the rule as the prompt states it — "never
  build a subject or object out of words the document does not
  contain" — mechanized; whether the composition is a fact the
  document states is the model's answer to judge, and an oversized
  one is `MAX_NAME_BYTES`' corrective issue.
- **Translations are removed**, now also for Latin renderings of a
  Japanese document that the pair rule let through. ADR 0013 §4
  accepted this limit; the prompt (#852) forbids translating names;
  the consolidation audit (ADR 0012) is where a legitimate
  cross-language twin gets proposed.
- **Scripts without spaces that are not ideographic** (Thai, Lao,
  Khmer) are sparse by this definition and have no word boundaries
  the name can show, so their runs count only as five-letter stems.
  Unmeasured; a corpus in one of them would recalibrate here.

## 5. Consequences

- **Behavior change, named in the changelog**: a strict-mode source
  whose answer carried a Latin-script subject/object built from words
  the input never uses now has that association removed with
  accounting, where the pair rule accepted it. A run whose names all
  come from the document sees no change.
- **Tests** pin the rule: the #853 shape end to end through
  `mechanical_interpret` and the `removed/` fixture corpus; the
  three- and five-letter boundaries one letter apart on each side;
  the original-name word boundary (`std::path` counts, `stdpath` does
  not); digit/symbol runs at two; every dense-script range with a
  mixed name; and the measured fabrications failing against English
  text.
- **SDK producers**: the Python and TypeScript LangChain producers
  still implement ADR 0001 §8 (ADR 0013 §5's follow-ups have not
  landed), so there is nothing to port; when they adopt ADR 0013 they
  adopt this amendment with it.
