# 0038. A deterministic sensitive-content gate: redact before the model, refuse at import

- **Status**: Accepted
- **Date**: 2026-09-05
- **Issue**: #809
- **Related**: ADR 0013 (mechanical removal and its accounting — the
  posture this reuses), ADR 0014 (dictionary-free segmentation — the
  same "deterministic, no model, no dependency" constraint), ADR 0024
  (loss records — the one field this deliberately does not copy), ADR
  0025 (the attempts log that keeps every prompt in full), ADR 0023
  (the trace file the redaction record rides on), ADR 0030 (manifest
  inputs), ADR 0001 §12.2 (default-off discipline), ADR 0010 §3 and
  #733 (taguru-code's gitignore boundary — the one gate that exists
  today), ADR 0009 (schema `mode` — the server-side enforcement
  precedent named for the follow-up)
- **Supersedes**: nothing. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

Keeping secrets and pattern-recognisable personal data out of the
knowledge graph, the passage store, and the records `taguru extract`
writes beside its batches — by a deterministic, dictionary-free,
model-free judgment, in the spirit of ADR 0013's mechanical removal
and ADR 0014's segmentation. Two layers are decided here: redaction of
the document text before anything reads it (`taguru extract`), and
refusal of a batch that carries such content (`taguru import`, local
and `--url`, client side). Out of scope, each named in §5 as its own
follow-up: server-side refusal on the HTTP import path, the SDK
producers (Python / TypeScript LangChain), and any judgment that needs
context rather than a pattern — names, addresses, "this number is a
salary". The rule set's exact regular expressions are the
implementation's (a versioned constant); this ADR fixes what the set
covers and what it deliberately does not.

## 2. Context

`extract` reads a document whole, sends it to a model, validates the
answer, and writes the batch: associations, aliases, questions, and
the document as its `passage`. Nothing between the file and the batch
asks whether a line is an API key or a personal e-mail address. If
the document holds one, four things happen, in this order: the text
reaches the model endpoint (an external one, unless the model is
local); it lands in the attempts log in full (ADR 0025 — by design,
"no more exposed than the batch beside it"); it lands in the batch's
`passage` and every checkpoint; and on import it becomes a passage
the search API returns to any caller, and possibly a concept name in
the graph. There is no layer where it stops.

taguru-code has the one boundary that exists: a gitignored file is
never in the sync universe (ADR 0010 §3), and #733 closed the hole
where a newly-ignored file was re-imported. That gate is structural
(the repository already says what is secret). Documents have no such
declaration, so the judgment must come from the content itself.

The constraints are ADR 0014's: the same answer on every run for the
same input (manifests and checkpoints key on inputs; a judgment that
drifts re-extracts for nothing), no model, no dictionary, no new
dependency. Under those constraints the reachable set is what a
pattern can recognise: known token formats, key-block delimiters,
credential assignments, e-mail addresses, phone numbers, and numbers
that carry their own check digit. Names and addresses are not in the
reachable set and are not pretended to be.

The redaction cannot stop at the prompt. ADR 0025's premise — the
attempts log exposes nothing the batch does not — holds only if the
masked text *is* the document from the reader's first byte onward:
the prompt, the passage, the checkpoint, the trace, and the log must
all see the same text, or the secret survives in whichever one was
left verbatim.

## 3. Decision

**With `--redact` on, the document `extract` works on is the redacted
text — masked at read, before chunking, candidates, the prompt, the
passage, and every record. Each redaction is accounted for by rule and
paragraph, never by content. `taguru import --refuse-sensitive`
refuses a batch that carries a match, naming the path and the rule.
Both are off by default; the rule set is a versioned constant plus an
optional user file, and the version is a manifest input.**

### 3.1 The rule set is fixed, versioned, and named `redact1`

Two groups, selectable: `--redact` alone means both; `--redact
secrets` or `--redact pii` selects one.

`secrets` — formats whose shape identifies them:

- provider access tokens by prefix and length: AWS access key ids
  (`AKIA`/`ASIA` + 16), GitHub tokens (`gh[pousr]_`, `github_pat_`),
  OpenAI-style (`sk-`), Slack (`xox[abpr]-`), Google API (`AIza`);
- a private-key block from its `-----BEGIN … PRIVATE KEY-----` line
  through the matching `-----END … -----` line, as one match;
- a JSON Web Token (three base64url segments, the first two decoding
  to `{`);
- a credential assignment — `password`, `passwd`, `secret`, `token`,
  `api_key` / `api-key` / `apikey`, case-insensitive, followed by
  `=` or `:` — masking the **value only**, so `password = «…»` still
  reads as a configuration line;
- an `Authorization: Bearer|Basic <value>` header, value only;
- URL userinfo (`scheme://user:secret@host`), the secret only.

`pii` — patterns with a shape or a check digit:

- e-mail addresses;
- phone numbers: Japanese fixed, mobile, and toll-free forms with
  their separators, and international `+CC …` forms; a bare digit run
  never matches (it is a date, a law article, a price);
- 個人番号 (the 12-digit "My Number") when the official check digit
  validates;
- payment card numbers of 13–19 digits in the known issuer ranges,
  grouped by spaces or hyphens or unbroken, when the Luhn check digit
  validates.

Deliberately **not** in `redact1`, each for a stated reason:

- high-entropy strings: every SHA-256 in a technical document is one,
  and the judgment is a threshold, not a shape — a false-positive
  source with no floor;
- names, postal addresses, dates of birth, account numbers without a
  check digit: context, not pattern; a rule would mask nouns at random
  and still miss most of them;
- IP addresses and hostnames: infrastructure documentation is made of
  them.

A user file (`--redact-rules FILE`, one `name<TAB>regex` per line,
applied after the built-ins) extends the set; the file's SHA-256
joins the version (§3.5), so a changed file re-extracts exactly as a
changed built-in would. A rule name is `[a-z0-9_]+` (it is written
into the placeholder) and may not repeat a built-in's.

**Matching is per paragraph, in a fixed order, and non-overlapping.**
The scanner runs every rule over one paragraph at a time (ADR 0003
§7's splitter, the same one the batch's `paragraph` locator uses),
so no match — built-in or user — can span a paragraph separator,
whatever the regex says; a private-key block that a blank line has
split is two matches, one per paragraph. Within a paragraph the
candidate matches of every rule are collected, ordered by start
offset ascending, then length descending, then rule order (built-ins
in the order §3.1 lists them, then the user file's lines in order),
and accepted greedily: a candidate overlapping an already accepted
match is dropped. One placeholder per accepted match, and the counts
in §3.6 are counts of accepted matches — so a bearer token that is
also e-mail-shaped is one `authorization` redaction, not two.

### 3.2 The placeholder

A match is replaced in place by
`«redacted <rule> <4 hex>»` — the rule's name and the first four hex
digits of `SHA-256(document_sha256 ‖ matched bytes)`. The tag lets two
occurrences of the same key read as the same thing and two different
keys as different things, and is deterministic per document without
being a global fingerprint of the secret (the document hash salts it;
sixteen bits identify nothing). Four digits are the *minimum*: when
two distinct matched byte strings under the same rule in the same
document share their first four digits, every placeholder of that
rule in that document is written with the shortest prefix length that
tells all of the rule's distinct matches apart (five digits, six, …
up to the full digest) — chosen from the document's complete match
set, so the length is deterministic for the document, the same secret
still reads as the same placeholder everywhere, and two secrets never
share one. The guillemets are outside every
script the segmenter (ADR 0014) treats as word characters and outside
the prompt's own `[N]` paragraph labels, so a placeholder is never a
candidate name and never looks like a label. The replacement holds no
newline: paragraph N of the redacted text is paragraph N of the
original file, so every paragraph coordinate the batch, the trace, and
the passage store carry still points into the file a person has.

A document may already contain the placeholder form — a passage
exported from a redacted run and fed back in, or a document quoting
one. The scanner recognises `«redacted <rule> <4 hex>»` as its own
(rule `preexisting`), leaves it byte for byte, and counts it apart in
§3.6's accounting, so a run reports what it masked and what arrived
masked as two numbers; §3.3's placeholder removal applies to a
pre-existing placeholder exactly as to a new one.

### 3.3 The redacted text is the document

Redaction runs in `read_document`'s result, before anything else
touches the text. Consequently, with no further mechanism:

- the prompt (every chunk, the chunk context of ADR 0033/0034, the
  candidates of ADR 0014, the Stage 2 corrective turn) shows the
  placeholder;
- the batch's `passage`, the checkpoint units (keyed by the chunk's
  hash — of the redacted chunk), the trace's `paragraph`/`loss`
  `text`, and the attempts log's `messages` all hold the placeholder;
  ADR 0025's premise is kept because there is no verbatim copy
  anywhere under `--out`;
- the occurrence check (ADR 0013 §4) rejects a subject or object that
  is the secret itself — it does not occur in the text the model was
  shown — so a model that "remembers" a key from elsewhere cannot
  write it into an association.

Two additions close the gaps that rule leaves:

- an association whose subject or object is, or contains, a
  placeholder is removed mechanically (ADR 0013 §3 item 1's list
  gains `rule: redacted_placeholder`, with the usual accounting) — a
  placeholder is not an entity;
- before the batch is written, the output is scanned with the same
  rule set as §3.4 does; a hit (a label carrying an e-mail address,
  say — labels are never occurrence-checked) fails the document with
  the path and the rule, the "extract cannot produce a file import
  would reject" invariant extended to this gate.

The manifest's `sha256` stays the hash of the file's bytes: it
answers "did the file change", and the redaction version (§3.5)
answers "did the reading change". Coverage (ADR 0016) and every other
pure function of the document text take the redacted text.

### 3.4 `taguru import --refuse-sensitive`

The same rule set — the built-ins, both groups, plus the same
`--redact-rules FILE` when given — run over each batch's `passage`,
association `subject`/`label`/`object`, alias spellings, and question
text before the batch is applied locally or packed for `--url`. A hit refuses the
**batch** (the unit import already refuses on — a partial batch is
never written), as issues in the shape `schema/check` already emits:
`batches[3].passage` / `batches[3].associations[7].object`, `rule`,
and the paragraph for a passage hit — never the matched text. The
command's summary counts refused batches under a `sensitive` reason
beside the existing ones. Import never rewrites content: it validates
or refuses, so the fix is to re-extract with `--redact` (or edit the
batch), not for import to mask on the way in.

### 3.5 Determinism, the manifest, and replay

`redaction` is a computation input of ADR 0030's manifest and of the
checkpoint fingerprint: `""` when off (existing manifests keep
matching default runs, the `candidates`/`structured_output`
precedent), `redact1` when on with the built-ins, the group selection
folded in (`redact1:secrets`), and with a user file the file's full
SHA-256 appended (`redact1+<64 hex>`) — the whole digest, so two rule
files never share a version by a truncated prefix. A manifest entry
or checkpoint written before this ADR has no `redaction` field and
reads as `""` (ADR 0024 §3.4's posture for a missing field), so it
keeps matching a default run and is invalidated by the first
`--redact` run over the same document; the implementation pins this
with a pre-ADR fixture. Toggling, changing the group, editing the
rules file, or a new built-in version each re-extract. `--replay strict`
replays what was sent — the redacted prompt — and a version mismatch
is a manifest mismatch, not a replay failure.

### 3.6 Accounting: rule and place, never content

Every redaction is recorded, path-first as ADR 0013 §3 item 4
requires, in three places:

- one stderr line per document: `redacted N match(es):
  aws_access_key ×2 (paragraphs 4, 9), email ×1 (paragraph 12)`;
- a `redaction` count on the report line beside `removed`;
- in the trace (ADR 0023), one `kind: "redaction"` record per match
  after the `document` record: `rule`, `paragraph`, `placeholder`,
  `bytes` (the match's length). **No `raw`.** ADR 0024's "every loss
  keeps its original text" is the right posture for an item the model
  wrote; it is the wrong one for the thing this record exists to keep
  out of the file. The paragraph number is the address a person needs
  to look at the original.

The diagnostics sidecar (ADR 0001 §10, metadata only) is unchanged.

### 3.7 Default and the endpoint notice

Both controls are off by default (ADR 0001 §12.2): a masked passage is
a changed passage, and an example key in an AWS how-to is a legitimate
false positive whose masking a user must have asked for. The default
is not a claim that the layer is optional in practice: when
`TAGURU_EXTRACT_URL` names a non-loopback host and `--redact` is off,
`extract` prints one line at the start of the run —
`note: --redact is off; document text is sent to <host> as written` —
once, not per document, so an operator sending a corpus to a hosted
model has been told exactly once where the text goes.

## 4. Consequences

- **Removed from the exposure chain**: with `--redact`, a matched
  secret is never sent, never logged, never stored, never returned by
  search. Without it, `--refuse-sensitive` still keeps a batch
  carrying one out of the graph — a second, independent line.
- **Not removed**: what the patterns do not cover (§3.1's exclusions).
  The docs say so in the same breath as the flag; a user who needs
  names masked needs a model-backed step and this ADR does not offer
  one.
- **The redacted text is a different document** for every consumer
  that hashes or cites it: anchoring (#793) judges the batch against
  the redacted passage, which is the passage it has; a reader
  following a paragraph number to the original file sees the secret
  there — that is the file, not the record.
- **Placeholders are stable across runs of the same file** (the tag is
  salted by the document hash) and unstable across an edit of that
  file — a changed document re-extracts anyway.
- **Candidates and vocabulary**: the segmenter never offers a
  placeholder (§3.2); a label vocabulary accumulated from a redacted
  run therefore cannot carry one either.
- **Rust-only** (ADR 0014's precedent): the LangChain producers gain
  nothing until a follow-up mirrors the module, and `import` on the
  server does not refuse — a producer that posts directly to the HTTP
  API bypasses §3.4. Server-side enforcement is a schema-`mode`-shaped
  decision (off/warn/refuse per context) and gets its own ADR.
- **False positives are visible**, not silent: every redaction is on
  stderr with its rule and paragraph, so a masked product code is
  found on the first run and answered by narrowing the group or
  adding a user rule — never by the tool guessing.

## 5. Follow-up issues

| Issue | Title | Implements |
|---|---|---|
| #881 | `sensitive` module: the `redact1` rule set, placeholder, scan and mask as pure functions | §3.1, §3.2 |
| #882 | `extract --redact [secrets\|pii]`: mask at read, placeholder removal and output scan, manifest/checkpoint input, redaction records, endpoint notice | §3.3, §3.5, §3.6, §3.7 |
| #883 | `import --refuse-sensitive`: batch-level refusal with paths, local and `--url` | §3.4 |
| #884 | `--redact-rules FILE`: user rules, version folding, docs | §3.1 (user file), §3.5 |

Deferred, named here so they are not forgotten: server-side refusal
on `POST /import` (own ADR); SDK producer parity (Python / TypeScript
LangChain).
