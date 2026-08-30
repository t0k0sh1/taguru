# 0033. Chunk context: what it is, where it comes from, how it is supplied

- **Status**: Accepted
- **Date**: 2026-08-29
- **Issue**: #782
- **Related**: #780 (the baseline whose findings this answers), #783
  (the measurement that judges it), ADR 0001 §7 (the ladder — a split
  piece is still a chunk here), ADR 0003 §7 (paragraph coordinates —
  unchanged by anything below), ADR 0013 (the occurrence check this
  extends to context text), ADR 0014 (candidate names — the existing
  in-document steering), ADR 0015 (`--vocabulary` — the existing
  ingested steering, and the seed of §3.5), ADR 0027 (steering
  records), ADR 0030 §3.6 (the steps this adds, as it foresaw), ADR
  0031 (replay — context is conversation content, so it is matched
  like the rest)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

"Context" here is 文脈 — what a reader needs to have in mind to read
a chunk — and never a Taguru context (the namespace `--context NAME`
targets). Everything user-facing below says **chunk context** for
the former; the flag, the env, the records, and the step names carry
`chunk_context`/`chunk-context`, so the two cannot be confused on a
command line or in a trace.

What `taguru extract` puts in front of a chunk so the model reads it
knowing where in the document it is and what came before — and what
it deliberately does not. Five kinds of context are defined (§3.1),
two sources (§3.2), and a staged supply strategy with one flag (§3.3–
3.6). Out of scope: the passage and paragraph coordinates (ADR 0003
§7 — context is prompt input only, §3.7), any change to what a
correct association is, the split rung's mechanics, and novel-grade
comprehension (§2, last paragraph).

## 2. Context

`chunk()` packs paragraphs into `--chunk-bytes` at blank-line
boundaries and hands each chunk to the model as `Document 'X', part
K of N` followed by the labeled text. Nothing else. A chunk beginning
at 第五条 does not carry the chapter it sits in, the definitions 第二条
gave, or the fact that 「同法」 three chunks back meant 電子署名法; a
minutes chunk does not know which bill the speakers are on; a
techdoc chunk starting mid-section has no heading. The 2026-08-21
verification (0.9.3) found the document types that *require* reading
the local against the whole — paper 4/10, novel 2/5, techdoc 2/5,
minutes 0/1 — were the ones that failed, and #780 (v0.9.5, after the
ladder fixes made them land) measured what landing without context
buys: the eval set's cross-unit association probes score law 0.06,
minutes 0.00, techdoc 0.00 against paper 1.00 (whose probes are
self-contained by construction), and the spot check's one
context-class error (law-13: an amendment's supplementary provisions
attributed to the wrong act — the act's name was outside the chunk)
is exactly the "which document am I in" gap.

Two things were established before this ADR, and it builds on both:

- **The document is already numbered.** `labeled_document` prefixes
  every canonical paragraph (`crate::paragraph::split`) with `[N]`,
  and the model tags facts by that number. Any context that repeats
  document text can therefore be told apart from the chunk by
  whether it carries a label (§3.7).
- **Steering already exists in two lanes.** ADR 0014's candidate
  names are computed from *this* document, deterministically, and
  offered non-restrictively; ADR 0015's `--vocabulary` names come
  from an *exported context* (already-ingested data) and are
  allowlisted past the occurrence check. Context is the same two
  lanes carrying more than names.

Novels are explicitly not the target. A narrative that withholds its
own meaning until the end (an unreliable narrator, a set-up paid off
chapters later) cannot be read correctly from any local window plus
a synopsis — no context design fixes that, and chasing it would
shape the design around the one type that cannot benefit. The
scenario set keeps one novel (山月記) only to confirm nothing broke.

## 3. Decision

### 3.1 Five kinds of context, defined

| Kind | Definition | Built by | Expected to help |
|---|---|---|---|
| **position** (構造上の位置) | The heading path in force at the chunk's first paragraph: every structural heading above it, outermost first, verbatim (`# 所有権を理解する › ## 参照と借用`; `第二章 › 第三条`; `◆質疑 › ○斎藤委員`) | mechanical, from the document | every type; the minimum |
| **overlap** (隣接テキストの重なり) | The paragraphs immediately before the chunk, verbatim, up to a byte cap — the sentence a chunk boundary cut the antecedent off from | mechanical, from the document | every type; cheapest; not a substitute for the three below |
| **references** (参照先の要約) | For each explicit reference the chunk makes to another structural unit (`第三条`, `前項`, `前条`, `§3`, `第2章`, a heading quoted by title), that unit's heading and opening — verbatim opening under stage (a), the unit's summary once (b) exists | mechanical resolution against the structure; summary from (b) | law, techdoc, paper, code |
| **synopsis** (前章までのあらすじ) | For each structural unit *before* the one the chunk starts in, a one-to-two-sentence summary — cut at **structural** boundaries, never at chunk boundaries, and never chained (each unit summarized from its own text, not from the previous summary) | one model pass over the document (stage b) | techdoc, paper, law, minutes |
| **cast** (登場人物/主体の概要) | The document's recurring subjects — people, organizations, products, defined terms — each with a short gloss, listed once for the whole document | the same model pass (stage b); from the ingested lane, their known relations (stage d) | minutes, news, techdoc |

Two of these are the issue's own insistence, kept here as rules:
a synopsis is per *structural unit*, so "structure" (position) is a
prerequisite and a chunk-by-chunk running summary is rejected (its
boundaries are meaningless and its errors compound); and overlap is
the floor, not the design — it is supplied because it is nearly
free, not because it replaces the rest.

### 3.2 Two sources

- **In-document**: everything §3.1 lists is first built from the
  document being extracted — the structure, the openings, the
  summaries. Deterministic where it can be (position, overlap,
  reference resolution), one model pass where it cannot (synopsis,
  cast).
- **Ingested**: what the target context already holds about the
  cast — the earlier minutes' bills and speakers, the earlier
  chapters' definitions. Supplied from the same artifact
  `--vocabulary` already reads (a `taguru export` of the context,
  ADR 0015): for each cast name (or ADR 0014 candidate) that the
  export's concepts contain, that concept's associations, top-K by
  weight. No server round-trip, no import-order surprise beyond the
  one `--vocabulary` already has (an export taken before the earlier
  document was imported simply lacks it), and no new computation
  input: the export's digest is already one.

A live-retrieval variant (`resolve`/`search` against a running
server at extract time) is *not* chosen: it makes an extraction's
prompt depend on server state at an instant no manifest can name,
which breaks resumability (#179), replay (ADR 0031), and the
manifest's "same inputs, same batch" contract at once. An export is
a file with a digest; that is the property that matters.

### 3.3 Staged supply, one flag

`--chunk-context MODE` (env `TAGURU_EXTRACT_CHUNK_CONTEXT`; beside
`--chunk-bytes`, the other knob that shapes what one call sees)
selects how much of §3.1 is supplied. Modes are cumulative:

| Mode | Supplies | Model calls added | Stage |
|---|---|---|---|
| `off` | nothing — today's prompt, byte for byte | 0 | — |
| `structure` | position + overlap + mechanically resolved references, and structure-aware chunk boundaries (§3.4) | 0 | (a) |
| `overview` | `structure` + synopsis + cast from one overview pass | 1 pass over the document (its own ladder) | (b) |
| `ingested` | `overview` + the ingested lane's relations for the cast (needs `--vocabulary`) | 0 more | (d) |

Stage (c) of the issue — reference *summaries* — is `overview`'s
references: once unit summaries exist, a resolved reference carries
the summary instead of the opening. It is not a separate mode.

**Default `off` in the release that ships `structure`.** The mode
is a manifest and checkpoint computation input (`""` when `off`,
the mode name otherwise — the `structured_output`/`candidates`
precedent), so flipping the default would re-extract every existing
manifest's documents on upgrade; and #783 judges the modes against
#780's baseline *before* one is made the default. The default moves
to the winning mode in the release after #783, as its own ADR
consequence line.

### 3.4 Structure is detected mechanically, and it moves chunk boundaries

The `structure` step (ADR 0030 §3.6 foresaw it under that name; after
`read`, before `plan`)
classifies each canonical paragraph as a heading of some level or as
body, dictionary-free and deterministic (ADR 0014's posture):
Markdown ATX headings by `#` count; Japanese statute headings (`第N章`,
`第N節`, `第N条`, a parenthesized 見出し line that immediately precedes
one); minutes structure (`◆` section markers, `○` speaker lines as
the innermost level); numbered headings (`1.`, `1.2`, `§3`). A
paragraph that matches nothing is body. The result — unit index,
level, heading text, paragraph range — is a `structure` trace
record per unit and the input to position, references, synopsis.

`plan` then packs paragraphs as before, with one added preference:
when the next paragraph opens a unit at the outermost level present
in the document and the current chunk is already past half the cap,
the chunk ends here. Boundaries fall on chapters when they can; no
chunk exceeds the cap; a document with no structure chunks exactly
as today. This is a plan-level change (`chunk_sha256` changes), gated
by the mode like everything else.

### 3.5 The overview pass

Under `overview`, the `overview` step (ADR 0030 §3.6's illustrative
`context` row, renamed for §1's reason: document scope, model call,
after `plan`, before `steer`) asks the model, unit by
unit in document order and within the same chunk cap, for each
unit's summary and its cast entries, as JSON. It runs on the same
completions machinery as Stage 1 — same ladder, same attempts records
(`stage: "overview"`), same checkpoint store (keyed by the unit text,
so a resumed run reuses summaries as it reuses chunk outputs) — and
its output is an `overview` trace record per unit. Its cost is
reported apart (the attempts log's stage tells it), so #783 can price
the synopsis.

The overview's answer is never a source of associations. It is
prompt input for the extraction pass and nothing else (§3.7).

### 3.6 How a chunk is told its context

The `annotate` step (chunk scope, no call, after `overview`, before
`steer`) renders one **context block** per chunk and the user message
becomes:

```
Document 'X', part K of N.

Chunk context (this document's own text and structure, for reading part K —
extract facts from part K only; a fact this block states is tagged
with its own [N] paragraph where part K repeats it, and otherwise not
extracted):
Position: 第二章 › 第三条
Cast: 電子署名法 — この法律; 主務大臣 — 認定を行う …
Before: 第一章 — 目的と定義。電子署名・認証業務・特定認証業務を定義する。
References: 第二条 — （定義）この法律において「電子署名」とは…
Preceding text: …前項の規定は、…

[12] 第三条 電磁的記録であって…
```

Rules, each of which a test pins:

1. **Context text carries no `[N]` labels.** The model tags facts by
   label; the block has none, so nothing in it can be tagged, and a
   `paragraph` value can only name a chunk paragraph (ADR 0003 §7
   untouched).
2. **The occurrence check reads the block.** A subject/object that
   appears in the context block but not in the chunk is not a
   fabrication — it is the document's own text (`同法` → 電子署名法 is
   the point). `user_message_document` returns chunk *and* block for
   the check; the `--vocabulary` allowlist still applies on top.
3. **The block is bounded**: a byte cap (default one quarter of
   `--chunk-bytes`), filled in the order position › references ›
   cast › synopsis › overlap, each kind truncated at a paragraph or
   entry boundary, so the chunk itself is never squeezed below the
   cap the operator set.
4. **The block is recorded**: a `chunk_context` trace record per chunk
   (its sha256, bytes, and which kinds contributed with what they
   contributed — unit ids, paragraph ranges, cast names) — so the
   correction tuple (ADR 0028) and a spot check can see what the
   model was shown, and #783 can attribute a gained fact to a kind.
5. **Duplicates fold as they always have**: a fact the model states
   from the chunk that an earlier chunk also stated is one line after
   `merge` (exact-triple folding); paraphrased twins are the
   consolidation audit's business (ADR 0012), not a reason to hide
   context.

### 3.7 Invariants kept

- Passage bytes and paragraph indices are exactly what they were:
  context never enters the passage, the batch, or a locator.
- No association is attributed to context: rule §3.6.1 makes it
  impossible to tag one, and the overview's own answer is not parsed
  for facts (§3.5).
- A mode is a computation input. `off` is `""`, so every manifest
  and checkpoint written before this ADR matches a default rerun;
  any other mode re-extracts, as a changed prompt should.
- `prompt_version` does not bump: under `off` the user message is
  byte for byte what it was, and the mode itself is the input that
  names the changed shape.
- Replay (ADR 0031) matches by conversation content, which now
  includes the block; a recording made under one mode replays only
  under that mode — correct, since the ask differs.
- Novels: nothing in the design is tuned for them; #783 checks 山月記
  is not worse.

## 4. Consequences

- **Delivery in this order**: (1) this ADR with the pipeline table
  update (ADR 0030 §3.6's procedure: `structure`, `overview`,
  `annotate` rows; `pipeline_version` bump); (2) `--chunk-context
  structure` — detection, boundaries, block, occurrence check,
  records, tests; (3) `--chunk-context overview`; (4) `--chunk-context
  ingested`.
  (2) alone answers law-13 and every position/antecedent gap and
  costs no model call, so it ships first; all four land in the same
  release (#782's decision, 2026-08-30), each measured by #783 as a
  mode of its own.
- **#783 measures per mode and per kind**: the `chunk_context` trace
  record's contribution list is what lets a gained probe be credited
  to position, references, synopsis, or cast, and the attempts log's
  `stage: "overview"` prices the overview pass. The default is chosen
  from that table, not here.
- **Cost bound**: `structure` adds bytes (≤ ¼ chunk) and no calls;
  `overview` adds one pass whose piece count is the unit count and
  whose ladder is Stage 1's, so its wall-clock is bounded exactly as
  the extraction's is.
- **A new env is registered where every `TAGURU_EXTRACT_*` is**:
  `src/cli.rs`, `src/config.rs`, `src/extract.rs`'s usage, and
  `docs/extract.html` (the four places ADR 0019's factor taught).
- **The overview's ladder is escalation only, and its failure is not
  the document's** (§3.5's "same ladder" read precisely): the ask has
  no piece to split, so a cut-off answer is resent once at ADR 0019's
  escalated budget and, if cut off again — or refused, empty, or not
  JSON — that ask contributes no synopsis or cast, reported once on
  stderr; context is advisory, so the document proceeds. Under the
  `json_schema` rung the pass sends `json_object`, the extraction
  schema being the wrong shape for this answer. The cast and synopsis
  lines are the overview model's words: §3.6.2's "the occurrence check
  reads the block" applies to the document-text lines only. The unit
  of one ask — a chunk, not a structural unit — and what a failed ask
  leaves in the checkpoint are ADR 0034's, which supersedes §3.5 on
  those two points.
- **`ingested` names its export in the manifest**: the mode value is
  `ingested:<digest>` where the digest is over the relations the
  export offers (each concept's strongest five by |weight|), so a
  changed export re-extracts as a changed vocabulary already does —
  §3.2's "no new computation input" holds for the *source* (the same
  export file `--vocabulary` reads) while the *offered* relations,
  being prompt content, are fingerprinted. The `Known:` line is the
  export's words, left out of the occurrence check like the cast.
- **Not done**: a chunk-by-chunk running summary (rejected, §3.1);
  live retrieval at extract time (rejected, §3.2); novel-grade
  comprehension (out of scope, §2).
