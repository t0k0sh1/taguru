# 0040. Terminology: `document` is the user's whole original file; `segment` is what `taguru extract` reads

- **Status**: Accepted
- **Date**: 2026-09-12
- **Issue**: #904 (implements a decision recorded in #851)
- **Related**: ADR 0003 §10 (the range-acceptance posture this rename's
  wire-version bumps follow), ADR 0013 (the occurrence check — unaffected,
  still judges the same text), every ADR numbered before this one (§4
  says how to read their own use of "document")
- **Supersedes**: nothing (no prior ADR ever defined "document" — see
  §2). / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

What the word "document" means when it appears in `taguru extract`'s
output, prompts, and on-disk records, and in `taguru benchmark`'s
public contract (manifest, runs, measurements, differences, retrieval
files) — all of which used it, before this ADR, for the single input
file one `extract` call reads. Not in scope, unaffected by this
decision:

- **Schema document**: `crate::schema::SchemaDocument`, `--schema
  FILE`'s installed content, `PUT/POST /contexts/{name}/schema`'s
  `document` field, MCP `validate_schema`'s `document` argument. A
  different concept that happened to share the word; #904 left it
  alone by name (see its issue body's "非対象" list).
- General English "documented"/"documentation" (a verb/noun, not this
  concept).
- The LangChain SDK connectors' own naming (`ConnectorDocument`,
  `ingest_documents()`, the `DocumentStarted` event) and the LangChain
  library's own `Document` class. #851's full inventory
  (issuecomment-5549203409) flags these as "個別判断" — judged
  case by case, not decided by this ADR.
- `src/api/evidence/rerank.rs`'s external reranker request body.

## 2. Context

`taguru extract` calls its one input file a "document" throughout: the
summary line (`extract: N written, N unchanged, N failed of N
document(s)`), the LLM prompt (`Document '<path>', part K of N:`),
trace/diagnostics/attempts records (`kind: "document"`,
`document_sha256`), and `docs/extract.html` (roughly 180 occurrences).
No prior ADR ever defined the word — it was simply assumed to mean
"one input file" throughout.

That assumption collided with ordinary English during #783's real-corpus
verification: the corpus's paper input was one 1,613-byte file — title,
metadata, and an abstract only — so it was one "document" in extract's
sense, but a report that "one piece of the document was lost" was read
by both a Japanese and an English speaker as "the whole paper was
lost." What a "document" is, in extract's usage, is decided entirely
by how the *user* chose to split their material into files — one
whole paper, or one file per section — and that variability is exactly
what the general English word does not signal. #851 traced this
confusion, inventoried every place the word (and its neighbors
`chunk`/`piece`/`context`/`source`/`batch`) appears across the CLI,
docs, wire formats, and SDKs, and settled the definition below.

## 3. Decision

**`document` now means only the general-English concept: the whole
original file or work a user wants to bring into Taguru. `segment` is
the concept name for one piece a human split a document into by
chapter or section; `segment file` names the physical file that one
segment lives in — the same thing `taguru extract` reads one of, per
invocation.** Translated from #851's decision (2026-09-05, "追記:
`document` / `segment` の定義"):

> - **document** = the user's whole original file they want to ingest
>   (a novel, a technical manual, a paper — one complete piece),
>   matching the general English word.
> - **segment** = one piece a human split a document into by chapter
>   or section. Used as the concept name.
> - **segment file** = the file holding one segment. Used when naming
>   the physical file.
> - The ingest flow is not "feed the whole document in, then split it
>   into chunks" — it is "split the document into segment files
>   first, then hand each segment file to extract." One segment file
>   becomes one source in its context. `chunk` is the width extract
>   shows the model; it never appears in a context or as a source.

And the operating rule (2026-09-05, "追記: 全用語の棚卸しに対する決定"):

> **A segment is always a passively-accepted unit, and a human decides
> that unit 100% of the time.** A human splits a document into
> segments and turns them into `.md`/`.txt` files. There is no
> scenario where the system does the splitting into segments. What the
> system does is split an accepted segment into chunks by paragraph —
> nothing more, nothing less.

Every place `taguru extract`'s inventory (#851, the full tally,
2026-09-05) found "document" meaning the input-file-one-call-reads
concept is renamed to `segment`/`segment file`: the summary line, the
system and user-turn prompts, the candidates/vocabulary/chunk-context
blocks, trace/diagnostics/attempts records, `docs/extract.html`,
`docs/long-running.html`, `docs/getting-started.html`,
`docs/modeling.html`, README.md, `src/llm-protocol.md`, and
`taguru benchmark`'s manifest/runs/measurements/differences/retrieval
files and their docs page.

### 3.1. What changed, mechanically

- **PROMPT_VERSION 5 → 6**: every occurrence of "document" the model
  is shown is now "segment" — the system prompt's framing, the user
  turn's `Segment '<path>'` preamble, and the
  candidates/vocabulary/chunk-context/overview blocks. Already-extracted
  segments re-extract once; the Python and TypeScript LangChain SDK
  prompt mirrors (`_extract.py`, `extract.ts`) moved in lockstep.
- **Wire keys, write-new/read-both**: `kind: "document"` records are
  now written as `kind: "segment"` (readers accept both);
  `document_id`/`document_sha256` fields are now
  `segment_id`/`segment_sha256` (readers fall back to the old name via
  `#[serde(alias = "...")]` or an explicit `.or_else` lookup, per
  field — never both spellings written at once, which would resurrect
  the duplicate-field trap ADR-adjacent code in `src/api/sources.rs`
  already learned from).
- **`taguru benchmark`'s public contract**: `segment.written_rate`/
  `segment.failed_rate` (was `document.*`), `run.segments_written`
  (was `run.documents_written`), `latency.segment_wall_seconds` (was
  `latency.document_wall_seconds`), the CSV `segment_id` column (was
  `document_id`), `measurements.json`'s `segments` section (was
  `documents`), `manifest.json`'s `segments`/`segment_id` (was
  `documents`/`document_id`), `differences.jsonl`'s
  `segment_coverage`/`segment_sha256` (was `document_coverage`/
  `document_sha256`), and `retrieval.json`'s
  `segments_imported`/`segments_failed` (was `documents_imported`/
  `documents_failed`).
- **Rust identifiers**: every function, type, field, and module name
  that named this concept (`expand_documents`/`read_document` →
  `expand_segments`/`read_segment`, `DocumentRecord`/`TraceDocument`/
  `DocumentCheckpoints`/`DocumentInfo`/`DocumentReport` →
  `Segment*`, the module file `src/extract/documents.rs` →
  `src/extract/segments.rs`), and the runtime error/warning strings
  that named it ("document cap", "--redact is off; document text is
  sent", "rename one of the documents", and so on).

### 3.2. Wire-version bumps (ADR 0003 §10)

Every bump below is a **repurposed key, not an added field** — ADR
0003 §10's range-acceptance posture treats that as requiring a stamp
bump even when, as is true for every one of these, no reader currently
gates its behavior on the version number (the old and new spellings
are both accepted regardless of the stamp). The bump is a documented
fact about the file's shape, not a compatibility gate in itself.

| Constant | Before | After | Reader still accepts |
|---|---|---|---|
| `PROMPT_VERSION` | 5 | 6 | N/A — a manifest's `prompt_version` field just stops matching, so the affected segment re-extracts once |
| `BENCHMARK_MANIFEST_VERSION` | 1 | 2 | 1 and 2 (`taguru_benchmark_manifest` is a gated range; `documents`/`documents_root`/`document_order`/`document_id` read via `#[serde(alias)]`) |
| `BENCHMARK_RUNS_VERSION` | 1 | 2 | both `kind` spellings and both `*_id`/`*_sha256` field names (ungated stamp; `Serialize`-only on the writer side, hand-parsed `serde_json::Value` on the reader side) |
| `BENCHMARK_MEASUREMENTS_VERSION` | 1 | 2 | N/A — `Serialize`-only, no reader parses `measurements.json` back |
| `BENCHMARK_DIFFERENCES_VERSION` | 2 | 3 | N/A — `Serialize`-only |
| `BENCHMARK_RETRIEVAL_VERSION` | 2 → 3 (this rename's own bump; the 1 → 2 step was an unrelated pairs-map-collision fix already on `main`) | 3 | N/A — `Serialize`-only |

## 4. Reading every earlier ADR

**Every occurrence of "document" in ADR 0001 through ADR 0039 means
what this ADR calls `segment` (or `segment file`, when the ADR text is
specifically naming the physical file).** Those ADRs are records of
decisions made under the pre-#904 vocabulary and are not rewritten —
following the same posture ADR 0039 §5 and its predecessors take
toward their own historical text. A reader who needs the current name
for a concept an earlier ADR discusses substitutes `segment` for
`document` at the point of reading; no earlier ADR's *decision*
changes as a result.

## 5. Consequences

- **Behavior change, named in the changelog**: every already-extracted
  segment's manifest entry stops matching once `PROMPT_VERSION` moves
  to 6, so the next `taguru extract` run (with or without `--force`)
  re-extracts it. `taguru benchmark`'s manifest/runs/measurements/
  differences/retrieval files written under this version read the new
  key names; files written before it keep loading (§3.2).
- **Prompt-quality regression check**: measured before/after with a
  local Ollama model (`qwen2.5:14b`, `num_ctx 16384`) over 8 arXiv
  abstracts, one run each. Both prompt versions wrote all 8 segments
  with zero failures and zero corrective turns; paragraph-citation
  attribution stayed 100% valid in both (76/76 associations cited
  vs. 87/87); elapsed time was comparable (256s vs. 282s). The
  association-count difference (76 vs. 87) is ordinary single-run
  sampling variance for a non-zero-temperature model, not a
  statistically separated signal — this is a smoke test against
  catastrophic regression, not proof of equivalent quality at scale.
- **No new terminology invented here**: every definition quoted in §3
  is #851's own decision, verbatim in translation; this ADR adds no
  explanatory language of its own beyond describing what changed and
  why the version numbers moved, per #851's explicit instruction that
  the concepts' definitions come only from that issue's comments.
- **`docs/*.html`'s shared "Segment extraction" nav title**: #851's
  full inventory flagged the "Document extraction" feature name and
  page title as an individually-judged item, not decided by the
  definition above. It is renamed here (to "Segment extraction",
  consistently across every page's nav) because leaving it as
  "Document extraction" while the page's own body now says "segment"
  throughout would recreate the exact half-renamed-vocabulary problem
  #851 exists to fix — not because #851 itself decided the feature
  name.
