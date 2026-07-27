# 0004. Retrieval and citation quality gate: CLI shape, execution, artifacts, thresholds

- **Status**: Accepted
- **Date**: 2026-07-28
- **Issue**: #271
- **Related**: #215, #248, ADR 0002, ADR 0003 §11
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru evaluate` measures the public retrieval and citation behavior of
one already-populated context against a labeled `eval.jsonl` and turns the
result into a CI-suitable pass/fail gate — the twelve decisions #271
requires before any implementation issue (#272–#279) can start without
risking a later, backward-incompatible rewrite of the artifact shapes.
ADR 0003 §11 already fixed the outer frame this ADR works inside: #215 is a
quality gate over one corpus (not a cross-corpus comparison, which is
`taguru benchmark search`'s job, #260), it owns rank-based metrics,
citation metrics, graded relevance, per-case thresholds, and a non-zero CI
exit, and it shares exactly one artifact — `eval.jsonl` — with #260. This
ADR does not reopen any of that; it fills the twelve gaps §11 explicitly
left for a follow-up ADR to close: CLI shape, the dataset's version
posture under #215's own extensions, the shared loader's location, the
graph-lane execution path, citation-locator semantics, the threshold
file's format, `evaluation.json`'s schema and versioning, the comparison
mode's input/output, `remote::Api` convergence, failure-detail bounds,
corpus-revision recording, and the structural guarantee that core
evaluation needs no answer-generation LLM.

Out of scope: the metric computation itself and the concrete Rust code
(#272–#279 own that), the CLI's broader flag/environment-variable
consistency audit (ADR 0002 §12.1's #248, which this ADR narrows and
defers to exactly as ADR 0003 did for itself), and any answer-faithfulness
scoring — §12.2 makes not building that in explicit.

## 2. Context

### 2.1 What ADR 0003 §11 already fixed, and the three gaps it left in the wire contract

§11 fixed the boundary prose, the shared artifact's name and stamp
(`eval.jsonl`, `taguru_eval: 1`, equality-checked, header line 1), the
core-vs-extension field split, and the HTTP-driven execution posture for
*both* #260 and #215 (*"Evaluate the public retrieval behavior, not
private scoring helpers"* — an in-process call would mean reconstructing
server state, routing, grant, and filter logic offline, measuring a path
no user takes). None of that is re-decided here.

What §11's own example schema left underspecified, because #260 never
needed to fill it in, are three gaps between what `eval.jsonl` can declare
and what the server can actually do with it:

- `options.sources` names no server field. `SearchPassagesRequest`
  (`src/api/sources.rs:311-329`) is
  `{query, limit, semantic_floor, tags, since, until}` — `tags` is a
  per-source label filter, not a source-id filter, and there is no
  source-id filter on the wire at all.
- The server's time window is half-open `[since, until)`
  (`sources.rs:325-329`), but `eval.jsonl` as specified in §11 can only
  bound the left edge.
- `options.floor` maps to `semantic_floor` only. `resolve`'s
  `dice_floor` (`src/api/resolve.rs:23-24`) has no eval.jsonl
  counterpart, so a case cannot pin the lexical-tier floor it exercises.

§6 resolves all three.

### 2.2 The retrieval surface as it actually exists

Four route families an evaluator needs, and their `Role` (`src/auth.rs`,
`required_role`):

- `POST /contexts/{name}/sources/search` — passage retrieval. Request
  `SearchPassagesRequest` (`sources.rs:311-329`); response `PassagePage`
  `{plan: SearchPlan, hits: Vec<PassageHit>}`. `PassageHit`
  `{source, paragraph, score, text, lanes: PassageLanes}`; `PassageLanes`
  `{bm25: Option<LaneEvidence>, vector: Option<LaneEvidence>}`;
  `LaneEvidence{rank, score}`. `score` is the fused reciprocal-rank
  number when the vector lane ran, raw BM25 otherwise; `paragraph` is
  0-based. `Role::Read`.
- `POST /contexts/{name}/resolve` and `/resolve_label` — cue → concept
  or label name. Request `ResolveRequest{cue, dice_floor, semantic_floor,
  limit}` (`resolve.rs:19-32`); response a bare `Vec<TieredResolution>`,
  `TieredResolution{name, score, tier, kind, gloss}`
  (`resolve.rs:38-52`). Scores are *"ordinal within a tier, never
  comparable across tiers"* (doc comment, `resolve.rs:36-37`). `Role::Read`.
- `POST /contexts/{name}/query` — exact-triple structural lookup.
  Request `QueryRequest{subject, label, object, limit, after}`; response
  `MatchPage{total, matches: Vec<AssociationOut>, plan}` (`api.rs:1176`).
  `AssociationOut{subject, label, object, weight, count, attributions:
  Vec<AttributionOut>}`; `AttributionOut{source, weight, count,
  paragraph, section}` — the graph side's own citation locator, same
  vocabulary as passage citations. `Role::Read`.
- `POST /contexts/{name}/citations` — locator validity. Request
  `CitationRequest{source, paragraph}` (`sources.rs:68-73`, `index` still
  accepted as a pre-#35 alias); response `Citation{text, source,
  section}` (`sources.rs:84-88`) — **not batched**, one HTTP round trip
  per locator, and `section` is *never omitted*: `null` is an assertion
  ("this paragraph is outside every stored section, or the source has
  none"), not an absent key. `Role::Read`.

Wire-type audit relevant to implementability: `MatchPage` (`api.rs:1176`)
derives `Serialize` only; `TieredResolution` (`resolve.rs:39`) derives
`Serialize` only; `Citation` (`sources.rs:84`) derives `Serialize` only.
`mod resolve` is private in `src/api.rs`. None of the four is reachable
today from anywhere outside the axum handler that produces it — §13.3
lists the one-line fix each needs, following the exact precedent
`pub(crate) mod sources;` already set for `PassagePage`/`PassageLanes`.

`Context::recall(cue)` returns every edge incident on a concept, and the
HTTP layer pages it at `clamp(limit, 100, 1000)`
(`src/api/recall.rs:53`; ceiling `api.rs:1006,1011`) — a natural-language
`cue` is not resolved to an exact stored id by `recall` itself, and a
hub concept's expected triple can fall outside the page. `query` pins
subject/label/object exactly and returns `total` with no such risk. §7
uses this to choose `query` over `recall`.

### 2.3 The shared dataset as built for #260

`src/benchmark/evalset.rs` (373 lines) is the loader `taguru benchmark
search` already ships. `EVAL_VERSION: u64 = 1` (`evalset.rs:29`),
equality-checked (`evalset.rs:161-166`: *"taguru_eval must be
{EVAL_VERSION}, got {}"*). `EvalHeader{taguru_eval, name, default_target}`
(`evalset.rs:31-43`), `EvalCase{case_id, query, cues, expected_sources,
expected_concepts, options, expected_labels, expected_associations,
expected_citations}` (`evalset.rs:83-108`), `ExpectedSource{source,
paragraphs, relevance}` (`evalset.rs:48-57`), `EvalOptions{limit, floor,
sources, since}` (`evalset.rs:67-74`) — every struct
`#[serde(deny_unknown_fields)]`. The three #215-only case fields and the
three #215-only option fields are typed `Option<Value>`, declared but
never read except by `carries_215_extension()` (`evalset.rs:76-79,
110-116`), which drives a once-per-run warning
(`evalset.rs:212-220`) that hard-codes the string `taguru benchmark
search` as the consumer name. §6 resolves both the typing and the
naming.

### 2.4 CLI and client audit

There is no argument-parsing framework in this codebase — `src/cli.rs:1-4`
states it on purpose (*"Hand-rolled on purpose — a default `serve`, three
offline subcommands, and one flag do not need an argument framework"*).
Every verb parses its own `&[String]`. The documented exit-code contract
today is `src/cli.rs:6-7`: *"0 success · 1 operation failure (corruption
found, server error) · 2 usage error."* No verb returns any other code.

`crate::remote::Api` (`src/remote.rs:112-137`) is the one HTTP client the
tree has settled on — module doc (`remote.rs:1-15`): *"The one HTTP door
every remote verb walks through."* Six live consumers today: `compact`,
`export`, `communities`, `calibrate`, `import`, `benchmark search`. Its
constructor reads the bearer token immediately
(`remote.rs:124,130,135` — `timeout_global(35s)`, token captured in the
same call), so `load_config` (which sets `TAGURU_API_TOKEN` from a
`--config` file) must run before `Api::new`. `reject_userinfo`
(`remote.rs:41-53`) is called by exactly two of the six consumers —
`export.rs:670`, `compact.rs:243` — not by `calibrate`, `communities`, or
`benchmark search`. `benchmark search` then writes the resolved URL
verbatim into `retrieval.json`'s `InputsBlock.url`
(`src/benchmark/search.rs:279-291`) — so `taguru benchmark search --url
https://user:token@host …` leaks a credential into a shareable artifact
today. §13.1 files the fix as a separate issue; §11 makes sure `evaluate`
does not repeat it.

`warn_on_version_skew` and "print the target before sending anything" are
ADR 0002 §5's rule for *mutating* remote verbs specifically; of the six
`Api` consumers, only `export`, `compact`, and `import` (all mutating)
call it. `calibrate`, `communities`, and `benchmark search` (all
read-only) do not. §11 does not claim precedent it lacks.

The `--config`-then-`default_base_url()` argument-parsing block
(`cli.rs:436-449`, `calibrate.rs:244-257`, `communities.rs:153-166`,
`benchmark/search.rs:191-207`) is copy-pasted at four call sites today.
ADR 0002 §12.1's #248 already owns *"`--config` coverage, other gaps"* as
open follow-up work; §11 declines to add a fifth copy's worth of new
shared-helper design here and instead records `evaluate` as the fifth
site for #248 to eventually collect.

### 2.5 Prior art

`taguru benchmark search` (#260, `src/benchmark/search.rs`, 1613 lines)
already drives retrieval over HTTP against a live server and writes
`retrieval.json`. Its `resolve_expected_source_path`
(`search.rs:911-963`) resolves `expected_sources[].source` against
`manifest.json`'s document dictionary — a resource #215 does not have,
since `evaluate` targets a context that was never built by `taguru
benchmark extract`. §6 gives `evaluate` its own resolution path instead.
Its `HitLocator{rank, source, paragraph}` (`search.rs:1585`) carries no
body text, matching ADR 0003 §9.4's no-corpus-text default for
`differences.jsonl`; §11 keeps that default and tightens it further.
ADR 0001's harness is the more distant ancestor both ADRs already cite;
nothing further to add here.

## 3. Options considered

### A. CLI: top-level `taguru evaluate`

New `src/evaluate.rs`, one dispatch arm in `cli.rs`, one `USAGE` block.
Owns its own `TOP_USAGE`/subcommand dispatcher internally (for
`evaluate compare`), following `src/benchmark.rs:37-118`'s shape without
depending on `mod benchmark`.

### B. `taguru benchmark evaluate`

A fourth subcommand inside the existing `benchmark` namespace, reusing
`docs/benchmark.html` and `benchmark.rs`'s dispatcher directly.

### C. Flat, hyphenated verb (`taguru eval-quality`, etc.)

ADR 0003 §3 E's rejected option for its own case, considered here for
completeness and rejected on the same terms.

### D. Graph-lane execution: `resolve` → `query` vs `resolve` → `recall` vs `activate`/`explore` vs none

Passage search has no graph lane (BM25 + vector only, `sources.rs:419`).
The candidates for exercising `expected_concepts`/`expected_labels`/
`expected_associations` are `recall` (paged, cue must already be an
exact stored id), `query` (exact triple, requires resolved names in all
three positions), and `activate`/`explore` (spreading-activation results
gated by `decay`/`max_depth`, which no eval field declares).

### E. Thresholds: separate file vs `eval.jsonl` header vs CLI flags

A dedicated user-authored JSON file; folding threshold keys into
`eval.jsonl`'s header; or open-ended `--min-X`/`--max-X` CLI flags.

### F. Metric-type reuse: extract to a new module vs promote `mod compare` vs duplicate

`benchmark::compare`'s `Distribution`/`Ratio`/`Count`/`MetricValue`/
`MetricDef` and their constructors either move to a new crate-level
module both `benchmark` and `evaluate` import from, or `mod compare`
itself is promoted `pub(crate)`, or `evaluate` grows its own copies.

### G. Comparison mode: subcommand vs flag vs external tool

`taguru evaluate compare BASE HEAD` as a third subcommand; a
`--compare BASE HEAD` flag on the default mode; or documenting jq-based
diffing over two `evaluation.json` files without a dedicated verb.

## 4. Decision

**A** (top-level `taguru evaluate`), **D** (`resolve` → `query`, `recall`
and `activate`/`explore` excluded), **E** (separate thresholds file),
**F** (extract to `src/measure.rs`, promoting nothing), **G**
(`taguru evaluate compare` as a subcommand).

The organizing principle: `evaluate` is a peer of `benchmark`, not a
tenant of it. Both share exactly the one artifact ADR 0003 §11 named —
`eval.jsonl` — and nothing else; `evaluate` therefore gets its own verb,
its own docs page, and its own copy of the argument-parsing boilerplate
ADR 0002 §12.1 already tracks unifying, rather than borrowing
`benchmark`'s namespace, its results-directory convention, or its
`mod extract`/subprocess dependency chain for a feature that needs none
of them.

## 5. Command surface, argument contract, and exit codes

`src/evaluate.rs`, `pub fn run(args: &[String]) -> i32`. One dispatch arm
in `cli.rs` alongside `benchmark`/`health`/`calibrate`
(`cli.rs:316-326`), one `USAGE` block. Two modes, dispatched on the first
argument the same way `benchmark::run` dispatches on subcommand names
(`benchmark.rs:102-118`), with one addition `benchmark` does not need:
the default mode has no name of its own, so the rule is **a leading
argument starting with `--` selects the default (run) mode; a leading
bare word selects a subcommand** (today, only `compare`). This is the
one hand-rolled-parser trap worth spelling out explicitly, since
`benchmark` never had to solve it (every one of its invocations names a
subcommand).

```
taguru evaluate --eval FILE --context NAME [--thresholds FILE]
                 [--url URL] [--config FILE] [--out FILE]
taguru evaluate compare BASE.json HEAD.json [--out FILE]
```

Exit codes, extending `cli.rs:6-7`'s documented contract (this ADR edits
that doc comment in the same PR as the implementation that needs it):

- `0` — run completed, no threshold file given, or every threshold
  satisfied.
- `1` — run could not complete (server unreachable past preflight,
  artifact write failure).
- `2` — usage or input error (bad flags, malformed `eval.jsonl`,
  malformed thresholds file, unknown metric name or `case_id` in a
  threshold override).
- `3` — run completed and a threshold was violated.

`3` is new; no verb in this tree returns it today. It is deliberate: a
CI job needs to tell "the corpus regressed, read the report" apart from
"the run itself failed, retry" and from "you passed a bad flag" — three
different remediations, one exit code apiece. The nearest existing
precedent, `taguru inspect` returning `1` for "corruption found," does
not generalize here because `inspect` has no separate "could not
complete" state to collide with; `evaluate` does (§9's preflight
`/health` check, matching `search.rs:217-219`'s hard-fail-on-unreachable
posture), and collapsing "gate failed" into the same code as "run
failed" would make the two indistinguishable from the exit code alone.
Without `--thresholds`, a completed run is always `0` — but `evaluate`
prints one stderr line noting the run is report-only, so a CI job that
forgot the flag does not pass silently (the same discipline ADR 0002 §5
uses for "print the target before sending anything").

## 6. The shared dataset: relocation, versioning, interpret-vs-carry-through

**Relocation.** `src/benchmark/evalset.rs` moves to `src/evalset.rs`,
`pub(crate)`. The precedent that argues *against* moving it —
`pub(crate) mod sources;` (`src/api.rs:34-36`) widened visibility in
place rather than relocating — does not transfer: `api::sources` has an
owner (the server) and `benchmark search` is a foreign consumer of it,
while after #215 lands, `evalset` has no owner inside `benchmark` at
all — both `evaluate` and `benchmark search` are peer consumers of what
ADR 0003 §11 already calls "the shared dataset." Marking items
`pub(crate)` without relocating buys nothing on its own — `identity.rs`'s
several `pub(crate)` items inside the still-private `mod identity` prove
the visibility keyword is inert until the module itself is reachable;
the relocation is the actual change, the visibility bump is what makes
it usable afterward.

**Versioning.** `taguru_eval` stays `1`. The rule: *completing a field
that was reserved-but-undefined finishes version 1; it does not revise
it.* `Option<Value>` on the three #215-only case fields and three
#215-only option fields was never a schema promise — it was what
`deny_unknown_fields` forced so a hand-written dataset's typo stays a
reported error (`evalset.rs:12-19` already says this). Because the stamp
is equality-checked and shared by two verbs, a bump would be lockstep
and would invalidate every archived v1 file that carries no #215
extension at all — an unacceptable cost for typing a field that was
declared unread from day one.

Two additive fields close the §2.1 gaps without forcing a bump (both
`#[serde(default)]`, so existing v1 files parse unchanged):

- `options.tags: Vec<String>` — replaces `options.sources`. There is no
  server-side source-id filter; `tags` maps directly onto
  `SearchPassagesRequest.tags` (`sources.rs:319-322`).
- `options.until: Option<u64>` — closes the window's right edge, mapping
  onto `SearchPassagesRequest.until`.

`options.sources` is retired, not carried through undefined — carrying
through a field with no server meaning at all would let a dataset author
believe it does something.

**Interpret-vs-carry-through: the loader needs a mode.** Once #272 gives
the three #215-only case fields (`expected_labels`, `expected_associations`,
`expected_citations`) and `options.floor` concrete types instead of
`Option<Value>`, a malformed value in one of them turns from a silent
pass-through into a hard parse error under `benchmark search` too — which
would mean `benchmark search` starts rejecting datasets over fields it
has no business validating, breaking ADR 0003 §11's *"carries through
untouched and does not interpret"* literally (a parse error is
interpretation). The loader therefore takes a mode:

```rust
pub(crate) fn load_eval_file(path: &Path, mode: Extensions) -> Result<LoadedEvalSet, String>

pub(crate) enum Extensions {
    /// #260: extension fields stay opaque `Value`s; a case carrying any
    /// of them adds one line to `warnings`, once per run, worded by the
    /// caller (never hard-coded to one verb's name).
    CarryThrough,
    /// #215: extension fields are typed and validated; a malformed one
    /// is a reported parse error like any other field. No warning is
    /// emitted — this loader call is the interpreter the extensions
    /// were declared for.
    Interpret,
}
```

This also retires `evalset.rs:212-220`'s hard-coded `"taguru benchmark
search"` string — `CarryThrough` callers supply their own verb name for
the warning text.

**Source resolution is a per-consumer concern, not the loader's.**
`benchmark search`'s `resolve_expected_source_path`
(`search.rs:911-963`) resolves `expected_sources[].source` against
`manifest.json`'s document dictionary. `evaluate` has no manifest — its
corpus was never built by `taguru benchmark extract`. `evaluate`
resolves instead against `GET /contexts/{name}/sources`
(`list_sources`, `Role::Read`), as a **preflight** step before any case
runs: an `expected_sources` entry naming a source absent from the
corpus is a reported error at startup, not a silent zero-recall case
buried in the aggregate.

## 7. Execution model: lanes, order, and what is deliberately not called

Per case, two independent lanes, fixed order, no fusion:

**Passage lane (always runs).** `POST /contexts/{context}/sources/search`
with `{query, limit, semantic_floor: options.floor, tags: options.tags,
since: options.since, until: options.until}`.

**Structural lane (runs only when the case declares
`expected_concepts`/`expected_labels`/`expected_associations`).**

1. Resolve. For each of `cues[]` (falling back to `query` when `cues` is
   empty), call `POST /contexts/{context}/resolve` (concepts) and
   `/resolve_label` (labels) with an **explicit `limit` of 5** — omitting
   it means "the ceiling itself," up to 1000 candidates
   (`resolve.rs:26-31`), which no case needs. Because tier scores are
   not comparable (`resolve.rs:36-37`), expand **only the highest tier
   present** — never mix a semantic candidate list with a lexical one.
   Record `resolved_names[]`, `resolve_tier`, and the `limit` used per
   cue in the artifact, so a structural miss is diagnosable as
   "resolution failed to find the name" versus "the graph has no such
   edge."
2. Query. For each `expected_associations[]` entry, resolve subject and
   object via `/resolve`, label via `/resolve_label`, then call
   `POST /contexts/{context}/query` with all three positions pinned.
   `query` is chosen over `recall` because `Context::recall(cue)` returns
   every edge incident on a concept and the HTTP layer pages it at
   `clamp(limit, 100, 1000)` (`recall.rs:53`) — a hub concept with more
   than 1000 incident edges can push the expected triple outside the
   page, producing a **silent false miss** that looks like a retrieval
   failure but is a paging artifact. `query` pins the exact triple,
   returns `total`, and its cost does not grow with the subject's
   degree.

**`activate`, `explore`, and `describe` are not called.** `activate`'s
and `explore`'s results depend on `decay`/`max_depth`, which no eval
field declares — there is no principled way to choose those parameters
for a case that never asked for them. `describe` is `Role::Read` and
equally tempting to reach for; it is named here explicitly so a future
reader does not assume it was overlooked rather than excluded.

**"Passage fallback" names no server behavior.** No lane fuses into or
falls back to the other — the phrase in #215's original prose describes
an aspiration, not an implemented mechanism. What `evaluate` reports
instead is a lane cross-tab (`structural_hit` / `passage_hit` / `both` /
`neither`) computed **only over cases that declare both a structural and
a source expectation** — a case with only `expected_sources` has no
structural lane to cross-tabulate and is excluded from that
denominator, not counted as a structural miss.

## 8. Expectation semantics: sources, concepts, labels, associations, citations

**Sources and concepts** use the core fields ADR 0003 §11 already fixed
(`expected_sources[]`, `expected_concepts[]`), matched against passage
hits and resolved names respectively.

**`expected_citations[]`** shape:

```json
{"source": "corpus/brewery.md", "paragraph": 3, "section": "沿革", "quote": "1897年に創業"}
```

`section` is `Option<Option<String>>` with `#[serde(default)]`: an
absent key means "don't check section," an explicit `null` means "assert
this paragraph is outside every stored section" — a real, checkable
claim, not a serde artifact, because the server's own `Citation.section`
doc comment (`sources.rs:76-82`) states the key is never omitted on the
wire. `quote` is optional free text.

Two measurements, **never merged into one score**:

- **Citation recall** — the fraction of a case's `expected_citations`
  whose `(source, paragraph)` appears among that case's served results:
  passage hits up to `limit`, plus `AttributionOut` locators from the
  structural lane when it ran. This measures whether retrieval *surfaced*
  the right evidence.
- **Locator validity** — for each expected citation, one
  `POST /contexts/{context}/citations {source, paragraph}` call,
  checking (i) it resolves (`no_source`/`no_paragraph` never fires),
  (ii) `section` matches when the key was present (including matching
  explicit `null`), (iii) `quote`, when given, is a normalized substring
  of the returned `text`. This measures whether the *locator itself* is
  correct, independent of whether that case's search happened to surface
  it — it runs even on a case whose passage-lane search missed entirely,
  because a citation can be wrong in a dataset that a working search
  would never expose.

Because the citations endpoint is not batched (`sources.rs:67-73`), the
cost is Σ|expected_citations| sequential round trips inside the client's
35s budget (`remote.rs:130`); `POST /contexts/{name}/sources/lookup`
(`sources.rs:25-27`) is not a substitute — it dereferences whole
documents, not paragraphs. `Citation.text` is exactly one paragraph
(`sources.rs:84-88`, matching `PassageHit.text`'s own paragraph-only
scope), so **a `quote` spanning a paragraph boundary can never match** —
the documented workaround is splitting it into two `expected_citations`
entries. The response does not echo `paragraph` back, so requests and
responses correlate by index.

**Normalization is `taguru::context::normalize_entry`
(`src/context.rs:936`, `pub` via `lib.rs`), not
`benchmark::identity::normalize_term`.** `identity::normalize_term`'s own
doc comment (`identity.rs:13-21`) declines to fold katakana specifically
because *"a model's choice of katakana vs. hiragana is exactly the kind
of run-to-run variation [association identity matching] exists to
measure"* — a rationale from a cross-model comparison context that does
not exist in #215, where the corpus is fixed and no model is being
compared. `normalize_entry` is what the passage index itself folds terms
with (`src/registry/terms.rs:102`), so a locator check made with it
agrees with the lane that produced the hit in the first place. The same
function is used for `expected_concepts`/`expected_labels` matching
against resolved names. The choice is recorded verbatim in
`evaluation.json`'s `matching` block, following ADR 0003 §9.4's
precedent for recording identity-matching choices in an artifact header.

## 9. Artifact schemas

### 9.1 `evaluation.json`

Single file, `--out FILE`, default `evaluation.json` in the current
directory. Stamp `taguru_evaluation: 1`, **range acceptance** (ADR 0003
§10's `IMAGE_VERSION` posture — taguru writes this file and its own
`compare` mode re-reads it, so within an accepted range a later revision
may only *add* a field, never remove or repurpose one). Every field
`#[serde(default)]`.

Metric values reuse `benchmark::compare`'s `Distribution`/`Ratio`/
`Count`/`MetricValue`/`MetricDef` shapes (relocated per §10). A metric's
concrete variant is read by first loading `definitions[metric].statistic`
(`"distribution" | "ratio" | "count"`) and deserializing that metric's
value into the type its own definition names — never through
`MetricValue`'s `#[serde(untagged)]` enum directly. That enum is
`Serialize`-only in the source today and is unsound for deserialization:
`Distribution` (`compare.rs:134-141`) has only `n: u64` as a required
field, with every other field `Option<_>`, so untagged deserialization
of a `Ratio`'s `{"value":0.0968,"n":31,"numerator":3}` succeeds silently
as `Distribution{n:31, min:None, …}` rather than failing — a compare mode
built on it would read the wrong variant with no error. Reading through
`definitions` first costs no wire change, uses no untagged deserialize,
and turns a metric with no matching definitions entry into a loud
failure instead of a silent one — making `definitions` load-bearing
rather than decorative.

Per-case block carries: `case_id`, `query`, `cues` (echoed, never used to
drive retrieval directly — see §7), `limit`, structural-lane fields
(`resolved_names`, `resolve_tier`), the passage-lane `hits[]` (full,
bounded only by `limit` itself — see §11), the structural-lane matches,
citation recall and locator-validity results, and a `missed[]` list of
unmet expectations, capped and counted per §11.

### 9.2 `changes.jsonl`

`taguru evaluate compare BASE.json HEAD.json [--out FILE]`, default
`changes.jsonl` in the current directory. Header stamp
`taguru_evaluation_changes: 1` on line 1 (ADR 0003 §10's JSONL-header
placement rule). One record per **changed** case only — `kind` ∈
`improved | regressed | added | removed`; `unchanged` is counted in the
header, never emitted per case, keeping the file's size bounded by the
number of cases whose outcome actually moved. `added`/`removed` remain
unbounded in principle (a wholesale eval-set swap emits one record per
case), which is accepted rather than capped: the eval set's own size is
already the practical bound. A human-readable terminal summary
accompanies the machine-readable file. Verdict vocabulary (`improved`,
`regressed`, `winner`-adjacent words) is legitimate here, unlike inside
`benchmark`'s artifacts — this is the one place in the tree whose whole
job is to say which run was better.

Mismatched `context`, `corpus.revision`, threshold file, or stamp
version between `BASE.json` and `HEAD.json` produce a loud warning, never
a refusal — matching ADR 0002 §10's "the warning never blocks" rule.

### 9.3 The thresholds file

`--thresholds FILE`, stamp `taguru_evaluate_thresholds: 1`,
equality-checked, `#[serde(deny_unknown_fields)]` — a direct application
of ADR 0003 §10's rule that a user-authored file gets `taguru_batch`-style
equality, not `IMAGE_VERSION`-style range acceptance. Rejected outright:
folding threshold keys into `eval.jsonl`'s header, which is itself
`#[serde(deny_unknown_fields)]` (`evalset.rs:31-32`) and would require its
own bump just to add a `thresholds` key — a strictly worse cost than a
second file; and open-ended `--min-X`/`--max-X` flags, an unbounded CLI
surface that is not reviewable as a single git diff the way a checked-in
JSON file is.

Three top-level keys:

```json
{
  "taguru_evaluate_thresholds": 1,
  "aggregate": {"recall.recall_at_k": {"min": 0.8}, "citations.recall": {"min": 0.9}},
  "cases": {"default": {}, "overrides": {"known-miss-003": {"citations.recall": {"min": 0.0}}}},
  "allow_unstable_corpus": false
}
```

`deny_unknown_fields` cannot enforce "an unknown metric name in
`aggregate` is an error" — `aggregate` is a map keyed by metric name, and
serde has nothing there to reject. That check is a post-load
set-difference against the metric names `evaluation.json` actually
emits, done explicitly in code, not left to serde. The same applies
symmetrically to `cases.overrides`: an override naming a `case_id`
absent from `eval.jsonl` is a reported error, not a silent no-op.

## 10. Versioning and compatibility

| Artifact | Key | Type | Rule |
|---|---|---|---|
| `eval.jsonl` (shared, user-authored, unchanged owner) | `taguru_eval` | `u64` | equality (ADR 0003 §10/§11); stays `1` |
| `evaluation.json` | `taguru_evaluation` | `u64` | range acceptance (ADR 0003 §10, `IMAGE_VERSION` posture) |
| `changes.jsonl` | `taguru_evaluation_changes` | `u64` | header-line-1 stamp (ADR 0003 §10) |
| thresholds file (user-authored) | `taguru_evaluate_thresholds` | `u64` | equality (ADR 0003 §10) |

Every row is a direct application of ADR 0003 §10's existing split
(user-authored ⇒ equality; taguru-written-and-reread ⇒ range acceptance
with additive-only field growth) — none of it is re-derived here.

`src/benchmark/compare.rs`'s `Distribution`/`Ratio`/`Count`/`MetricValue`/
`MetricDef` and their constructors (`from_samples`, `ratio_metric`,
`def`) move to a new crate-level module, `src/measure.rs` (`src/metrics.rs`
is already the server's Prometheus module), `pub(crate)`. `Count` gains a
constructor it currently lacks — none of its fields are exported today
(`compare.rs:229-234`), the same gap that leaves `MetricValue::Count`
unreachable from outside `compare.rs` as things stand. `benchmark::compare`
and `benchmark::search` both import from `src/measure.rs` afterward,
rather than `mod compare` being promoted `pub(crate)` in place — the
distinction is the same one §6 draws for `evalset`: after this move,
`measure` is a peer module both `benchmark` and `evaluate` consume, with
no owner inside `benchmark`. This move is filed as its own issue (§13.1),
sequenced before #272, so the first `evaluate` implementation PR is not
majority unrelated diff.

## 11. Remote access, auth, failure semantics, and artifact masking

`evaluate` becomes `remote::Api`'s seventh consumer; no new HTTP client
code. Flags: `--url URL`, `--config FILE` — the idiom `benchmark search`
already uses (`search.rs:318-386`), not the positional-URL idiom
`health`/`calibrate`/`communities` use. No `--token` flag, matching ADR
0002 §7's explicit not-filed decision. Resolution order: `--config` →
`TAGURU_CONFIG` → `load_config()` → `--url` → `default_base_url()`
(`cli.rs:492-521`); `load_config` runs before `Api::new` is constructed,
since the bearer token is captured at construction
(`remote.rs:124,135`). A `GET /health` preflight hard-fails with exit `1`
on an unreachable server, matching `search.rs:217-219`'s posture (ADR
0003 §11's own degradation clause covers a missing `plan`/`lanes` field
or one case's failure, never an unreachable server). `evaluate` prints
its target and calls `warn_on_version_skew("evaluate")` even though it
is a read-only verb and ADR 0002 §5's rule technically scopes that to
mutating verbs — the departure is deliberate, for the same reason a
quality gate wants its provenance recorded: a CI run that silently ran
against a skewed server is a worse failure mode for this verb than for
`calibrate` or `communities`.

Every endpoint `evaluate` touches — `sources/search`, `resolve`,
`resolve_label`, `query`, `citations`, `contexts/{name}/sources`,
`contexts/{name}` — is `Role::Read` (§2.2), so `evaluate` must run to
completion on a read-only key. This is asserted with a test on the model
of `auth.rs:1489`'s `explain_routes_share_their_base_endpoints_role`.

`evaluate` calls `reject_userinfo` (matching `export`/`compact`, not
`benchmark search`'s current gap — §13.1 files that gap separately) and
records the resolved URL in `evaluation.json` as scheme + host + port
only, never with userinfo, and never the bearer/`Authorization` value in
any form. Server errors are recorded as `{code, message}`, `code` drawn
from the stable `ErrorCode` vocabulary, `message` truncated to a fixed
byte cap on a UTF-8 char boundary. Per-case `missed[]` is capped at 3
entries with a `missed_truncated` count of the entries *dropped* (not
the total), following the tree's existing truncation vocabulary
(`truncate_issues`, `sources.rs:22`; `text_truncated`, ADR 0003 §9.4).

**No corpus body text is written into `evaluation.json`**, applying the
tree's existing no-text-by-default posture (`HitLocator` carries no text,
`search.rs:1585`; ADR 0003 §9.4 makes `differences.jsonl`'s text opt-in)
and going one step further — `evaluate` forbids it outright rather than
making it opt-in, because this artifact is a CI gate's output that lands
in build logs and PR comments by default, unlike `differences.jsonl`,
which an operator produces deliberately as a local file. The one bounded
exception: on a `quote` mismatch, the artifact records the user's own
declared `quote` and a boolean match result, never the served paragraph
body — nothing in #215's acceptance criteria requires echoing corpus
text, and `(source, paragraph, declared_quote, matched: false)` fully
diagnoses the mismatch. The human-readable terminal summary prints the
reproduction command (`POST /contexts/{context}/citations {source,
paragraph}`) so a reader can pull the actual text themselves, at zero
cost to the artifact.

`hits[]` is **not** capped below `limit` (a smaller, earlier draft of
this decision proposed a flat 20-hit cap and is rejected here): capping
below `limit` would contradict §12's premise that the artifact's own
locators are sufficient for an external offline scorer, since a 20-hit
echo cannot support a `recall@50` case, and it would diverge from #260,
which echoes every hit up to its own validated `limit` ceiling of 1000
(`search.rs:1550`, validated at `search.rs:387-395`). The bound is
`limit` itself; a `HitLocator` is roughly 40 bytes, so even the ceiling
is cheap.

## 12. Reproducibility: corpus revision, provider identity, comparability

**Corpus revision is bracketed, not snapshotted once.** `GET
/contexts/{name}` returns `DirectoryEntry.revision`
(`registry.rs:357-374`, type `ContextRevision{graph, passages, config}`,
`registry.rs:258-287`). `evaluate` reads it once before the first case
and once after the last, recording `revision_before`, `revision_after`,
and `stable: bool` — equality across all three lanes, never ordering, per
`ContextRevision`'s own documented guarantee. This deliberately inverts
`api/communities.rs`'s own precedent (`api/communities.rs:20-25`):
communities snapshots a revision *before* analysis specifically so a
write landing mid-analysis leaves the recorded revision older than the
analyzed graph — erring toward reporting staleness, never freshness it
doesn't have, because it takes one snapshot around one atomic operation.
`evaluate` spans a window of many independent HTTP calls with no
transactional boundary, so bracketing before-and-after is the only way
to detect a write that landed *during* the run at all.

A changed revision does not abort the run — aborting a multi-minute CI
run because one unrelated document was imported mid-run is worse than
finishing and flagging it — but an unstable run **fails the gate by
default**; opting out requires `"allow_unstable_corpus": true` in the
thresholds file (§9.3), declared alongside `aggregate`/`cases` as the
schema's third top-level key. A gate that passes on a corpus that moved
under it isn't a gate. This interacts with §5's exit codes as follows:
without `--thresholds`, every run exits `0` regardless (report-only), so
"the gate fails by default on instability" only bites once a thresholds
file is actually supplied — worth stating explicitly rather than leaving
implicit.

`ContextRevision`'s own doc comment
(`registry.rs:277-282`) states the residual risk directly: *"a cache that
outlives the process must treat a server restart (and a delete-recreate
of the same name) as invalidation: values can repeat with different
content there."* Recording the server's version alone does not close
this — a restart at the same server version repeats the same counters
with different underlying content, and `/health`'s version field is not
a boot id. `evaluate` additionally records
`DirectoryEntry.usage.last_write_epoch` (`registry.rs:254-255`) as a
wall-clock witness that a delete-recreate would move, and this residual
gap — no boot-id field exists anywhere in the tree today — is stated
here rather than silently assumed closed.

Left as a stated, not resolved, trade-off: the `config` lane bumps both
on metadata edits (description, pin, floors) *and* on embedding refreshes
that publish something (`registry.rs:271-274`) — one genuinely
invalidates a retrieval comparison, the other's effect depends on what
was refreshed. Treating all three `ContextRevision` lanes identically for
the stability gate is a deliberate simplification, not an oversight; a
future ADR may split it if false-positive instability flags become a
practical problem.

**No answer-generation LLM is required — enforced, not merely claimed.**
Nothing in Rust's module system prevents a future `src/evaluate.rs` from
importing `crate::extract` or `crate::embedding`; "no seam exists" is not
by itself a guarantee, so this ADR does not claim one. What actually
enforces AC 8 is two checks, both landing with #278 (the offline fixture
suite):

1. A source-level structural test — in the spirit of
   `cli.rs:555`'s `every_documented_variable_is_a_known_key`, which
   already reads `USAGE`'s own text as a check — over `src/evaluate.rs`
   and its submodules, asserting they contain no `TAGURU_EXTRACT_`/
   `TAGURU_EMBED_` string literal and no `crate::extract`/
   `crate::embedding` import.
2. The #278 fixture run itself: no provider configured, the server's
   vector lane reports `LanePlan{ran: false, reason: "..."}` with the
   exact string `vector_off_reason` produces (`sources.rs:608-626`), and
   this string is asserted present in `evaluation.json`. This is the
   stronger check — it proves the *server*, not just `evaluate`'s own
   process, had no embedding provider configured, which is the actual
   claim AC 8 makes.

The two are also run through `scrub_taguru_env`
(`tests/common/spawn.rs:15-45`), which already strips `TAGURU_EMBED_*`
and `TAGURU_EXTRACT_*` before a hermetic subprocess spawn — #278 is
required to run its fixture server through it.

The distinction this ADR draws and records: `evaluate` itself never
calls an LLM, but the *server* it queries may embed the query text for
the vector lane — that is server-side configuration, not something
`evaluate` does, and it is recorded (not enforced against) via `GET
/contexts/{name}/embeddings` (`EmbeddingsStatus{provider_model, glosses,
passages}`, `registry.rs:1481` / `registry/embeddings.rs:28-53`) plus
`LanePlan.vector.reason`/`.floor` in the passage-lane response — this is
AC 7's *"record model identity and the effective semantic floor,"* not a
prohibition on the server having a provider at all.
`EmbeddingsStatus.provider_model` carries a model name only, never a URL
or key, so recording it verbatim in `evaluation.json` is safe.

A future answer-faithfulness adapter is explicitly, deliberately **not
filed**, and no seam is added in anticipation of it — following ADR 0003
§12.1's own *"Explicitly and deliberately not filed"* framing: the
per-case `hits[]` locators this ADR already puts in `evaluation.json`
(§11) are sufficient for an external tool to score generated-answer
faithfulness offline, so building a hook for it now would be
speculative.

## 13. Consequences

### 13.1 Follow-up issues

| Issue | Depends on | Implements |
|---|---|---|
| relocate `src/benchmark/evalset.rs` → `src/evalset.rs` and `benchmark::compare`'s metric types → `src/measure.rs`, both `pub(crate)`; add `Count`'s missing constructor | none | §6, §10 |
| fix `benchmark search`'s unmasked `--url` credential leak: call `reject_userinfo`, mask `InputsBlock.url` | none | §2.4, §11 |
| add `Deserialize` to `MatchPage`/`TieredResolution`/`Citation`; `pub(crate) mod resolve;` in `src/api.rs` | none | §2.2, §7, §8 |
| #272 — evaluate: shared loader in `Interpret` mode, typed extension fields | this ADR | §6, §8 |
| #273 — evaluate: execution harness and `evaluation.json` skeleton | #272 | §5, §7, §9.1, §11, §12 |
| #274 — evaluate: rank metrics and concept/association coverage | #273 | §7, §8, §9.1 |
| #275 — evaluate: citation recall and locator validity | #273 | §8, §9.1 |
| #276 — evaluate: configurable regression thresholds, exit 3 | #274, #275 | §5, §9.3 |
| #277 — evaluate: comparison mode | #274, #275 | §9.2 |
| #278 — evaluate: deterministic offline fixture suite | #273–#275 | §7, §12 |
| #279 — docs: `docs/evaluate.html` and boundary documentation | #276–#278 | §13.4 |

Each of #272–#279's existing issue bodies is updated (`gh issue edit`) to
reflect this ADR's decisions before work starts, per #271's own
completion condition.

### 13.2 Not filed, and why

- A shared `--config`/URL argument-parsing helper. ADR 0002 §12.1's #248
  already owns this scope verbatim; this ADR records `evaluate` as the
  fifth copy of the parsing block (§2.4) and defers the unification to
  #248 rather than building a one-off helper here that #248 would need
  to reconcile with later.
- A `--token` flag. ADR 0002 §7 already decided against one; nothing
  here reopens it.
- In-process retrieval (bypassing HTTP). ADR 0003 §11 already decided
  this for both #260 and #215 together; not re-argued here.
- An answer-generation-faithfulness adapter or any seam for one. §12.2's
  final paragraph states why.

### 13.3 Migration and API compatibility

- `src/cli.rs:6-7`'s documented exit-code contract gains a fourth line
  (`3` — threshold violation) in the same PR that implements it.
- `MatchPage`, `TieredResolution`, and `Citation` gain `Deserialize`;
  `mod resolve` in `src/api.rs` becomes `pub(crate) mod resolve;`. None
  of these are wire-format changes — existing HTTP responses are
  unaffected; only in-binary reachability changes, mirroring the exact
  precedent `pub(crate) mod sources;` already set.
- `eval.jsonl` gains `options.tags` and `options.until`, both
  `#[serde(default)]`; `options.sources` is retired as a documented key
  (unknown-field rejection under `deny_unknown_fields` means any dataset
  that already used it — none exist, since it was never implemented —
  would now be rejected; no known consumer is affected).
- No existing verb, flag, environment variable, or artifact format
  changes meaning as a result of this ADR. `taguru evaluate` is an
  entirely new top-level verb.

### 13.4 Documentation impact

- New `docs/evaluate.html`, in the existing "Reference" nav group beside
  "Extraction benchmark" (`docs/index.html`'s card list and the
  hand-copied `<nav class="side-nav">` block present in all 15 existing
  pages both need the mechanical addition). Not an anchored section on
  `docs/benchmark.html`: that page spends 808 lines establishing "no
  single score, no ranking, no verdicts" as `benchmark`'s whole thesis,
  and co-locating a verdict-producing CI gate there would recreate, on
  the one page most likely to be misread, exactly the confusion ADR 0003
  §11 exists to prevent.
- Following ADR 0003 §12.3's own rule, the page's content is written by
  whichever of #272–#279 lands the verb, not speculatively in this ADR's
  own PR — only the nav-list addition is scoped to #279 (§13.1's table).
- `CONTRIBUTING.md` needs no edit; ADR 0004 follows the same
  `adr/000N-*.md` convention it already describes.

## Appendix: requirement traceability

| #271 requirement | Section |
|---|---|
| 1. CLI shape (`taguru evaluate` vs `taguru benchmark evaluate`) | §3 A–C, §4, §5 |
| 2. `eval.jsonl` version policy under #215's extensions | §6 |
| 3. Shared loader's location | §6 |
| 4. Graph-lane evaluation path | §3 D, §7 |
| 5. `expected_citations` shape and locator-validity criteria | §8 |
| 6. Threshold configuration format | §3 E, §9.3 |
| 7. `evaluation.json` schema name and version | §3 F, §9.1, §10 |
| 8. Comparison mode input/output | §3 G, §9.2 |
| 9. Convergence on `remote::Api`; `--url`/`--config`/token resolution | §11 |
| 10. Failure-detail bounds and masking | §11 |
| 11. Corpus-revision recording | §12 |
| 12. Structural guarantee that no answer-generation LLM is required | §12 |
