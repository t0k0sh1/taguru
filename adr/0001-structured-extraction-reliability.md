# 0001. Guaranteed structured extraction and truncation strategy

- **Status**: Proposed
- **Date**: 2026-07-23
- **Issue**: #188
- **Related**: #178, #179, #180, #181, #182, #185, #187
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How Taguru's three extraction producers — the Rust CLI `taguru extract`
(src/extract.rs), the Python `langchain-taguru` SDK, and the TypeScript
`langchain-taguru` SDK — make structured extraction *reliably complete*,
without "retry until the model happens to emit valid JSON" as the primary
correctness mechanism. Out of scope: the benchmark CLI's shape (#189), remote
CLI access (#190), retrieval-path behavior, model recommendations, and the
concrete flag/event/exception names of the follow-up implementations.

## 2. Context

### 2.1 Motivating failure

A real local-paper ingestion run with `taguru-extract-12b` produced three
source failures after both LangChain attempts ended in malformed JSON
(`Unterminated string starting at: …`). Some Ollama calls reached
approximately the run's configured 2048 output-token ceiling, but the
artifacts of the time retained no per-attempt finish/usage metadata, so
truncation could not be distinguished from model syntax error. The
instrumentation gaps are since closed — #177/#183 added per-attempt
diagnostics and #178/#184 added finish-reason capture in all three
producers — and this ADR decides the strategy those instruments were built
to inform.

### 2.2 Guarantee taxonomy

The word "reliable" hides five different guarantees. This document keeps them
separate, and §5 states which mechanism buys which:

| # | Property |
|---|---|
| G1 | Syntactically valid JSON |
| G2 | Schema-valid extraction fields (the canonical `ModelOutput` shape) |
| G3 | Completion before an output/time bound |
| G4 | Semantically valid facts and attribution |
| G5 | Complete source import with no partial writes |

### 2.3 Current producer reality (audited 2026-07-23)

- **The Rust CLI is weaker than "Option A" as commonly understood.** Its
  request body is exactly `{model, temperature: 0, messages}`
  (src/extract.rs:946-951): no `response_format`, no `tools`, and **no
  output-token parameter of any kind** — the "2048 ceiling" in §2.1 lived in
  the serving side's configuration, not in this repo. The canonical JSON
  Schema exported by #185 (`model_output_json_schema()`,
  src/extract.rs:1456) is `pub(crate)` and `#[allow(dead_code)]`: tests read
  it; the wire never sees it.
- **The SDKs' `structured_output` opt-in is tool-calling-gated.** Both call
  `with_structured_output(MODEL_OUTPUT_JSON_SCHEMA, include_raw=True)` and
  refuse at construction when the model lacks `bind_tools`
  (sdk/python-langchain/src/taguru_langchain/ingest.py:225-229,
  sdk/typescript-langchain/src/ingest.ts:187-200). Neither pins `method=`,
  so whether a provider resolves that to schema-constrained decoding or to a
  forced tool call is LangChain's per-provider choice, not Taguru's.
- **The two producer families do not even speak the same wire to the same
  backend.** Rust speaks OpenAI-compatible `/chat/completions` only (finish
  field `choices[0].finish_reason`, src/extract.rs:974-976); the SDKs, via
  `ChatOllama`, speak Ollama's native `/api/chat` (finish field
  `done_reason` — the very reason the SDK metadata reader checks
  `done_reason` → `finish_reason` → `stop_reason` in that order). Capability
  rows in §6 are therefore keyed by *(backend, wire)*, not backend.
- **Parse strictness is asymmetric.** Rust deserializes leniently per
  field/per item (`lenient_vec`/`lenient_string`/`lenient_f64`/`lenient_u32`,
  src/extract.rs:1301-1385): a wrong-typed scalar costs the field, a
  malformed array item costs the item, never the chunk. Python (pydantic)
  and TypeScript (`coerceOutput`) reject the *whole answer* over one bad
  nested field, consuming a corrective attempt Rust would not have spent.
  §8 rules on this; results/lenient_vs_strict_delta.csv quantifies it.
- **All three producers share a deeper integrity gap: merge-level silent
  drop.** `merge()` (src/extract.rs:1559,
  sdk/python-langchain/src/taguru_langchain/_extract.py:454,
  sdk/typescript-langchain/src/extract.ts:597) drops business-rule-invalid
  items — bad weight, empty name, dangling alias — counts them in a
  `dropped` tally, and still reports the extraction as successful. That is
  precisely the "subset with deleted malformed facts called complete" this
  issue's acceptance criteria forbid. #180/#181 already plan the fix for
  Python/TypeScript; no issue covered Rust until this ADR's follow-up.
- **Diagnostics are Python-only.** The #177/#183 event stream (10 event
  types; `ProviderMetadata{finish_reason, input/output/total tokens}`)
  exists only in the Python SDK. TypeScript has no events module; the Rust
  CLI has no per-attempt output at all beyond one stderr line per failed
  document.
- **What #178 already shipped everywhere** (and this ADR builds on, not
  replaces): bounded attempts (`max_attempts` 1..=10, default 2), the
  rebuild-not-accumulate corrective turn, `corrective_context_bytes`
  replay bounding, `fact_budget`, and the truncation-aware "answer
  SHORTER" corrective driven by `finish_reason`/`done_reason` `"length"`
  or Anthropic's `"max_tokens"`.

### 2.4 Evidence

All measured claims below come from the checked-in harness under
[`adr/0001/`](0001/) — 242 trials over 4 mechanisms × 2 wires × 6 fixture
documents × output ceilings {512, 2048, unbounded} against a local Ollama
(primary model `taguru-extract-12b`, the §2.1 model itself; spot-check
`qwen2.5:7b`), replaying the production corrective loop with the production
prompts imported from the Python SDK. Aggregates live in
[`adr/0001/results/`](0001/results/); §6 marks every non-measured cell as
documentation-sourced.

One production-path confirmation ties the harness to the shipped binary:
an unmodified `taguru extract` run against the same backend and model
(`TAGURU_EXTRACT_URL=…/v1/chat/completions`, 30 s timeout) failed the
corpus's *two-sentence* document — `after 4 attempts: chat request failed:
timeout: global` — exactly the harness's 0%-on-/v1 finding (§6), on the
real code path.

## 3. Options considered

Each option: what it is → evidence → what it guarantees → verdict.
(Verdicts are summarized here; §4 composes them into the decision.)

### A. Status-quo JSON mode plus corrective retry

Prompt-engineered JSON (with or without a provider "JSON mode"), free-text
parse, bounded corrective retries. This is what all three producers do
today; the Rust CLI does not even send JSON mode.

- Evidence: results/failure_rates.csv (mechanisms A0/A1),
  results/lossless_correction.csv.
- Measured (primary model, native wire, core tier): bare prompting (A0)
  completed 66.7% of trials — 87.5% at the 2048 budget (the residual
  failures are 180 s timeouts on the fact-dense documents), 50% unbounded
  (rambling generation blows the wall clock), 0% at 512 (all
  `length`-terminated, and A0 alone could not salvage even a shorter
  answer). JSON mode (A1) lifted that to 91.7% and eliminated every
  timeout at adequate budgets. The corrective retry recovered 7.1-8.3% of
  native trials — useful, and nowhere near a correctness mechanism: on the
  broken wire (§6's thinking quirk) it recovered 0% of anything. On the
  compliant secondary model even A0 scored 100% — mechanism differences
  only appear when the model or wire misbehaves, which is exactly when a
  guarantee is wanted.
- Guarantees: none of G1-G3 — G1/G2 are probabilistic, G3 unbounded below a
  provider default. The corrective loop converts some failures into
  successes at the price of extra calls.
- Verdict: **Compatibility fallback**, never the preferred path. A1 (JSON
  mode) is a worthwhile intermediate rung where json_schema is
  unavailable; A0 remains only as the last rung and as today's unchanged
  default until a capability probe verifies something better.

### B. JSON-Schema-constrained decoding

Pass the canonical `ModelOutput` schema (#185) to the provider:
`format: <schema>` on Ollama native, `response_format: {type: "json_schema"}`
on OpenAI-compatible wires, `with_structured_output` in LangChain.

- Evidence: results/failure_rates.csv (mechanism B),
  results/capability_matrix.csv, results/golden_recall.csv,
  results/item_counts_and_completion.csv, results/latency_tokens_percentiles.csv.
- Measured: on the capable wire, 91.7% core completion (100% at 2048 and
  unbounded), identical to A1/C — and at no measured cost: latency p50/p90,
  merged-association counts, and golden recall are indistinguishable from
  A1/C on the same cells (long_dense: 21 associations, triple recall 0.18,
  pair 0.33 for both A1 and B). The schema held on both Ollama wires in the
  canary, including suppressing the thinking model's thought-leak that
  broke JSON mode on /v1. What B does *not* survive: truncation — at the
  512 budget, 50% of trials still `length`-terminated (constrained decoding
  cut mid-structure is invalid JSON like anything else), and on the
  /v1-with-thinking pathology B stalled at the timeout cap like every
  other mechanism. Both confirm the caveat below.
- Guarantees: G1 and G2 *within a completed response*, by construction of
  constrained decoding — but **not G3**, and a response cut off at the
  output cap is not schema-complete: truncation converts a guaranteed-valid
  answer into a syntactically invalid one. B without an output-budget
  policy is a false guarantee.
- Verdict: **Preferred mechanism** wherever the capability probe verifies
  it — always layered on E's budget policy, never alone.

### C. Tool/function calling

One `ModelOutput` tool; the model "calls" it with the extraction as typed
arguments.

- Evidence: results/capability_matrix.csv, results/failure_rates.csv
  (mechanism C).
- Measured: on the capable wire, exactly B's numbers (91.7% core, 100% at
  adequate budgets, marginally more associations on the densest document —
  31 vs 26). But the canary shows tool-call *presence* is unforceable where
  it matters: on Ollama's OpenAI-compatible wire, `tool_choice` was ignored
  and the forced-call probe came back with no tool call at all (the
  thinking model burned the budget instead). The 24 v1 C cells were
  capability-gated off rather than burned — recorded as
  `skipped_unsupported`, which is itself the finding.
- Guarantees: argument-shape conformance where the provider enforces it;
  tool-call *presence* is not enforceable on every wire (Ollama's
  OpenAI-compatible surface ignores `tool_choice`), and arguments are still
  subject to the output cap (G3 unresolved).
- Verdict: **Not a primary mechanism** — no measured advantage over B and a
  real portability gap. It stays reachable through LangChain's per-provider
  `with_structured_output` resolution in the SDKs (some providers implement
  structured output *as* tool calling); Taguru does not build its own
  tool-calling path.

### D. Caller-controlled extraction units

Bound each request by a deterministic input unit (e.g. paragraph range) so
no single response needs to grow without bound; merge units into the one
source-level batch only after every unit succeeds.

- Evidence: results/failure_rates.csv and results/golden_recall.csv, D rows
  (per-paragraph units of long_dense vs. the whole document).
- Measured: 15/15 per-paragraph units completed at the 2048 budget, and
  extracted *more* than the same document presented whole: golden-weighted
  pair recall 0.58 per-unit vs 0.30 whole-document (same mechanism, wire,
  budget). Bounded input bounds output — the timeout/truncation exposure
  that killed the *densest* whole document (§3.A) cannot arise when no
  unit is dense — and smaller units lose fewer facts to the model's own
  summarization. The cost is call count (5 calls vs 1 here) and cross-unit
  spelling divergence, absorbed partly by document-level `merge()` and the
  vocabulary discipline.
- Guarantees: G3 becomes *achievable* (output size is bounded by input
  size); costs more calls and risks cross-unit inconsistency (alias/spelling
  divergence between units) — the existing whole-document `merge()` and the
  document-level vocabulary discipline absorb part of that, and #179's
  checkpoint keys already anticipate "chunking parameters" as an input.
- Verdict: **Adopted as the escalation ladder's split rung** (§7): not the
  default request shape, but the deterministic next step when budget
  escalation is exhausted — and a caller-selectable unit policy for
  documents known to be fact-dense.

### E. Output-budget policy

Stop treating the output cap as ambient serving configuration: set it
explicitly per request, detect cap terminations from provider metadata, and
make the next action deterministic — escalate the budget or split the input
(D), never re-ask under the very limit just proven too small.

- Evidence: results/truncation_crosstab.csv (ceiling sweep 512/2048/∞),
  E rows of results/failure_rates.csv (escalate-on-`length` prototype).
- Measured: truncation is a pure budget property, not a mechanism property
  — at 512 every mechanism `length`-terminated on the dense documents
  (100%), at 2048+ none did. The escalate-on-`length` prototype recovered
  3/3 trials (attempt 1: 512 → `length`; attempt 2: unbounded → valid).
  One instructive defect in the prototype: it escalated the budget while
  still sending #178's "answer SHORTER" corrective, and the model obliged —
  9-11 associations against the 26 a direct adequate-budget run produces.
  Escalation must therefore *regenerate under a neutral ask* (exactly
  #180/#181's "regenerate truncated output from the original source"), and
  the SHORTER corrective is reserved for the no-budget-control fallback.
  The same completeness cost shows up in the core tier: 512-budget trials
  "salvaged" by the SHORTER corrective produced 8 associations where an
  adequate-budget run of the same 52-golden-fact document produces 26 —
  valid, and quietly one-third as complete.
- Guarantees: G3 by termination (a finite escalation ladder ending in
  split-or-fail); turns the §2.1 ambiguity ("truncation or syntax error?")
  into a classified, logged state.
- Verdict: **Adopted, mandatory, mechanism-independent** — the budget is
  set explicitly per request, `length` termination is classified from
  metadata, and the next action is the §7 ladder.

### F. Passage-first completion with graph-extraction status

Make the original passage searchable even when graph extraction fails;
record graph extraction as complete/failed/pending per source and retry it
independently.

- Evidence: none gathered — this is an architectural contract question, not
  a measurable mechanism property.
- Analysis: it inverts the batch contract that #187 just hardened. A Taguru
  batch is *one source's complete truth*, imported atomically; import
  refuses alias-conflicting batches before any mutation lands. A
  passage-now-graph-later source is a partially-true source the retrieval
  side cannot distinguish from a fully-ingested one without a new
  server-side degraded-status surface, new protocol wording, and a second
  writer path — all to mitigate a failure mode that D+E are designed to
  close upstream. The availability concern it targets (long documents,
  interrupted runs) is #179's checkpoint/resume territory, producer-side,
  without weakening the import contract.
- Verdict: **Rejected** for the default contract. If checkpoint/resume plus
  the D+E ladder still leave a demonstrated availability gap, a future ADR
  may revisit it with a server-side status design; nothing in this decision
  precludes that.

## 4. Decision

**A layered strategy: capability-probed schema-constrained decoding (B)
over an explicit output-budget policy (E), with prompted JSON plus
corrective retry (A) as the compatibility fallback and unit splitting (D)
as the escalation rung. C and F are not built.**

1. **Mechanism selection is per (backend, wire), verified, never assumed.**
   At construction/startup each producer resolves a rung on the ladder:
   `json_schema` constrained decoding (B) → JSON mode (A1) → bare prompted
   JSON (A0). Resolution comes from the §6 matrix plus a capability check
   against the live backend, because the same backend answers differently
   per wire (the measured Ollama /v1-vs-native split) and a provider that
   *accepts* a parameter has not necessarily *honored* it. A response
   produced under B still runs the full parse/merge validation — the wire
   narrows shapes; validation stays the authority — and a B-path response
   that fails validation is counted as provider non-conformance and takes
   the corrective path.
2. **The output budget is an explicit, first-class request parameter**
   (today it is not even sent), with §7's deterministic reaction to
   `length`/`max_tokens`: escalate once under a *neutral* regeneration ask,
   then split the unit (D), then fail the source. The #178 "answer SHORTER"
   corrective remains only where the producer has no budget control.
   Rationale: the evidence shows truncation is a budget property that no
   mechanism survives (§3.B), and that the SHORTER salvage quietly costs
   two-thirds of the extraction (§3.E) — reliability bought by shrinking
   the answer is a completeness failure under §8's integrity ruling.
3. **The corrective-retry loop stays, demoted to what it measurably is** —
   a recovery path (7-8% of trials on the capable wire, 0% on the broken
   one), improved by #180/#181's path-addressed issues, bounded by the
   #178 controls. It is no longer the mechanism reliability rests on.
4. **What remains probabilistic is named, not papered over**: extraction
   *quality* (G4). The measured golden recall stayed low (≤0.33 pair
   recall whole-document) even in cells with 100% structural validity —
   the model translated labels against the prompt's explicit instruction
   in every mechanism. Structure is what this ADR can guarantee;
   discipline is model quality (docs/extract.html's existing guidance),
   measurable by #189, and no mechanism choice substitutes for it.

Per producer: the Rust CLI implements the ladder for its OpenAI-compatible
wire (schema on the wire at last — #185's export stops being dead code) and
gains the budget control it entirely lacks. The SDKs keep
`structured_output` (LangChain's `with_structured_output`) as their B rung
and add the same budget/escalation policy around it. MCP changes nothing
mechanism-wise (§11).

## 5. Guaranteed vs. probabilistic properties

"Guaranteed" below always means *within the stated scope* — nothing makes an
LLM's answer unconditionally valid; the design makes every departure
detected, classified, and either corrected or terminal.

| Property | Under B (primary) | Under A (fallback) | Enforcement point |
|---|---|---|---|
| G1 syntactic JSON | Guaranteed per *completed* response (constrained decoding); truncated responses excluded by definition | Probabilistic; corrective retry recovers a measured 7-8% | wire (B) + parse layer (always) |
| G2 canonical schema shape | Guaranteed per completed response | Probabilistic | wire (B) + schema/parse revalidation (always) |
| Business-rule validity (finite non-zero weight, name caps, alias resolution) | Probabilistic — deliberately not in the wire schema | same | merge validation + §8 corrective turn |
| Completeness (no silent item loss) | Guaranteed as *accounting*: every lost item is detected and either corrected or fails the source | same | §8 lenient-parse-strict-accounting, all paths |
| G3 termination before a bound | Guaranteed termination via §7's finite ladder (escalate → split → fail); *completion* within the first budget is probabilistic but classified | same | E budget policy, mechanism-independent |
| G4 semantic/factual quality, attribution discipline | **Probabilistic, always** — measured: label-language violations at 100% structural validity | same | model choice (docs/extract.html), measured by #189 |
| G5 atomic source import | Guaranteed | Guaranteed | server (#187) + producer batch self-check + §9 |

## 6. Provider capability and fallback matrix

Rows are *(backend, wire)* pairs — §2.3 explains why. **Provenance**:
`measured` = adr/0001 harness against Ollama 0.32.0 on 2026-07-23 (see
results/capability_matrix.csv and results/canary.json); `documented` =
vendor/upstream documentation as of this ADR's date, to be re-verified by
the implementing PR against the versions it ships with.

| Backend, wire | JSON mode | json_schema constrained | Tools | Finish-reason spelling | Usage metadata | Output-cap param | Provenance / quirks |
|---|---|---|---|---|---|---|---|
| Ollama, native `/api/chat` (SDK producers via ChatOllama) | yes (`format:"json"`) | yes (`format:<schema>`) — held in every measured trial | yes (no forced `tool_choice` equivalent) | `done_reason`: `stop`/`length` | `prompt_eval_count`/`eval_count` | `options.num_predict` (`-1` = unbounded) | measured. `think:false` works only for models with the `thinking` capability (send conditionally); the Modelfile can bake `temperature`/`num_ctx` that requests must override |
| Ollama, OpenAI-compat `/v1/chat/completions` (Rust CLI's wire) | accepted, but a thinking model leaks its thought as the JSON answer | accepted and enforced (canary) — but see quirk | accepted; **`tool_choice` ignored** (forced-call probe returned none) | `finish_reason`: `stop`/`length`/`tool_calls` | `usage.prompt_tokens`/`completion_tokens` | `max_tokens` | measured. **This wire has no way to disable thinking** (`think`/`num_ctx`/`keep_alive` do not exist in the OpenAI protocol): the thinking-capable primary model completed **0% of trials on this wire across every mechanism** (timeouts and thought-burn), while the non-thinking secondary scored 100% on the same cells. Wire capability decides before mechanism choice does |
| Rust `taguru extract` today (any OpenAI-compat backend) | not sent | not sent | not sent | `choices[0].finish_reason` | not read | **not sent** | measured via source read (src/extract.rs:946-951) — the gap the follow-ups close |
| OpenAI API | `response_format: json_object` | `response_format: json_schema` (strict mode requires every property listed in `required` — the canonical schema's optional `weight`/`paragraph` need `strict:false` or a strict-mode variant) | yes (incl. strict) | `finish_reason`: `stop`/`length`/`tool_calls`/`content_filter` | `usage.*` | `max_completion_tokens` (`max_tokens` legacy) | documented |
| Anthropic Messages API | no | structured outputs in beta as of early 2026; the stable path is tool-use | yes | `stop_reason`: `end_turn`/`max_tokens`/`tool_use`/`refusal` | `usage.input_tokens`/`output_tokens` | `max_tokens` (required) | documented. `max_tokens` being mandatory makes an explicit output-budget policy unavoidable on this backend |
| llama.cpp server (`/v1`) | yes | yes (grammar-backed) | template-dependent | `finish_reason`: `stop`/`length` | yes | `max_tokens` / `n_predict` | documented |
| vLLM (`/v1`) | yes | yes (guided decoding) | model/template-dependent | `stop`/`length` | yes | `max_tokens` | documented |
| LM Studio (`/v1`) | yes | yes | yes | `stop`/`length` | yes | `max_tokens` | documented |
| Bedrock via LiteLLM-style proxy (docs/bedrock.html pattern) | proxy-mapped | provider-dependent (Anthropic-on-Bedrock → tool-use path) | yes | normalized to OpenAI spellings (`length`) by the proxy | mapped | `max_tokens` | documented |
| LangChain `with_structured_output` (meta-row: what the SDKs delegate to) | n/a | provider integration picks the method when `method=` is unpinned — the SDKs deliberately do not pin it (ingest.py:226) | n/a | surfaced via `response_metadata`, spelling preserved per provider | `usage_metadata` | provider kwarg | documented — inspect the installed `langchain-*` version at implementation time |

**Fallback ladder** (per backend, evaluated at construction/startup, not
per-chunk):

1. `json_schema` constrained decoding (B) — when the matrix row plus a live
   capability check verify it (a backend may *accept* the parameter without
   honoring it; the harness's canary — constrained probe vs. unconstrained
   baseline on a prose-inviting prompt — is the model for that check).
2. JSON mode / `json_object` (A1) — when only unconstrained JSON forcing
   exists.
3. Bare prompted JSON (A0) — today's behavior, the unconditional floor and
   the unchanged default until a probe verifies a higher rung.

Orthogonally, on every rung: the E budget policy (§7), the corrective
retry (#178/#180/#181), and full parse/merge validation. The SDKs'
`structured_output=True` *is* rung 1-or-tool-calling as resolved by
LangChain's provider integration; the Rust CLI implements the ladder
directly on its wire (follow-up). A thinking-capable model on a wire that
cannot disable thinking (measured: Ollama /v1) is a capability failure of
the *wire*, not the mechanism — surface it at startup, do not burn
per-chunk attempts discovering it.

## 7. Detection and escalation state machine

Every attempt terminates in exactly one of these states, classified from
provider metadata *before* any parse-level interpretation — output-limit
termination always has a deterministic next action distinct from
malformed-`stop` correction:

| State | Signal | Next action |
|---|---|---|
| `STOP_VALID` | finish `stop` (or tool_calls), answer parses and validates | Success — proceed to merge/import. |
| `STOP_MALFORMED` | finish `stop`, parse or validation failure | Corrective turn (bounded by `max_attempts`, rebuild-not-accumulate, replay bounded by `corrective_context_bytes`) — #178's loop, with #180/#181's path-addressed issues once they land. Under a schema-constrained primary this state signals provider non-conformance; it takes the same corrective path and is counted in diagnostics as non-conformance. |
| `LENGTH_LIMITED` | finish `length` / `max_tokens`, regardless of parse outcome | **Never re-ask under the same limit.** Deterministic ladder: (1) if the request's budget is below the backend ceiling, escalate the budget once; (2) otherwise split the extraction unit (D) and extract per-unit; (3) if already at minimum unit, fail the source. A truncated answer is never salvaged as a prefix (it is regenerated, not repaired), and the "answer SHORTER" prompt of #178 remains only as the no-control-available fallback for backends where the producer cannot set a budget. |
| `EMPTY` | HTTP 200, empty content | Distinguished diagnostic (thinking-budget burn, docs/extract.html) — corrective retry once, then fail with the named diagnosis, never the generic parse error. |
| `REFUSAL` | finish `content_filter` / `refusal` | **New, currently unhandled in all three producers** (it falls into the generic corrective path today, which cannot fix a policy refusal). Ruling: terminal for the source, no corrective turn, distinct failure class in diagnostics. Assigned to the follow-up implementations. |
| `TIMEOUT` / `TRANSPORT` | HTTP/transport layer | Existing transport retry policy (4 attempts, jittered backoff, `Retry-After` honored) — unchanged by this ADR. |

## 8. Correction taxonomy

Three buckets, and a ruling that unifies the producers:

1. **Lossless-automatic (no model call, always applied):** fence stripping,
   widest-braces slicing, whitespace/BOM trimming — today's set — plus the
   unambiguous syntax normalizations #180/#181 add (e.g. a trailing comma
   whose removal is unambiguous). These repair *transport syntax*, never
   content.
2. **Corrective-turn (bounded model calls):** everything that changes
   content goes back to the model with path-addressed issues
   (`associations[1].weight: expected finite non-zero number, got string`),
   the #180/#181 contract: return the complete object, preserve every item,
   correct rather than delete, add nothing, JSON only. Revalidate the whole
   corrected answer; if still invalid, fail the source and import nothing.
3. **Never-silent-drop (the integrity ruling):** the default mode never
   imports a subset of a source's knowledge-bearing items while reporting
   success.

**Parse-strictness ruling (question 6).** The producers unify on **lenient
parse with strict accounting**:

- *Parse leniently everywhere* (Rust's current semantics become the
  three-producer semantics): one wrong-typed field must not reject fifty
  good associations, and must not spend a corrective attempt re-generating
  what was already 98% usable. results/lenient_vs_strict_delta.csv
  measures the two philosophies on identical answers — and, honestly
  reported: the delta was **zero** across all ~200 parsed answers of this
  run, because the extraction-tuned primary model emits type-clean JSON
  and its failures are truncation-shaped (syntax), which strictness does
  not affect. The unification is justified by architecture, not by this
  measurement: the asymmetry is real in code (§2.3), it bites exactly on
  the weaker models the SDKs meet in the wild, and carrying two
  philosophies means the same model answer can succeed on one producer
  and burn a retry on another — the opposite of §11's parity contract.
- *Account strictly everywhere*: what lenient parsing nulled and what
  `merge()` would drop is no longer a silent tally — it becomes the
  path-addressed issue list of bucket 2. Items lost to leniency trigger the
  corrective turn; only a fully-valid (or corrected-to-valid) extraction
  imports. This replaces the SDKs' reject-whole-answer *and* all three
  producers' merge-level silent drop with one shared discipline.
- An explicit lossy/partial opt-in (per #180/#181's integrity policy) may
  keep today's drop-and-proceed behavior available; it is never the
  default, and its outcome is always reported as partial.

## 9. Source-completion semantics

- **What "complete" means (question 7):** one source's batch — every
  association, alias, question, and the passage the run asked for —
  validated end-to-end, imported atomically by the server, with #187's
  guarantee that a refused batch mutates nothing. A source either imports
  whole or does not import; `extract` already re-parses every emitted batch
  with the import parser before writing it (src/extract.rs:652-657), and
  that self-check stays.
- **Passages before graph (question 8):** No — Option F is rejected (§3.F).
  Under the default contract a source becomes visible only when its
  complete batch lands. Long-document availability is #179's
  checkpoint/resume concern, producer-side.
- **Failed replacement:** a failed re-ingest of an existing source leaves
  the previous import untouched (server-side atomicity, #187) — the
  producers' obligation is only to never emit a partial batch as if
  complete, which §8 enforces.

## 10. Diagnostics retention (question 9)

- **Vocabulary:** Python's `ProviderMetadata` (finish_reason,
  input/output/total tokens) plus per-attempt latency, the §7 state, and
  the parse/validation issue list becomes the shared cross-producer
  diagnostic shape. The harness's per-attempt record
  (adr/0001/harness.py) is a working prototype of it.
- **Where:** Python — the existing #177 event stream. TypeScript — the
  ported event stream (follow-up). Rust CLI — an opt-in `--diagnostics-out`
  JSONL sidecar (follow-up); the manifest stays a skip-index, not a log.
- **Bounded and safe:** metadata-only by default. Raw model text is opt-in
  and byte-capped (the `corrective_context_bytes` precedent), because raw
  responses can embed the source document's own content; whatever cap
  applies is applied at capture time, not display time. No
  chain-of-thought is ever retained — only provider-surfaced metadata and
  validation results.

## 11. Producer-consistency contract (question 10)

"Parity" is itemized; each item names its enforcement mechanism:

| Item | Status | Mechanism |
|---|---|---|
| Prompt text, PROMPT_VERSION, corrective texts | in parity | hand-mirrored, parity-matched unit tests (the #178/#184 discipline) — treat the three as one artifact |
| Canonical schema | in parity | three hand-mirrored copies + shared fixtures tests/fixtures/model_output/, cross-checked by the harness's startup self-test |
| Finish-reason normalization | in parity | `"length"`/`"max_tokens"` in lockstep (PR #197's three-language rule) |
| Request-shape (mechanism per §4) | **new obligation** | each follow-up implements the same §6 ladder for its wire; parity-matched tests assert the emitted request shape |
| Parse strictness + issue accounting | **changes** (§8) | Rust semantics + path-addressed accounting in all three; shared accepted/repaired/rejected fixtures (#180/#181's fixture plan) |
| Per-attempt diagnostics | **Python-only today** | TS port (follow-up); Rust sidecar (follow-up); one shared field vocabulary (§10) |
| MCP | different duty | MCP does not run the model; its obligation is the structured validation-error shape and tool-description wording — #182 owns that surface entirely |

`sdk/spec/surface.yaml`'s mechanical gate does not cover `TaguruIngester`'s
options today; parity there rests on the hand-matched tests above. That has
held through #178/#184/#197 and stays the mechanism of record; extending
surface.yaml to ingester options is optional hardening, not required by
this ADR.

## 12. Consequences

### 12.1 Follow-up issues

New issues created from this ADR:

| Issue | Title | Producer | Implements |
|---|---|---|---|
| #198 | extract: wire structured output onto the request and add an output-budget control with deterministic length escalation | Rust CLI | §4.1 ladder + §4.2/§7 budget policy on the OpenAI-compatible wire |
| #199 | extract: replace merge-level silent item drop with path-addressed corrective retry | Rust CLI | §8 (the Rust twin of #180/#181) |
| #200 | extract: emit bounded per-attempt diagnostics | Rust CLI | §10 (`--diagnostics-out` sidecar) |
| #201 | typescript-langchain: per-attempt ingestion progress and diagnostics events | TypeScript SDK | §10/§11 (the TS twin of #177, kept separate from #181 as Python kept #177 separate from #180) |

Amendments to existing open issues (comments, no scope rewrites):

| Issue | Amendment |
|---|---|
| #180 / #181 | Re-positioned as the corrective layer beneath §4's mechanism; parse unifies per §8 (lenient parse, strict accounting); fixtures shared with #199 |
| #182 | Reuses §7's terminal-state vocabulary and §8's path-addressed issue shape verbatim; owns the whole MCP surface |
| #179 | Coordination: §7's split rung changes unit boundaries mid-run — checkpoint keys must capture per-unit splits (e.g. unit content hash) |

Dependency order: none of the four has a hard prerequisite. #198 unlocks
the most value; #199 pairs with #180/#181 (shared fixtures); #200/#201
are independent observability work; #182 consumes the shared vocabulary
once any of them lands.

### 12.2 Migration and API compatibility

- Every new control ships **default-off / behavior-preserving**: explicit
  output budgets, schema-on-the-wire, diagnostics sinks, and events are
  opt-in or default to today's values (the #178 discipline).
- The one deliberate behavior change is §8's integrity ruling: items that
  were silently dropped become corrective-turn subjects and can fail a
  source that previously "succeeded" partially. This ships inside
  #180/#181 and the Rust twin, each with its own explicit lossy opt-out,
  and is called out in their changelogs as a behavior change.
- `PROMPT_VERSION` stays 2: schema-on-the-wire and budget parameters are
  transport-level and do not alter the prompt text. Any follow-up that
  edits prompt wording bumps it in all three producers at once.
- The SDKs' `structured_output` flag keeps its meaning (LangChain-delegated
  structured output); the §4 mechanism selection refines what it does per
  provider rather than renaming it.

### 12.3 Documentation impact

- docs/extract.html's trust section overstated reality ("the output is
  schema-constrained"); corrected in this ADR's PR to "validated against
  the extraction contract before anything is written."
- docs/extract.html gains the output-budget/structured-output flags when
  the Rust follow-up lands (that PR's duty, like every feature PR here).
- src/llm-protocol.md and MCP tool descriptions: no changes beyond #182 —
  that issue owns the whole MCP-facing correction contract; this ADR only
  supplies its vocabulary (§7/§8).

## 13. Experiment appendix

[`adr/0001/`](0001/) is this decision's reproducible evidence trail: the
fixture corpus (six issue-mandated categories, with golden ground truth
derived from the 青嶺酒造 corpus of examples/paragraph_corpus and
tests/qa_recall.rs), the harness (`harness.py`, importing the Python SDK's
`_extract.py` so prompts/parsing cannot drift from production), the rollup
(`rollup.py`), and the aggregated results (`results/*.csv`,
`results/canary.json`, the journaled `results/attempts.jsonl`). It is
distinct from tests/fixtures/model_output/ (permanent, CI-checked parser
fixtures owned by #178/#185): adr/0001 is decision evidence, re-runnable
with `uv run adr/0001/harness.py`, not a CI gate. #189's benchmark tooling
may treat it as a seed/prototype; nothing here constrains #189's design.

## Appendix: requirement traceability

| Issue #188 requirement | Section |
|---|---|
| Q1 preferred mechanism per provider capability | §4, §6 |
| Q2 compatibility fallback | §4, §6 |
| Q3 guarantees per path | §5 |
| Q4 output limits selected/detected/escalated | §7 |
| Q5 split vs. retry | §7 (LENGTH ladder), §3.D |
| Q6 lossless corrections / never silently dropped | §8 |
| Q7 successful source completion | §9 |
| Q8 passages before graph | §3.F, §9 |
| Q9 diagnostics retention | §10 |
| Q10 four-surface consistency | §11 |
| AC: evidence from Ollama + OpenAI-compatible path | §2.4, §6 (measured rows) |
| AC: guaranteed vs. probabilistic separated | §5 |
| AC: capability/fallback matrix | §6 |
| AC: no valid-prefix / deleted-subset completion | §7 (LENGTH), §8, §9 |
| AC: deterministic length action distinct from malformed-stop | §7 |
| AC: alignment with #187 atomicity | §9 |
| AC: follow-up issues per producer | §12.1 |
| AC: documentation and migration impacts | §12.2, §12.3 |
