# 0006. Budgeted evidence assembly: candidate model, ranking, budget, and API shape

- **Status**: Accepted
- **Date**: 2026-07-30
- **Issue**: #302
- **Related**: #216, #215, #220, #299, #300, #301, #303, #304, #305, #306, #307, #308,
  ADR 0004, ADR 0005 §8
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The candidate model, ranking discipline, budget semantics, deduplication and
diversity rules, contradiction handling, optional-reranker boundary, and public
API shape for #216's evidence-assembly layer — the twelve decisions #302
requires before #303 (candidate normalization), #304 (deterministic budgeted
selection), #305 (opt-in HTTP/MCP/SDK integration), #306 (SDK/MCP parity), and
#307 (optional reranker) can start without risking a later, backward-
incompatible rewrite of the response shape. #308's equal-budget evaluation
depends on this ADR naming stable metrics to compare against.

ADR 0005 (Accepted, 2026-07-30) already fixed the wire-contract rules this
design must satisfy — its own §8, "Constraints this hands to #216 / #302"
(`adr/0005-wire-contract-compatibility.md:378-398`), is binding here and is
quoted in full at §2.6. This ADR does not reopen any wire-contract
classification question ADR 0005 already answered; it only applies those
rules to this specific feature's shape.

No code changes ship with this ADR. It is design and decision only — #303
through #308 are the implementation.

Out of scope, filed to the issue that owns it:

- The optional reranker's concrete wire protocol (request/response JSON,
  environment variable names, timeout/retry defaults for an HTTP adapter) —
  #307. This ADR fixes the reranker's *boundary* (§12): what it may and may
  not do, and the provider-trait shape it must implement.
- Cross-context/cross-group evidence assembly (the `contexts?`/`groups?`
  pattern `POST /recall` and `POST /sources/search` already carry) — deferred
  entirely; §13.4 explains why not now.
- The concrete metric computation and evaluation harness — #308 owns the
  implementation; §14 only names what must be measured and against what
  baseline.
- Golden wire fixtures and the breaking-change CI guard — #301's job, orthogonal
  to this feature.
- Any answer-generation behavior. Taguru returns evidence, never prose; this
  ADR does not touch, and does not enable, an answer-generation code path.
- `langchain-taguru`'s own retriever fusion logic (§13.3 records why it is out
  of scope for v0.6.0 rather than silently ignoring it).

## 2. Context

### 2.1 The composed `retrieve` loop as it actually exists today

There is **no HTTP endpoint for `retrieve`**. It exists as three independent,
hand-synchronized client-side implementations, each re-running the same
five-step loop over ordinary single-call endpoints:

- The MCP composed tool, `run_retrieve`/`run_retrieve_bounded`
  (`src/mcp/retrieve.rs:34,63`), advertised in `tool_definitions()`
  (`src/mcp/schema.rs:561-589`) and deliberately exempted from the
  one-tool-one-route invariant the `every_advertised_tool_routes_to_a_request`
  test enforces (`src/mcp.rs:56-81`, exemption at `:69`) — `retrieve` is
  documented there as the sole hardcoded exception, because it "issues a
  variable number of requests built from earlier ones' results"
  (`src/mcp/retrieve.rs:41-52`).
- Python SDK `AsyncContext.retrieve`/`Context.retrieve`
  (`sdk/python/src/taguru/_async/client.py:1349`, generated `_sync` twin via
  `sdk/python/scripts/generate_sync.py`), reimplementing the same five steps by
  calling `self.resolve`/`self.describe`/`self.query`/`self.activate`/
  `self.cite_passage`/`self.search_passages` directly.
- TypeScript SDK `Context.retrieve` (`sdk/typescript/src/client.ts:1481-1497`),
  a third independent reimplementation.

`sdk/spec/surface.yaml:156-170`'s `retrieve` entry is the only operation in the
parity spec with no `route:` key — it has no single HTTP endpoint to point at.
A fourth, independent reimplementation exists outside the parity spec
entirely: `langchain-taguru`'s two retrievers
(`sdk/python-langchain/src/taguru_langchain/retrievers.py:45,152,501` and its
TypeScript twin) call `resolve`/`activate`/`cite_passage`/`search_passages`
directly and RRF-fuse the results themselves, never going through
`Context.retrieve()` at all (§13.3).

The composed result is an ad-hoc `serde_json::json!` object with no backing
Rust struct (`src/mcp/retrieve.rs:321-329`):

```rust
Ok(json!({
    "resolved": resolved, "outline": outline, "associations": associations,
    "activations": activations, "citations": citations,
    "passage_hits": passage_hits, "search_plan": search_plan,
}))
```

ADR 0005 §2.4 (`adr/0005-wire-contract-compatibility.md:129-133`) already
records this shape as the one MCP tool whose contract `mcp_contract` alone
owns, precisely because it has no HTTP shape to inherit.

Today's loop never calls `communities/search` — no community lane exists in
`retrieve` at all.

### 2.2 Candidate types and what provenance each already carries

Five wire types already carry most of what #216 asks a normalized candidate to
preserve. None of them are comparable to each other on their `score`/`weight`
field — that is the problem this ADR's §7 exists to solve, not something any
of these types claims to solve on their own.

| Lane | Wire type | Definition | Carries |
|---|---|---|---|
| Graph query | `AssociationOut` | `src/api.rs:1531-1545` | `subject, label, object`, averaged signed `weight: f64`, `count`, `attributions: Vec<AttributionOut>` |
| — attribution | `AttributionOut` | `src/api.rs:1517-1525` | `source, weight, count, paragraph: Option<u32>, section: Option<String>` |
| Graph activate | `ActivationOut` | `src/api.rs:1600-1605` | `strength: f64`, `path: Vec<String>` (origin-first graph path), `association: AssociationOut` |
| Passage (BM25/vector) | `PassageHit` | `src/api/sources.rs:426-433` | `source, paragraph, score, text, lanes: PassageLanes` |
| — per-lane evidence | `LaneEvidence` | `src/api/sources.rs:445-449` | `rank: usize` (1-based, within that lane's own pool), `score: f32` (that lane's raw score) |
| Citation | `Citation` | `src/api/sources.rs:100-105` | `text, source, section: Option<String>` |
| Community | `CommunityHit` | `src/api/communities.rs:235-258` | `community, score, text, paragraph, level?, parent?, concept_count?, members: Vec<CommunityHitMember>` |

`AssociationOut`'s ranking is already documented as intentionally *not*
score-comparable across unrelated queries: the graph page order is "strongest
`|weight|` first…, ties broken lexicographically on `(subject, label,
object)` — deliberately NOT insertion order" (`src/api.rs:1304-1315`, function
`rank`). `PassageHit.score` is "the fused reciprocal-rank number when the
semantic lane ran, the raw BM25 score otherwise" (`src/api/sources.rs:420-424`)
— already a fusion, and already lane-tagged via `PassageLanes.bm25`/`.vector`,
each an `Option<LaneEvidence>` carrying that lane's own 1-based rank
(`src/api/sources.rs:435-449`). `CommunityHit.score` is a third, unrelated
scale (`src/api/communities.rs:239`). None of `AssociationOut.weight`,
`ActivationOut.strength`, `PassageHit.score`, and `CommunityHit.score` share a
unit, a sign convention, or a range — #216's "do not compare raw scores"
requirement is a restatement of a fact already true of the existing wire
types, not a new constraint this ADR invents.

### 2.3 Deduplication today

`run_retrieve_bounded` runs two independent, exact-key dedup passes, both
`HashSet`-based, both local to one composed call:

1. Anchor dedup (`src/mcp/retrieve.rs:158-162`): linear containment check on
   the picked resolve name.
2. Association triple dedup (`src/mcp/retrieve.rs:184,198-206,222-235`): a
   `HashSet<(String, String, String)>` keyed by `(subject, label, object)` via
   `triple_of` (`src/mcp/retrieve.rs:12-18`). `query`'s copy of a triple wins
   over `activate`'s later copy — but the `ActivationOut` still appears
   verbatim under `activations`, so the strength/path evidence is not
   discarded, only the duplicate `Association` payload is.
3. Citation locator dedup (`src/mcp/retrieve.rs:245-265`): a
   `HashSet<(String, u64)>` on `(source, paragraph)`, deduplicated before
   dispatching any `cite_passage` call at all.

There is no cross-lane dedup (an association's attribution and a passage hit
at the same `(source, paragraph)` are never recognized as the same evidence),
no near-duplicate/text-similarity dedup, and no reranking anywhere in the
tree today.

### 2.4 Existing budget machinery, and the absence of token counting

The one existing byte-budget precedent is `run_retrieve_bounded`'s running
total (`src/mcp/retrieve.rs:63-85`): `spent += text.len()` after every
dispatched call, refusing the *next* call before it fires once the total
exceeds the caller's budget. It is documented as deliberately
over-counting — it sums raw response bytes, not the fields actually kept —
"so this can cut off a little early but never late" (`src/mcp/retrieve.rs:57-62`).
The HTTP MCP transport wraps this with `max_result_bytes` from
`TAGURU_MCP_MAX_RESULT_BYTES` (default 8 MiB,
`DEFAULT_MCP_MAX_RESULT_BYTES` at `src/env.rs:206`) and a second, final
post-hoc check on the whole composed value (`src/remote_mcp.rs:158-164`); the
stdio bridge passes no budget at all (`src/mcp/retrieve.rs:34-39`).

**No token counting exists anywhere in this tree.** Every existing budget is
bytes, item counts, wall-clock seconds, or semaphore permits:
`DEFAULT_RETRIEVAL_CACHE_BYTES = 32 MiB` (`src/registry/retrieval_cache.rs:47`),
`DEFAULT_MCP_MAX_RESULT_BYTES = 8 MiB` (`src/env.rs:206`),
`MAX_MATCH_LIMIT = 1000` (`src/api.rs:1078`),
`MAX_ASSOCIATIONS_PER_REQUEST = 10_000` (`src/api.rs:1090`),
`MAX_INPUT_ITEMS = 1000` (`src/api.rs:1100`), `MAX_ASSOCIATION_WEIGHT = 1e6`
(`src/api.rs:1123`), `MAX_NAME_BYTES = 1024` (`src/api.rs:1131`),
`MAX_CONTEXT_NAME_BYTES = 64` (`src/api.rs:1136`), the numeric ceiling funnel
`clamp(value, default, ceiling)` (`src/api.rs:1235`), and
`limits::enforce_timeout`/`HeavyOpsLimiter` (`src/limits.rs`, wall-clock and
concurrency only). §8's estimated-token budget is therefore new territory for
this codebase, not a reuse of an existing mechanism, and must be specified
precisely enough that two implementations produce the same number.

### 2.5 Plan and provider-telemetry precedent

The existing per-lane plan shapes this feature's `plan` field must match in
spirit: `SearchPlan { contexts: Vec<SearchContextPlan> }`
(`src/api/sources.rs:483-485`), `SearchContextPlan { context, lanes:
SearchLanesPlan, filter: Option<FilterPlan> }` (`:493-498`), `SearchLanesPlan
{ bm25: LanePlan, vector: LanePlan }` (`:517-520`), and `LanePlan { ran: bool,
reason: Option<String>, floor: Option<f32> }` (`:527-533`) — "it ran … or it
did not and `reason` says why, in the same prose the explain endpoint uses"
(comment above `LanePlan`). The graph side has its own, narrower plan,
`MatchPlan { contexts: Vec<String> }` (`src/api.rs:1258-1263`), with the
explicit note that "graph searches have no lanes" — there is nothing to fuse
on the graph side today, which is exactly why §7's RRF treats "graph query"
and "graph activate" as two separate lanes rather than borrowing
`MatchPlan`'s single-lane shape.

Provider identity and degrade-without-erroring precedent, both directly
reusable in §12: `EmbeddingsStatus { provider_model: Option<String>, glosses:
Option<GlossSidecarStatus>, passages: Option<PassageSidecarStatus> }`
(`src/registry.rs:1481-1512`, served at `GET /contexts/{name}/embeddings` via
`src/api/coverage.rs:45` and `src/registry/embeddings.rs:30`) exposes the
*configured* provider identity beside the `(model, width)` an artifact was
actually built with — the shape a reranker's "which model actually ranked
this" answer should follow. `EmbeddingProvider` (`src/embedding.rs:284-303`)
is the trait shape: `model(&self) -> &str`, a `deadline`-aware `embed(...)`,
and `breaker(&self) -> Option<&EmbedBreaker> { None }` defaulting to
breaker-free for mocks. Its breaker, `EmbedBreaker`
(`src/embedding.rs:69-276`), opens after `BREAKER_THRESHOLD = 3`
(`:73`) consecutive attempt failures, cools down `BREAKER_COOLDOWN = 30s`
(`:75`), and exposes states `Closed{consecutive_failures} / Open{since} /
HalfOpen{probing}` with a single-probe half-open admission gate. Retry policy
is bounded: `RETRY_ATTEMPTS = 2` (`:326`) with a 100 ms initial backoff,
5x per step. Metrics surface the breaker's state as a gauge plus
consecutive-failure/opened/short-circuit counters
(`src/metrics.rs:1207-1255`).

### 2.6 ADR 0005 §8 — the binding wire-contract constraints, quoted in full

`adr/0005-wire-contract-compatibility.md:378-398`:

> #302 designs the evidence-assembly API's candidate/budget/dedup/reranker
> semantics; this ADR does not reach into that. It does fix the wire-contract
> rules #302's design must satisfy, since #216 ships opt-in and therefore adds
> without changing anything existing (`http_contract` stays `1`):
>
> - Responses use the existing `ApiResponse<T>` envelope — no new top-level
>   envelope shape.
> - New failure modes are expressed as additional `ErrorCode` variants inside
>   an existing HTTP status class (§4), never a new status-to-meaning mapping.
> - Any selection-trace or provenance field with a closed set of values is
>   designed open from the start (plain string on the wire, and the
>   TypeScript model must not use a closed literal union — §5's rule applies
>   to new fields, not only retrofits).
> - If the response needs pagination, it reuses one of the thirteen existing
>   envelope shapes (§2.3) rather than minting a fourteenth.
> - The MCP surface for this feature is an ordinary routed tool, inheriting
>   `http_contract` via pass-through (§2.4) — it does not grow a second
>   `retrieve`-style ad-hoc composed shape unless a real composition need
>   forces it.

§4-6 below satisfy each bullet explicitly; §2.6 exists so a reader of this ADR
never has to cross-reference ADR 0005 to check compliance.

## 3. Options considered

### A. Public surface: new endpoint vs. opt-in `retrieve` argument vs. both

- **New HTTP endpoint, `POST /contexts/{name}/evidence` — adopted.** Retrieval
  fan-out and assembly both run server-side, in one round trip, behind one
  typed request/response pair.
- Opt-in argument on the existing composed `retrieve` — rejected. `retrieve`
  has no server-side implementation to extend (§2.1); "opt-in on retrieve"
  would mean writing the assembly algorithm three times (MCP bridge, Python
  SDK, TypeScript SDK) and proving all three stay bit-for-bit deterministic
  against each other, which is strictly harder than writing it once on the
  server. It would also be permanently unreachable to a caller that talks
  HTTP directly without an SDK.
- Both — rejected for the same reason as above, plus ADR 0005 §8's last
  bullet: "it does not grow a second `retrieve`-style ad-hoc composed shape
  unless a real composition need forces it." A real HTTP endpoint removes the
  need to ever touch `retrieve`'s ad-hoc shape at all.

### B. Cross-lane comparability

- Inventing a shared raw-score scale across graph weight, BM25/cosine, and
  community score — rejected; #216 forbids it explicitly, and §2.2 shows the
  four scales share no unit today.
- **Reciprocal-rank fusion over each lane's own internal rank — adopted**
  (§7). Every lane already produces an internal, comparable rank (the graph
  page order, `LaneEvidence.rank`, or an equivalent), so RRF needs no new
  scoring model, only a rank extraction per lane.
- Reserve cross-lane ordering for the optional reranker only, with no
  deterministic fallback ordering — rejected; #216 requires a stable
  no-provider ordering, and one candidate item may have no reranker access if
  none is configured, deliberately the default state (§12).

### C. Budget units

- Bytes only — rejected; #216 explicitly names an "estimated-token budget" as
  a first-class concept.
- Estimated tokens only — rejected; no tokenizer dependency exists in this
  tree (§2.4), and pinning one drags a specific vocabulary into a
  provider-agnostic feature.
- **Bytes, an estimated-token count, and a hard item ceiling, all three
  independent hard limits — adopted** (§8). Matches the existing house style
  of `clamp`-bounded, independently-named ceilings (`MAX_MATCH_LIMIT`,
  `MAX_ASSOCIATIONS_PER_REQUEST`, …) rather than one derived number.

### D. Overflow policy

- Stop assembling at the first candidate that would overflow the budget —
  rejected; a large, low-relevance candidate early in fused order would
  starve every smaller, more relevant candidate behind it, and #216 requires
  omitted counts/reasons to be observable rather than an implicit truncation
  point.
- **Skip the over-budget candidate and continue evaluating the rest —
  adopted** (§9). Matches `run_retrieve_bounded`'s own "cut off early, never
  late" philosophy (§2.4) generalized from "stop the whole call" to "skip one
  item."

### E. Where citations live

- Treat a citation as its own candidate, competing for budget on equal
  footing with associations/passages/communities — rejected; a citation has
  no independent relevance rank of its own (§2.2's `Citation` type carries no
  score), and letting the same source text be selected once as a passage hit
  and again as a citation double-spends budget on identical bytes.
- **A selected item carries `citation_refs` (locators only); the citation
  text itself lives exactly once, in the package's top-level `citations`
  list — adopted** (§6, §10). This is also how #216's "never emit an evidence
  item whose cited source is omitted from the package" and "do not count the
  same source-derived assertion as independent corroboration in several
  representations" acceptance criteria become structurally true rather than
  something the selection algorithm has to remember to check.

### F. Diversity mechanism

- Maximal marginal relevance (MMR) over a learned or invented similarity
  metric — rejected; it needs exactly the cross-lane comparable scale §B
  forbids, or a second, separately-specified similarity function this ADR
  would then also have to freeze.
- A hard per-source cap (e.g. "at most 2 items per source") — rejected; too
  coarse when one source dominates a genuinely narrow query, and it would
  silently drop the single most relevant item from an over-represented source
  in favor of a weaker item from another, which is a worse selection, not a
  more diverse one.
- **Tier-based round-robin admission — adopted** (§9). Candidates are grouped
  into fixed-width tiers by fused rank; within a tier, a source that has not
  yet appeared in the package is preferred over a repeat source. This changes
  *admission order*, never relevance ranking, and needs no new scale.

### G. Where the reranker sits in the pipeline

- Reranking after selection (re-order the already-budgeted package) —
  rejected; a candidate the reranker would have promoted into the package may
  already have been omitted for budget before the reranker ever saw it.
- Reranking *inside* selection, with the ability to add or drop candidates —
  rejected; #216 requires the deterministic policy to remain provider-free
  and stable, and letting a reranker mutate membership (not just order) makes
  every §9 invariant conditional on reranker behavior instead of universal.
- **Reranking before selection, strictly reordering the candidate pool the
  deterministic selector then runs over — adopted** (§12). The reranker's
  entire effect is "which candidate the greedy selector considers first";
  every invariant in §9 holds identically whether or not a reranker ran.

## 4. Decision

Add one new opt-in HTTP endpoint, `POST /contexts/{name}/evidence`, that
gathers graph associations, graph activations, passage hits, and (opt-in)
community hits for one context, normalizes each into a typed candidate
carrying full provenance, ranks candidates by reciprocal rank fusion over
each lane's own internal order (optionally reordered by a configured
reranker, never restructured by one), and greedily selects a deterministic,
provenance-preserving, citation-complete evidence package under three
independent hard budgets — byte, estimated-token, and item-count — skipping
over-budget candidates rather than stopping early, protecting corroborating
and contradicting evidence from silent collapse, and reporting exactly what
was omitted and why. The same feature is exposed as one ordinary MCP routed
tool and one typed method on each SDK. `http_contract` and `mcp_contract`
both stay `1`; every existing endpoint, tool, and SDK method is unchanged.

## 5. Public API shape

### 5.1 HTTP

`POST /contexts/{name}/evidence` → `ApiResponse<EvidencePackage>` (§10), using
the existing `ApiResponse<T>` envelope (`src/api.rs:77-91`) per ADR 0005 §8's
first bullet.

Request body, every field but `origins` optional — an all-optional-but-one
body is itself an ordinary additive shape (ADR 0005 §4: "add an optional
request field" is compatible), so later options can grow the same way:

```jsonc
{
  "origins": "or [\"...\"]",             // required, same contract as retrieve's origins
  "labels": "or [\"...\"]",               // optional, same as retrieve
  "dice_floor": 0.6,                      // optional
  "semantic_floor": 0.6,                  // optional
  "resolve_limit": 20,                    // optional
  "activate_decay": 0.5,                  // optional
  "activate_limit": 20,                   // optional
  "text_fallback_query": "...",           // optional
  "search_limit": 5,                      // optional
  "include_communities": false,           // optional, default false (§6)
  "budget": {                             // optional; §8 gives every default
    "max_bytes": 65536,
    "max_tokens": 4000,
    "max_items": 40
  },
  "rerank": { "provider": "..." }                 // optional, §12; absent = no reranker
}
```

Dedup and diversity carry no request-level tuning: like §7's RRF constant,
their policy is a single fixed rule the implementation documents once (§9),
not a per-call knob — nothing in #302's or #216's acceptance criteria asks a
caller to tune either, and a config surface with no named consumer is
scope this ADR does not add. A future caller need for tuning is an ordinary
additive request field later (ADR 0005 §4), the same way §13.4 defers
cross-context support rather than speculatively building it now.

`origins` reuses `retrieve`'s own cue-list contract, including the
`MAX_ORIGIN_CUES = 1000` ceiling (`src/mcp/retrieve.rs:28`) — a caller
migrating from `retrieve` changes nothing about how it names origins.

### 5.2 MCP

One ordinary tool, `assemble_evidence`: a `tool_definitions()` entry
(`src/mcp/schema.rs`) plus a `route_tool` arm (`src/mcp/route.rs`) that maps
straight onto the new HTTP endpoint, exactly like every other tool but
`retrieve`. This is the ordinary case ADR 0005 §8's last bullet asks for; the
`retrieve`-exemption count in `src/mcp.rs:69` stays at exactly one. `assemble_evidence`
inherits `http_contract` by pass-through the way 45 of the 46 existing tools
already do (ADR 0005 §2.4).

### 5.3 SDKs

`sdk/spec/surface.yaml` gains an `assemble_evidence` entry *with* a `route:`
key (unlike `retrieve`'s entry), keeping it inside the CI-enforced parity
check both `check_surface.py` and `check-surface.ts` already run. Python adds
`async def assemble_evidence(...)` to `_async/client.py` only; the sync
client is regenerated via `scripts/generate_sync.py`, never hand-edited.
TypeScript adds `assembleEvidence(...)` to `client.ts`, a named
`EvidencePackage` interface (unlike `retrieve`'s inline options object, this
response is complex enough — and reused across three languages — to warrant a
named exported type) added to `models.ts` and re-exported from `index.ts`.
Every enum-like field on the TypeScript side uses `Open<T>`
(`sdk/typescript/src/models.ts:116`), never a closed literal union, per ADR
0005 §5 and §8's third bullet.

### 5.4 Cross-cutting

Auth, rate limiting, and heavy-ops gating match the read endpoints this
feature composes: an ordinary context-scoped key check (not the heavy-ops
limiter — this is a bounded composition of ordinary reads, not a
compute-heavy admin operation like `compact` or `communities`), and the same
`Deadline` extension every other handler in `src/api/*.rs` checks before and
during its work. No new `ErrorCode` variant is introduced (§11).

## 6. Candidate model (for #303)

A candidate's `kind` is one of `association`, `passage`, or `community` — a
plain, open string on the wire per ADR 0005 §8's third bullet, not a closed
enum. **A citation is not a fourth kind.** Every item — regardless of kind —
carries a `citation_refs: [{source, paragraph}]` list of locators; the
citation *text* itself is never duplicated per item. It lives exactly once,
keyed by `(source, paragraph)`, in the package's top-level `citations` array
(§10). This single structural decision is what makes two of #216's acceptance
criteria true by construction rather than by selection-logic vigilance:
"never emit an evidence item whose cited source is omitted from the package"
(there is nothing per-item to omit — the shared list either has the entry or
it doesn't, and §9's admission rule keeps them atomic) and "do not count the
same source-derived assertion as independent corroboration in several
representations" (two items citing the same `(source, paragraph)` share one
citation entry, so nothing about the shared citation can inflate a
corroboration count).

Every candidate, regardless of `kind`, carries:

- `context` — the context this call targeted (single-context only; §13.4).
- `candidate_id: String` — an opaque, deterministic identity every candidate
  has, including one with no locator at all: `kind` NUL-joined with §7's
  `canonical_key` (`"association\u0000{subject}\u0000{label}\u0000{object}"`
  or `"passage\u0000{context}\u0000{source}\u0000{paragraph}"`, etc.) — the
  same NUL-delimited convention `citationKey`
  (`sdk/typescript/src/models.ts:833-839`) already uses for exactly this
  reason: no field value can collide with the delimiter. This is the field
  every cross-item reference in §10 (`contradicts`, `duplicate_of`,
  `OmittedCandidate`) points at — a locator alone cannot name an unsourced
  association candidate, which has no `(source, paragraph)` of its own.
- `source: Option<String>`, `paragraph: Option<u32>`, `section: Option<String>`
  — for a passage/community candidate, straight from the hit
  (`PassageHit`/`CommunityHit`, §2.2). For an association/activation
  candidate, which may carry several attributions from different sources,
  the *primary* attribution's fields: deterministically the attribution
  with the greatest `|weight|` among this one candidate's own
  `attributions` (a within-candidate comparison over values that already
  share a scale — not the cross-candidate, cross-lane comparison §6's
  discipline forbids), ties broken by the lexicographically smallest
  `source` then smallest `paragraph`. An association/activation candidate
  with zero attributions (an unsourced graph fact) carries `None` for all
  three; §9's diversity step treats `None` as its own always-novel bucket,
  never counted as a repeat of a named source or of another unsourced
  candidate.
- `lane` — which retrieval call produced it: `graph_query`, `graph_activate`,
  `passage_bm25`, `passage_vector`, `passage_fused`, or `community`. A
  `PassageHit` whose `PassageLanes` carries both `bm25` and `vector` is
  `passage_fused`; carrying only one is that lane's own name.
- `lane_rank: usize` — the 1-based rank *within that lane's own pool*, the one
  number §7 is allowed to compare across lanes (only after RRF, never
  directly).
- `graph_path: Option<Vec<String>>` — `ActivationOut.path` verbatim
  (`src/api.rs:1600-1605`) when the candidate came from `graph_activate`;
  `None` otherwise. Never invented for a lane that has no path concept.
- `signed_weight: Option<f64>` — `AssociationOut.weight` / `ActivationOut.strength`
  for graph candidates; `None` for passage/community, which have no signed
  weight at all.
- `citation_refs: Vec<{source, paragraph}>` — every attribution locator this
  candidate's underlying evidence carries. Empty for an unsourced graph fact
  or a passage/community hit whose own `(source, paragraph)` is already the
  item's own locator (self-citing — no separate reference needed).
- `origins: BTreeSet<(String, u32)>` — the full set of `(source, paragraph)`
  pairs this candidate's evidence is *attributed to*: for an
  association/activation candidate, every attribution `AssociationOut`'s own
  `attributions` list already carries, aggregated server-side into one
  candidate before this feature ever runs (§2.2 — the graph engine, not this
  ADR's dedup step, is what merges multiple sources asserting the same
  `(subject, label, object)`); for a passage/community candidate, the
  singleton `{(source, paragraph)}` of the hit itself. §9 renders this set
  as the `corroboration` field (§10) so a fact several independent sources
  assert is never summarized down to one opaque count. Never serialized to
  the wire as `origins` itself — `corroboration` and `citation_refs` are its
  two public projections.

Corpus body text and API credentials are never duplicated into a diagnostic
field — an `omitted` entry (§10) names a candidate by its `candidate_id`
(never by re-embedding its text), which, unlike a locator, still identifies
an unsourced graph candidate with no attribution at all.

Comparison discipline, restated as a hard rule: nothing outside §7's RRF step
may compare the four incomparable score fields named in §2.2 against a
candidate of a different `kind`. Comparing two candidates' `lane_rank` values
is only valid when they come from the same lane.

## 7. Ranking

Reciprocal rank fusion: a candidate's `fused_score = 1 / (k + lane_rank)` for
a single, fixed `k` chosen and documented once by #303/#304's implementation
— not request-configurable, and not required to match any particular value,
since `fused_score` itself is never serialized (only the ordinal `fused_rank`
reaches the wire, §10) and this server-side implementation has no second,
independent reimplementation that needs to agree on the same raw number the
way `retrieve`'s three client-side copies would have (§2.1). Summed across
every lane the same underlying evidence appeared in (a
candidate whose `origins` already merged two lane appearances via §9's dedup
step sums both lanes' contributions before this step runs, not after).
Fused order sorts descending by `fused_score`; ties break on
`(kind, canonical_key)` lexicographic order, where `canonical_key` is
`(subject, label, object)` for an association/activation candidate and
`(source, paragraph)` for a passage/community candidate — fully deterministic,
matching the graph page's own existing tie-break discipline
(`src/api.rs:1304-1315`). This is the same `canonical_key` §6's
`candidate_id` is built from — one definition, two consumers.

Each lane's `lane_rank` is read from what that lane's endpoint already
produces, never recomputed: graph-query candidates keep the `AssociationOut`
page order (`src/api.rs`'s `rank`, §2.2); graph-activate candidates keep
`ActivationOut`'s strength-sorted order; passage candidates keep the incoming
`PassageHit` order (`src/api/sources.rs:420-424` — already RRF-fused
internally between BM25 and vector; this step does **not** re-fuse
`PassageLanes.bm25`/`.vector`, it treats the whole `PassageHit` as one lane
whose rank is its position in the incoming list); community candidates keep
`CommunityHit`'s incoming order.

The response's `fused_rank` field (§10) is documented on the wire, in its own
doc comment, as incomparable to any lane's raw score — restating §6's
comparison discipline at the one place a caller is most likely to reach for
a raw number instead.

## 8. Budget semantics

Three independent hard ceilings — `max_items`, `max_bytes`, `max_tokens` —
none a priority over another; whichever is reached first is binding, and
reaching any one of them stops admitting further candidates. Every default
and ceiling routes through the existing `clamp(value, default, ceiling)`
funnel (`src/api.rs:1235`), matching the house style of every other numeric
limit in `src/api.rs`.

- `max_items`: default 40, ceiling 1000 (matching `MAX_MATCH_LIMIT`,
  `src/api.rs:1078`).
- `max_bytes`: counts the compact (no extraneous whitespace) UTF-8
  serialization length of exactly the `items` array plus the `citations`
  array, **excluding each item's own `bytes` and `estimated_tokens` fields**
  (§10) — those two fields each report a count *derived from* this same
  measurement, so counting them as part of what they measure would make the
  measured length depend on its own value (a longer number takes more bytes
  to write, which would change the length, which could change the number).
  Never the response envelope, `plan`, `budget`/`omitted` metadata, or those
  two per-item accounting fields. A client computing the same sum over the
  same JSON, minus those two keys, gets the same number; this is a closed,
  checkable definition, not an implementation detail left to whichever
  server built the package. Concretely, an implementation computes every
  candidate's content byte count first (the exclusion above), uses that
  fixed number for every admission decision (§9), and only stamps the
  admitted item's own `bytes`/`estimated_tokens` fields afterward — they
  ride along in the response as informational metadata, never fed back into
  the ceiling check that already ran. Default 65536 (64 KiB), ceiling chosen
  well under §2.4's `DEFAULT_MCP_MAX_RESULT_BYTES` (8 MiB) so that a package
  built at this endpoint's own ceiling never gets silently truncated a
  second time by the MCP transport's independent cap — 1 MiB.
- `max_tokens`: an **estimate**, not a real tokenizer count — this codebase
  deliberately carries no tokenizer dependency (§2.4). The estimator is fixed
  by this ADR and is itself part of the wire contract: changing its formula
  changes what a given package's `budget.tokens_used` means without changing
  its type, which ADR 0005 §4 classifies as "change a field's meaning …
  without changing its type" — **breaking**, "the worst kind." The estimator:
  for each Unicode scalar value in the counted text (the same `items` +
  `citations` slice `max_bytes` counts), add 0.25 tokens for a scalar in the
  Basic Latin block (`U+0000`-`U+007F`) and 1.0 token for any other scalar
  (covering CJK ideographs, kana, hangul, and everything else at a
  conservative one-token-per-character estimate); the estimated total is the
  ceiling of the sum. This is deliberately biased toward *overestimating* for
  non-Latin scripts rather than reusing a bytes-per-token heuristic tuned for
  English — the corpora this project ships and tests against (§2's own
  fixtures, `adr/0001/corpus/`) are Japanese-heavy, and a bytes/4-style
  estimate undercounts a UTF-8-multibyte-heavy corpus badly enough to make
  the budget meaningless. Default 4000, no server-enforced ceiling beyond
  `max_items`/`max_bytes` implicitly bounding it (an operator wanting a
  harder cap sets a smaller `max_bytes`).

Zero and near-zero budgets are ordinary, valid input — not an error. A
`max_items: 0` (or a `max_bytes`/`max_tokens` too small for even the smallest
candidate) call returns an empty `items`/`citations` package with every
candidate that would otherwise have been considered listed under `omitted`
with `reason: "budget_exceeded"` (§10) — never a 4xx.

## 9. Selection algorithm (for #304)

Pipeline: normalize → exact-key dedup → contradiction grouping → rank (§7) →
near-duplicate suppression (using that rank) → optional reranker reordering
(§12) → diversity-aware greedy admission under budget (§8) → trace assembly.
Near-duplicate suppression is staged *after* ranking specifically because its
survivor rule needs a fused rank to compare candidates by — putting it
earlier, before §7 has produced one, would leave "keep the higher-ranked
candidate" undefined at the point it runs.

**Exact-key dedup.** Association candidates dedup on `(subject, label,
object)` after alias resolution (the same normalization `resolve`/`query`
already apply). Passage and community candidates dedup on `(context, source,
paragraph)` — the identical locator can never appear twice regardless of
which lane produced it (a passage hit and a community hit can share a
`(source, paragraph)` only if community summaries somehow shared a source id
space with ordinary passages, which they do not — `CommunityHit.community`
is always prefixed `community:{id}`, so this case is structurally
impossible, not merely handled). A passage hit and a citation sharing
`(source, paragraph)` resolve to one candidate whose `citation_refs` already
covers itself — no separate citation duplicate is created for a candidate
that already self-cites. This step needs no rank and nothing to compare
scores by — it only ever collapses two *identical*-key appearances of the
same underlying evidence, never two different pieces of evidence into one
(see below).

**Corroboration is never lost.** An association/activation candidate's
`origins` (§6) is exactly the attribution set `AssociationOut.attributions`
already carries — aggregated server-side, by the graph engine, before this
feature ever runs (§2.2). Two lane appearances of the same `(subject, label,
object)` triple within one call (e.g. the same edge surfacing from both
`query` and `activate`) therefore always carry the *identical* attribution
set, never a disjoint one — exact-key dedup above can only ever discard a
duplicate copy of the same evidence, never fold two independently-sourced
candidates into one. What actually protects corroboration is narrower and
structural: dedup never truncates an admitted candidate's attribution list,
so the `corroboration` field (§10) always names every contributing source in
full — a fact two sources independently assert is never summarized down to
a single opaque count, and there is no "merge decision" for this ADR to get
wrong.

**Contradiction.** A candidate contradicts another when they share
`(subject, label)` but disagree on `object`, or when a candidate's
`signed_weight` is negative and it shares `(subject, label, object)` with a
positive-weight candidate. Contradiction is not always pairwise — three
candidates sharing one `(subject, label)` with three different `object`
values all mutually contradict each other. The transitive closure of
"contradicts" over a set of candidates is one **contradiction group**;
membership rides on each member's `contradicts` field (§10) as the
`candidate_id`s of every *other* member. **A contradiction group is one
admission unit throughout diversity tiering and budget admission** — never
tiered or admitted separately, so no member is ever split apart from the
rest of its group by either mechanism below. Concretely: diversity tiering
never delays one member of a group while advancing another, and admission
decides the whole group together — if it does not all fit, none of it is
admitted, and every member goes to `omitted` with `reason:
"contradiction_group_exceeds_budget"`. This preserves #216's requirement to
"preserve explicit negative-weight/contradictory evidence instead of
silently selecting only the majority claim" — a caller sees every side of a
live disagreement or none of them, never a one-sided majority view. This
step needs no rank either — it only compares candidates' own
`(subject, label, object)`/`signed_weight` fields, not a cross-candidate
score.

**Near-duplicate suppression.** Beyond exact-key dedup, a fixed detector also
drops textually redundant *passage* candidates, staged after §7 has ranked
the pool: #304 fixes and documents one deterministic similarity function
over normalized text and a single default threshold — not
request-configurable, for the same reason §7's RRF constant is fixed rather
than tunable: no wire field ever exposes a raw similarity score, so nothing
needs the exact function or threshold pinned in this ADR, only that it be
deterministic and documented once, and that it run after §7 so "keep the
higher-ranked candidate" is well-defined. A pair the function calls
near-duplicate keeps the higher-`fused_rank` candidate and omits the other
with `reason: "duplicate_passage"` and a `duplicate_of` (§6's `candidate_id`
of the survivor).

**Diversity.** Tier-based round-robin, not a re-ranking: admission units (an
ordinary candidate, or a contradiction group per above, counted as one) are
grouped into tiers of fixed width, `tier_width = max(1, max_items / 4)` — a
single fixed rule, not request-configurable, for the same reason §7's RRF
constant is fixed rather than tunable — by their position in the order the
pipeline has produced by this point: §7's fused order after near-duplicate
suppression, or a configured reranker's permutation of it when one ran
(§12) — never a second, independent ordering pass, so tier 0 always holds
whatever candidates are first in that one order, tier 1 the next
`tier_width`, and so on. Within one tier, a unit whose primary source (§6)
has not yet appeared anywhere in the package-so-far is admitted ahead of a
unit whose source has already appeared once in that same tier — a
same-tier, same-source second unit is pushed to the back of its own tier,
never into a different tier and never past a later tier's higher-ranked
unit. An unsourced candidate (§6's `None` bucket) is never treated as
repeating another unsourced candidate. This is deliberately weaker than MMR
(§3 F): it changes *admission order inside a tier*, never overall relevance
rank, and needs no invented similarity metric.

**Admission.** Walk the same order diversity tiering just adjusted, once,
greedily, one admission unit at a time. For each unit: if admitting it —
together with any `citation_refs` locators not already present in the
package — would exceed any of the three §8 budgets, skip it (`reason:
"budget_exceeded"` for an ordinary candidate, or the contradiction-specific
reason already named above for a group) and continue to the next unit rather
than stopping (§3 D). A citation locator already present in the package
(from an earlier-admitted unit) costs nothing extra when a later unit reuses
it — `citations` dedups by `(source, paragraph)` the same way §9's own
dedup does.

**Invariants**, holding identically whether or not a reranker ran (§12):

- **I1 — no orphan citations, in both directions.** Every `citation_refs`
  entry on an admitted item has a matching entry in the package's
  `citations`, and every entry in `citations` is referenced by at least one
  admitted item.
- **I2 — budgets are hard ceilings.** `items.len() <= max_items`, the
  `max_bytes`-defined byte count of `items` + `citations` never exceeds
  `max_bytes`, and the §8 token estimate over that same slice never exceeds
  `max_tokens`.
- **I3 — determinism.** The same request against the same corpus revision
  produces byte-identical JSON, field order included, every time.
- **I4 — corroboration is never silently collapsed.** An admitted
  association/activation item's `corroboration` (§10) names every source in
  its underlying `origins` (§6) — dedup and near-duplicate suppression never
  truncate that list down to a single opaque count.
- **I5 — contradiction groups are atomic.** No member of a contradiction
  group (§9) appears in `items` without every other member that exists as a
  candidate also appearing, and vice versa — a negative-weight association's
  positive-weight counterpart, or any one of several same-`(subject,
  label)` candidates with differing `object`s, is never partially admitted.
- **I6 — every omission is explained.** Every candidate not present in
  `items` appears in `omitted` (subject to §10's own listing cap) or is
  counted in `omitted_total`/`omitted_by_reason` even when the itemized list
  is capped.

## 10. Response shape

```rust
struct EvidencePackage {
    items: Vec<EvidenceItem>,
    citations: Vec<CitationEntry>,           // {source, paragraph, citation: Citation}
    budget: BudgetUsage,                     // {items_used, bytes_used, tokens_used, limits: BudgetLimits}
    omitted: Vec<OmittedCandidate>,          // capped, like MAX_LISTED_ISSUES (src/api.rs:437)
    omitted_total: usize,                    // always exact, never capped
    omitted_by_reason: BTreeMap<String, usize>, // always exact, never capped
    plan: EvidencePlan,
}

struct EvidenceItem {
    candidate_id: String,       // §6's opaque, deterministic identity
    kind: String,               // open: "association" | "passage" | "community" | future values
    fused_rank: usize,
    lane_ranks: Vec<LaneRankEntry>,   // {lane: String, rank: usize} — one per contributing lane
    citation_refs: Vec<CitationRef>, // {source, paragraph}
    #[serde(skip_serializing_if = "Option::is_none")]
    corroboration: Option<Corroboration>,   // {sources: Vec<String>, attributions: Vec<CitationRef>}
    #[serde(skip_serializing_if = "Vec::is_empty")]
    contradicts: Vec<String>,   // candidate_ids of every item this one contradicts (§9); empty when none
    bytes: usize,                // this item's content-only §8 byte contribution — excludes
                                 // this field and estimated_tokens themselves (§8)
    estimated_tokens: usize,     // likewise, this item's §8 token-estimate contribution
    // kind-specific payload, embedding the EXISTING wire type verbatim —
    // no parallel type is minted:
    association: Option<AssociationOut>,
    passage: Option<PassageHit>,
    community: Option<CommunityHit>,
}

// {candidate_id, kind, reason, duplicate_of: Option<String>} — `duplicate_of`
// (present only for reason: "duplicate_passage") names the surviving
// candidate_id, never a locator, so an unsourced candidate can appear on
// either side of the reference.
struct OmittedCandidate { /* … */ }

struct EvidencePlan {
    lanes: EvidenceLanesPlan,     // {resolve, query, activate, passages, communities, citations: LanePlan}
    selection: SelectionPlan,     // {dedup_dropped, contradiction_groups, diversity_tier_width}
    reranker: RerankerPlan,       // §12; {configured: bool, ran: bool, model: Option<String>, reason: Option<String>}
}
```

`EvidencePlan.lanes` reuses the existing `LanePlan { ran, reason, floor }`
shape (`src/api/sources.rs:527-533`) per-lane, so a caller already familiar
with `sources/search`'s `plan.contexts[].lanes` reads this the same way. The
`communities` lane's `LanePlan.ran = false` with `reason` naming "no
derived-communities artifact built" is a **degrade, not an error** (§11) —
deliberately different from `communities/search`'s own behavior of refusing
outright when no artifact exists, because here community evidence is one
opt-in input among several rather than the entire point of the call.

`items[]` embeds the *existing* wire types (`AssociationOut`, `PassageHit`,
`CommunityHit`) verbatim as the kind-specific payload rather than inventing
parallel, evidence-specific mirrors of them — exactly one of the three
payload fields is present per item, selected by `kind`. `omitted[]` itself is
capped the same way `Issue` lists are capped by `MAX_LISTED_ISSUES = 20`
(`src/api.rs:437`) — bounding response size for a caller that overwhelmingly
wants to know *whether* truncation happened, not read every dropped
candidate — but `omitted_total` and `omitted_by_reason` are never capped,
so the itemized list's own truncation is itself always observable, per
#216's "return omitted counts/reasons so truncation is observable rather than
silent."

Every optional field uses `#[serde(skip_serializing_if = "Option::is_none")]`
(or the `Vec`/bool equivalents already used elsewhere in `src/api/*.rs`),
never emitting `null`, per ADR 0005 §5. `kind`, every `lane` string, and
`OmittedCandidate.reason` are plain, open strings — never a closed Rust enum
serialized as a fixed set of variants, and never (on the TypeScript side) a
closed literal union — per ADR 0005 §8's third bullet.

## 11. Errors

**No new `ErrorCode` variant is introduced.** Every failure mode this
endpoint can produce already has a home in the existing 25-variant vocabulary
(`src/api.rs:150-207`): a malformed `budget`/`rerank` object is
`invalid_argument` (400, same as any other malformed request
field); an oversized `origins` list is `over_limit` (400, the same code
`retrieve`'s own `MAX_ORIGIN_CUES` check would use if it were server-side);
whatever the composed `resolve`/`query`/`activate`/`search_passages`/
`citations` calls themselves would return on failure propagates unchanged.
This is a deliberate instance of ADR 0005 §4's "add a new `ErrorCode`
variant that keeps an existing status" pattern applied in the negative — no
new variant was needed at all, which is stronger evidence of compatibility
than adding one compatibly would have been.

Two situations that might look like errors are **degrades**, not errors, and
never produce a non-2xx response:

- No derived-communities artifact exists for a context that requested
  `include_communities: true` — `plan.lanes.communities` reports
  `{ran: false, reason: "no derived communities artifact; run `taguru
  communities` first"}` and the call proceeds with every other lane.
- A configured reranker is unreachable, times out, or returns an invalid
  result — `plan.reranker` reports the failure (§12) and selection proceeds
  under the deterministic §9 order.

## 12. Reranker boundary (for #307)

The reranker's provider trait follows `EmbeddingProvider`'s shape
(`src/embedding.rs:284-303`), not `ChatClient`'s (`src/extract.rs:2182`) —
this is an interactive read path serving one caller's request synchronously,
the same posture embeddings serve, not extraction's batch-oriented,
higher-retry posture:

```rust
struct RerankCandidate {
    kind: String,       // same open kind as §6
    lane_rank: usize,
    text: String,       // the candidate's own text — the passage text, community
                         // summary paragraph, or a plain rendering of
                         // subject/label/object for an association
}

trait EvidenceReranker: Send + Sync {
    fn model(&self) -> &str;
    /// Returns a permutation of `0..candidates.len()`: the candidate
    /// indices in the reranker's preferred order. Anything other than a
    /// complete permutation of the input indices is treated as an invalid
    /// response (§ below).
    fn rerank(&self, query: &str, candidates: &[RerankCandidate], deadline: Deadline)
        -> Result<Vec<usize>, String>;
    fn breaker(&self) -> Option<&EmbedBreaker> { None }
}
```

`query` is canonical and deterministic regardless of how the request phrased
`origins`: it is `text_fallback_query` verbatim when the request supplied
one (already phrased as a natural-language query, the same text used for the
passage-search fallback); otherwise it is `origins` — normalized to its list
form whether the request sent a bare string or an array — joined with `"; "`
in request order. A request whose `origins` is a single string and a request
whose `origins` is that same string as a one-element array therefore rerank
identically.

**A reranker may only reorder.** It receives the full candidate pool — after
§9's exact-key dedup, contradiction grouping, §7's ranking, and
near-duplicate suppression have already run, immediately before
diversity-aware admission — and returns a strict permutation of that pool's
indices; it cannot add, remove, edit, or merge candidates. If `rerank`
returns anything that is not a complete permutation of
`0..candidates.len()` — wrong length, an out-of-range index, or a repeated
index — the entire result is discarded and selection falls back to §7's
deterministic RRF order (the same order near-duplicate suppression already
used), with `plan.reranker.reason = "invalid_permutation"`. Every invariant
in §9 (I1-I6) holds identically whether the candidate order came from §7
alone or from a reranker's permutation of it — reordering the input to a
deterministic, invariant-preserving admission process cannot itself violate
an invariant.

Provider policy mirrors `EmbeddingProvider`, not `ChatClient`: per-attempt
timeout is `min(configured_timeout, deadline.remaining())`
(`src/embedding.rs:478`'s pattern), bounded retry on transient failures only
(matching `RETRY_ATTEMPTS`/backoff shape at `src/embedding.rs:326-327`), and
an `EmbedBreaker`-shaped circuit breaker (`src/embedding.rs:69-276`) so a
persistently failing reranker stops being tried mid-burst rather than adding
latency to every subsequent call in the same outage. Metrics follow the
`taguru_embedding_breaker_*` naming shape (`src/metrics.rs:1207-1255`) with a
`rerank` prefix instead of `embedding`.

No credential or network access is required by default: with no reranker
configured, `plan.reranker = {configured: false, ran: false}` and every other
behavior in this ADR is completely unaffected — the deterministic §9 pipeline
is not merely a fallback, it is the whole feature in that configuration, per
#216's requirement that Taguru "must continue to work without a reranker
provider."

**Privacy.** Candidate text is sent to a configured reranker provider and
nowhere else — never into a log line, an error message, an `omitted` entry,
or a metric label. The only reranker-identifying information that reaches
the response or metrics is the model identity string (`RerankerPlan.model`);
no API key or credential value is ever placed on the wire or in a diagnostic
string, matching the existing discipline around `TAGURU_EXTRACT_API_KEY`/
`TAGURU_EMBED_API_KEY` never appearing in `AttemptRecord` or embedding
metrics.

The reranker's concrete wire protocol — request/response JSON shape for an
HTTP-backed adapter, its specific environment variable names, and its
concrete timeout/retry numeric defaults — is #307's job, constrained to
satisfy every rule in this section.

## 13. Backward compatibility, scope boundary, and privacy

### 13.1 Wire contract

`HTTP_CONTRACT` and `MCP_CONTRACT` (`src/api.rs:99,108`) both stay `1`. No
existing endpoint, MCP tool, or SDK method changes behavior for a caller that
does not opt in — `retrieve` in particular is untouched by this ADR; #216's
own acceptance criterion ("existing retrieve/direct endpoints do not change
without opt-in") is satisfied by this feature living entirely behind a new,
separately-named surface rather than by any defensive check inside
`retrieve` itself.

### 13.2 Auth and credential/privacy boundary

The new endpoint enforces the identical context-scoped key check every
composed lane already enforces individually — it grants no broader access
than the caller already had to `resolve`/`query`/`activate`/
`sources/search`/`citations`/`communities/search` on the same context.
Masking of any secret-shaped value follows the same discipline the composed
lanes already apply; this feature introduces no new credential material of
its own to protect beyond the reranker provider key covered in §12.

### 13.3 `langchain-taguru` is explicitly out of scope for v0.6.0

`sdk/python-langchain/src/taguru_langchain/retrievers.py` and its TypeScript
twin reimplement their own graph+text RRF fusion directly against
`resolve`/`activate`/`cite_passage`/`search_passages` (§2.1) and are not
covered by `sdk/spec/surface.yaml`'s parity guarantee at all. Giving them
parity with this feature would mean either rewriting their fusion logic to
call the new endpoint (a behavior change to an already-shipped integration,
outside an ADR that promises no unintended behavior change) or maintaining a
second, independent evidence-assembly implementation inside the LangChain
package. Neither is filed as a v0.6.0 child issue; a future ADR can decide
whether `TaguruRetriever` should call `assemble_evidence` once this feature
has shipped and stabilized.

### 13.4 Cross-context/cross-group assembly is deferred, not forgotten

`POST /recall`, `POST /query`, and `POST /sources/search` all already accept
`contexts?`/`groups?` and merge results with a context tag on each hit
(`src/llm-protocol.md:308,310,325`). This feature's v0.6.0 shape is
single-context only (`POST /contexts/{name}/evidence`, no cross-context
sibling) for two reasons: first, §7's RRF fusion already has to reconcile
four lanes' internal ranks into one order — adding "and also reconcile
across N independently-searched contexts" at the same time multiplies the
determinism surface this ADR has to specify correctly before #303 can start;
second, §8's budget semantics are already non-trivial for one context's
worth of candidates, and a cross-context budget would need to additionally
decide whether budget is shared or per-context, a genuinely new design
question this ADR does not need to answer to unblock #303-#308. Adding a
cross-context sibling endpoint later is an ordinary additive, compatible
change under ADR 0005 §4 ("add a new HTTP endpoint" — compatible) whenever a
concrete need for it exists.

## 14. Evaluation method (for #308)

Reuse the `eval.jsonl` dataset and `taguru evaluate` harness ADR 0004 already
specifies rather than building a second evaluation path. Compare, at equal
budget (§8, same `max_bytes`/`max_tokens`/`max_items` across all three
configurations):

1. **Baseline**: today's fixed-limit `retrieve` (or the single-lane
   `sources/search`, whichever ADR 0004's harness already exercises) with no
   assembly step.
2. **Deterministic assembly**: this feature with no reranker configured
   (`plan.reranker.configured == false`).
3. **Configured reranker**: this feature with a reranker configured, where
   available — recorded, per §12, alongside the model identity actually used.

Metrics, computed per ADR 0004 §8's existing citation-recall semantics plus
one new metric this ADR introduces:

- Citation recall and citation-locator correctness (ADR 0004 §8, unchanged
  definition).
- Rank-based retrieval metrics already in ADR 0004's scope (recall@k, MRR/
  nDCG) computed over `items`.
- **Source diversity**: the count of distinct `source` values across a case's
  admitted `items`, at equal budget — the metric §9's tier-based round-robin
  admission exists to move, and the one #216 names explicitly ("source
  diversity at equal evidence budget") that ADR 0004 does not already define.
- Latency distribution, per ADR 0004's existing harness shape.
- `plan.reranker.ran` / degrade rate, to quantify how often a configured
  reranker's failure path (§12) actually triggers under the fixture corpus.

The default repository gate stays offline, deterministic, and provider-free
— configuration 3 above (configured reranker) is opt-in exactly as ADR 0004's
own embedding-provider suites already are, never required for the gate to
pass. Regression thresholds for citation recall and source diversity are
recorded in ADR 0004 §9.3's existing thresholds-file format, not a new
format.

## 15. Consequences and follow-up

| Child issue | What it implements from this ADR |
|---|---|
| #303 | §6 (candidate model), §7 (ranking/RRF) |
| #304 | §8 (budget), §9 (selection algorithm and its invariants) |
| #305 | §5 (HTTP/MCP surface), §10 (response shape), §11 (errors), §13.1-13.2 |
| #306 | §5.3 (SDK parity), §10 (typed mirrors in both SDKs) |
| #307 | §12 (reranker's concrete wire protocol, within the boundary fixed here) |
| #308 | §14 (evaluation method and equal-budget comparison) |

Not filed, and why: a cross-context/cross-group sibling endpoint (§13.4 —
no concrete need yet, and an ordinary additive change whenever one appears);
`langchain-taguru` parity (§13.3 — a behavior-changing decision for an
already-shipped integration, deliberately left to a future ADR); refining the
§8 token estimator beyond the fixed formula this ADR names (changing it is
itself a breaking change under this ADR's own §8 reasoning, so refinement
needs its own ADR, not a follow-up issue); adding `total` to
`PassagePage`/`CrossPassagePage`/`CommunityPage` (already recorded as
optional future work in ADR 0005 §9, unrelated to this feature beyond both
touching passage/community responses).

## 16. Documentation impact

No documentation ships with this ADR. `src/llm-protocol.md`'s `## API` table
gains the new endpoint's row, and a new subsection explains how to read
`plan.selection`/`plan.reranker` — both land in #305's implementation PR,
per ADR 0005 §2.6's own rule that the PR adding a route is the PR that edits
`llm-protocol.md`, not the ADR that designed it. `docs/` pages and the
CHANGELOG entry are likewise #305's (API surface) and #308's (evaluation
workflow) responsibility.

## Appendix: requirement traceability

| #302 completion criterion | Section |
|---|---|
| ADR defines candidate types, provenance, rank features, citation retention | §6 |
| ADR defines byte/estimated-token/hard-item-ceiling budget meaning and priority | §8 |
| ADR defines dedup, source diversity, corroboration, negative/contradictory evidence rules | §9 |
| ADR defines the deterministic-fallback/optional-reranker boundary | §12 |
| ADR fixes HTTP/MCP/SDK opt-in public API and selection-trace/error shape | §5, §10, §11 |

| #216 acceptance criterion | Section |
|---|---|
| ADR defines candidate types, scoring/ranking boundaries, budget semantics, dedup, contradiction handling, fallback behavior, and the public API | §5-§9, §12 |
| Assembly is opt-in and existing retrieval APIs remain backward compatible | §13.1 |
| A deterministic no-provider policy produces stable evidence packages | §9 (I3), §12 |
| An explicit byte/token estimate budget and hard item limit are enforced before returning the package | §8, §9 (I2) |
| Selected evidence retains context, source, paragraph/citation, lane, and graph provenance | §6 |
| Redundant evidence is removed without collapsing independent corroboration or contradictory claims | §9 (dedup, I4, I5) |
| Provider failure degrades to the deterministic policy and is visible in the execution plan and metrics | §11, §12 |
| Python, TypeScript, MCP, and HTTP surfaces have equivalent typed response semantics where applicable | §5.2, §5.3 |
| #215 demonstrates quality at equal budget and guards against citation-recall regression | §14 |
| Tests cover mixed graph/passage evidence, duplicate passages, multiple sources, negative/contradictory evidence, tiny budgets, provider failure, deterministic ordering | §9 (implementation guidance for #304's test plan) |
| Documentation makes clear this prepares evidence for an external answer model | §16, §1 (non-goal statement) |
| Repository and SDK quality gates remain green | N/A — enforced by #303-#308's own CI, not this ADR |
