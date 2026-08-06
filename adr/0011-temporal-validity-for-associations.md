# 0011. Temporal validity for associations — source-derived assertion time on the graph lanes

- **Status**: Accepted
- **Date**: 2026-08-07
- **Issue**: #420
- **Related**: #421, #423, ADR 0005, ADR 0009 §9.2 / §12, ADR 0010 §7
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How time-varying facts — "X is CEO of Y (until 2024)" — become expressible
and queryable on the association graph. Concretely: where an association's
temporal position lives, how `since`/`until` reach the graph lanes
(`activate`, `explore`, `recall`, `query`), what a time window does to edge
weights, and what all of it costs in wire contract, on-disk format, and
cache terms.

One term is fixed up front, because the whole design hangs on it: what this
ADR gives an association is an **assertion time** — the effective time of
the sources that attest it — not a validity *interval*. A window selects
"what had been asserted when," and an as-of query reads "the knowledge
available at time T." A finite end of validity ("until 2024") is **not
representable as stored data** in this model; §6 fixes how it is expressed
as evidence plus reader judgment, and §7 names the derivation path a future
ADR would take if first-class intervals are ever needed. The issue's word
"validity" survives in the title because assertion time is its load-bearing
first half — not because this document delivers intervals.

Design-first, the #370 / #225 shelf: this document fixes the representation
and the semantics before any implementation issue is cut. Out of scope,
answered in §9 rather than silently dropped: bi-temporal modeling, validity
intervals and end-time derivation, automatic supersession or forgetting,
and temporal *scoring*.

## 2. Context: where time already lives in the tree, and where it doesn't

Time exists on exactly one side of the store today:

- **Sources carry it.** `SourceMeta` (`src/passages.rs:101`) holds `date`
  (caller-declared document time) and `stored_at` (server-stamped ingestion
  time), both epoch seconds. The passage lane already filters on it:
  `SourceFilter::matches` (`src/passages.rs:130`) compares
  `date.or(stored_at)` — the *effective time* rule — against a half-open
  `[since, until)` window, and `eligible_sources` (`src/passages.rs:998`,
  issue #167) precomputes the set of qualifying source ids before the search
  lanes run. A source with neither field does not match a time-filtered
  query; that rule is documented and shipped.
- **The graph does not.** `EdgeRecord` (`src/context.rs:514`) is
  subject/label/object plus adjacency chains, an attribution chain, and the
  accumulated `count`/`sum` whose quotient is the published weight. Nothing
  in `src/context/` references `SourceMeta`, `since`, `until`, or any clock.
  `activate`/`explore`/`recall`/`query_any` (`src/context/traverse.rs`,
  `src/context/query.rs`) are time-blind: an association attested in 2019
  and one attested yesterday are indistinguishable to every graph read.

Three existing mechanisms determine the shape of the answer:

- **Attributions are the edge↔source join, and they carry per-source
  weights.** Each edge chains `AttributionRecord`s (`src/context.rs:551`) —
  one per distinct source, each holding that source's own `(count, sum)`.
  `retract_source` unlinks a source's attributions; compaction preserves
  "per-source weights within float re-accumulation error." An edge's weight
  restricted to a subset of its sources is therefore *computable exactly*,
  not approximable.
- **The edge's `source` and the passage store's source id are the same
  string by convention, not by reference.** `Context` interns the
  association's `source` as an opaque string; `SourceMeta` lives in the
  passage store keyed by the same name. Nothing in the library resolves one
  to the other — the join exists at the registry layer, where both stores
  are in hand.
- **"Read paths filter ineligible edges" is already a rule.** ADR 0010 §7
  fixed it for attribution-less ghosts: an edge no source attests anymore
  must not be served. A time window generalizes the same predicate — an
  edge no *in-window* source attests is invisible to that query.

And one deferred design is waiting on exactly this: ADR 0009 §9.2 defers
`cardinality` (functional relations, "one object per subject") because
"one object *ever*" is unanswerable under retract-then-apply and under
facts that legitimately change. §7 below records how valid time reframes
that question without implementing it.

## 3. Options considered — where valid time lives

**A. Edge-carried validity: `valid_from`/`valid_to` on the association.**
The textbook shape, rejected. It touches every layer at once — `AssocOp`,
`WalOp::Associate`, the batch line, `IMAGE_VERSION` (the 48-byte
`EdgeRecord` grows), export/import, replication — a coordinated format bump
across five surfaces for fields most writers cannot fill (an extractor
rarely knows a fact's end date at assertion time; end dates are usually
*discovered* when the superseding fact arrives). Worse, it creates a second
authority for "when was this true" beside the source dates that already
exist — the same dual-authority class of bug ADR 0009 §9.5 refused for
relation canonicalization: two answers, resolvable in different orders on
different paths, undiscoverable until a query returns the wrong facts.

**B. Source-derived assertion time: an edge's temporal position is the
effective time of the sources that attest it.** Chosen; §4. No write-format
change of
any kind: `date` is already declared per source at ingest, attributions
already join edges to sources, and per-source weights make windowed
recomputation exact. The claim this makes is honest about provenance: a
fact is temporally located *because some dated source asserted it* — which
is also the only evidence the store actually has.

**C. Bi-temporal (valid time × transaction time as independent axes).**
Rejected by the issue itself as a non-goal, and `date`/`stored_at` already
give the useful degenerate form: `date` approximates valid time,
`stored_at` is transaction time, and the `date.or(stored_at)` effective
rule prefers the former when a caller bothered to state it. Nothing in
option B forecloses a future bi-temporal refinement; a new ADR would own
that.

## 4. Decision

**An association's temporal position is its attributing sources' assertion
time — the effective time `date.or(stored_at)`, the `SourceFilter` rule
verbatim. There is no stored end time. The graph lanes gain optional
`since`/`until` request parameters; a windowed query sees an edge iff at
least one in-window source attests it, and sees that edge's weight
recomputed from its in-window attributions only. Nothing is written,
nothing is invalidated, no format moves.**

Mechanically, the passage lane's own #167 shape is reused one lane over:

1. The registry computes the eligible source set for the window —
   `eligible_sources` with a `SourceFilter`, exactly as `search_passages`
   does today. This happens where it must: the registry is the one layer
   that holds both the passage store (metadata) and the context (graph).
2. The library's graph reads accept an optional eligibility view — the
   eligible source names, resolved to interned `SourceId`s at the boundary.
   `Context` stays metadata-agnostic: it receives "these sources qualify,"
   never a date. The library never learns what `SourceMeta` is.
3. Edge eligibility inside a windowed read: the attribution chain contains
   at least one record whose source qualifies (and whose `count > 0` —
   the ADR 0010 §7 ghost rule composes, it is the same walk).
4. Edge weight inside a windowed read: `Σ sum / Σ count` over qualifying
   attributions only. `activate`'s conductance (`sum.abs()` at
   `src/context/traverse.rs:348`) uses the windowed sum, so activation
   cannot flow through an edge on the strength of out-of-window evidence —
   filtering eligibility but conducting full weight would quietly lie at
   exactly the margins a temporal query exists to probe.

An unwindowed query (`since`/`until` absent) takes none of these branches
and behaves byte-for-byte as today.

**Sources with no effective time are invisible to windowed queries.** This
is `SourceFilter`'s shipped rule, adopted unchanged rather than
re-litigated: one rule for "does this source exist at time T," whichever
lane asks. The graph-specific consequence is named honestly: an
associations-only source (a batch that never stored a passage) has no
`SourceMeta` at all today, so its edges vanish from every windowed query.
For corpora that care, the zero-format-change remedy already exists —
declare `date` and store even a one-line passage — and a follow-up may
later give associations-only sources a metadata record; that would be a
new decision, not a silent extension of this one.

## 5. Query semantics: what a window means

`[since, until)` half-open, epoch seconds, both optional — the passage
lane's contract, verbatim. Per lane:

- **`query` / `recall`**: the association list (and recall's fact lines)
  contains only in-window edges, weights windowed. The natural reading of
  `until: T` alone is **as-of**: "the graph as a reader in year T could
  have known it." `since` alone reads "only what has been asserted since."
- **`activate` / `explore`**: eligibility applies to *every hop*, not just
  the frontier — activation must not conduct through an edge that the
  window excludes, or the walk reaches nodes the windowed graph does not
  connect. This falls out of step 3/4 applying uniformly inside the
  traversal, and it is why the filter lives inside the walk rather than as
  a post-filter on results.
- **`resolve` / `describe`**: unchanged. ADR 0009 §12 already drew this
  line for the type filter: `resolve` is entry-point lookup by spelling,
  and a filter changes what it means. A windowed exploration starts from a
  resolved entry point, then explores with the window.

## 6. Supersession and the signed weight: what a window does *not* do

"X is CEO (until 2024)" is representable as *evidence*, not as an edict:
the 2019-dated source attests `X --CEO--> Y`; the 2024-dated source attests
`Z --CEO--> Y`. An as-of-2023 query sees X alone. An as-of-2025 query — and
an unwindowed one — sees **both**, because assertion times order the
evidence but end nothing: the window answers "what had been asserted by
then," never "which assertion still holds." That is the model's honest
limit and its deliberate posture at once — the store forbids silent
rewrites (every consolidation is proposal-only, ADR issue #421), so the
*current-truth* judgment belongs to the reader, not to a write-path side
effect. A reader who wants "the CEO as of 2025" reads both dated facts and
judges — or asks #421's audit, which §7 gives the ordered input it needs.
Three consequences, fixed here:

- **No auto-invalidation.** A new fact for the same `(subject, label)`
  invalidates nothing. Detection of the resulting contradiction is #421's
  audit (which this design simplifies: "two in-window sources attest
  conflicting objects for a functional label" is now a well-posed query);
  resolution is an explicit client write, as ever.
- **The signed weight is orthogonal and composes.** A negative-weight
  assertion is itself dated evidence; inside a window it subtracts from the
  windowed sum exactly as it does globally. "Contested in 2024" and
  "uncontested as of 2023" are both expressible with the existing
  machinery, which is the strongest argument that weights and valid time
  are different axes and must not be conflated (e.g., no time-based weight
  decay — §9).
- **Explicit retraction stays global.** `retract_association` /
  `retract_source` erase evidence from every window, past included — they
  mean "should never have been asserted," not "stopped being true." The
  distinction already exists in the library's own doc comments; valid time
  finally gives "stopped being true" a home that is not misuse of
  retraction.

## 7. The cardinality deferral, revisited but not resolved

ADR 0009 §9.2 deferred functional relations because "one object ever" is
the wrong invariant under change: a batch's retract-then-apply window
violates it transiently, and reality violates it permanently (companies
change CEOs). The invariant that survives change is **"at most one object
at a time"** — but stating it takes intervals, and this ADR stores none
(§1). What assertion times *do* make well-posed, with no write-path read
at all, is the audit-side question one step short of it: for a functional
label, **temporally ordered assertions of different objects** — X asserted
2019, Z asserted 2024 — are a supersession-or-conflict candidate, ordered
and dated. Whether the earlier fact *ended* (derive: X's validity closes
at Z's assertion time) or the sources genuinely conflict is a judgment;
the derivation rule that would close intervals from successor assertions
is **inference**, so by ADR 0009 §9.2's own line it belongs to the
cardinality follow-up ADR — proposal-side, in #421's audit, never a
storage fact. This ADR deliberately designs neither the check nor the
derivation. What is fixed here is only that the temporal representation
(source-derived, per-attribution) is sufficient *input* for both — no
edge-carried validity is needed to get there, and nothing here forecloses
either.

## 8. Surface changes and compatibility

Everything below is additive under ADR 0005 §4; `http_contract` stays 1.

- **HTTP**: optional `since`/`until` (epoch seconds) on the graph-lane
  request bodies — recall, query, explore, activate. "Add an optional
  request field: compatible" (§4 table; no request body denies unknown
  fields). Response shapes unchanged — windowed weights flow through the
  existing `weight` field, which is a *semantics-under-parameters* change
  the caller opted into by sending the window, not a silent meaning change
  of an unparameterized response.
- **The window's input contract is the passage lane's, by shared code, not
  by convention.** `source_filter` (`src/api/sources.rs:395`) already owns
  it — epoch-second `u64`s, either bound optional, and `since >= until`
  refused up front as `InvalidArgument` ("the window is half-open
  `[since, until)` and this one selects nothing") — and its doc states the
  reuse rule this ADR extends: shared by every entrance "so the surfaces
  cannot drift on what a legal filter is." The four graph lanes call the
  same helper; a deliberately empty window is therefore unrepresentable
  on every lane rather than a silently empty result on some.
- **MCP**: the corresponding tools gain the same two optional
  `inputSchema` properties — "add an optional property: compatible." No
  `mcp_contract` bump; the shapes are HTTP's (ADR 0005 §3 boundary).
- **SDKs**: optional keyword arguments; surface.yaml addition. No decoding
  change.
- **Library (`taguru` crate)**: the graph reads need the eligibility view
  (§4 step 2). New methods or an options struct beside the existing
  signatures — additive, pre-1.0 minor per the crate's existing
  discipline; the "lib surface stays minimal" policy is respected by
  passing an opaque eligible-source view, not a metadata type.
- **On-disk / batch / WAL / replication**: untouched. `IMAGE_VERSION`
  stays 6, `taguru_batch` stays 1, `WalOp::Associate` unchanged. This is
  the payoff of option B and the reason it wins ADR 0005-cheap.
- **Cache**: the window enters `RetrievalKey.params`
  (`src/registry/retrieval_cache.rs:70`) — distinct windows are distinct
  entries, as for any other request parameter. The lane pairing already
  happens to be correct: `Recall | Query → [graph, passages]`
  (`op_lanes`, `:216`), and a windowed result now genuinely depends on the
  passage store (source metadata lives there), so the `passages` lane
  invalidating on metadata change is load-bearing, not incidental. A
  windowed `explore`/`activate` is cursor-paged and uncached today;
  nothing new to invalidate.
- **Authorization**: no new routes; existing read-role classification
  carries (ADR 0009 §12.5 pattern, nothing to add).

Cost note, honest: the eligible-set computation is O(sources) per windowed
request (`eligible_sources`' existing cost), and eligibility inside the
walk is O(attribution chain) per touched edge. Unwindowed requests pay
zero. If a windowed hot path ever matters, a per-request memo of
edge→eligibility is a contained optimization inside the walk; nothing in
the contract above constrains it.

## 9. Non-goals

- **Bi-temporal modeling** — §3 option C; a future ADR if ever.
- **Validity intervals and end-time derivation** — §1's term fix made
  binding: no stored end time, and the "close the earlier interval at the
  successor's assertion time" rule is inference that belongs to the
  cardinality follow-up ADR (§7), never a quiet extension of this one.
- **Automatic supersession, expiry, or forgetting** — §6; conflicts with
  the store's explicit-writes-only posture, and #423's umbrella owns the
  lifecycle question.
- **Temporal scoring** (recency boosts, time-decayed weights) — the same
  line ADR 0009 §12.4 drew for type-aware scoring: a score contribution
  that is not explainable is out, and decay would conflate the weight axis
  with the time axis §6 keeps apart.
- **Edge-carried `valid_from`/`valid_to`** — rejected in §3A, recorded as
  a non-goal so a future need is forced through a superseding ADR rather
  than a quiet field addition.
- **Server-side inference of end dates** — "the 2024 fact ends the 2019
  fact" is a judgment; the server serves both, dated (README boundary:
  language understanding is the client's job).

## 10. Consequences and follow-ups

- Time-filtered graph queries become possible with no migration: existing
  corpora that declared `date` are queryable as-of immediately; corpora
  that didn't can adopt it source by source.
- The quality of temporal answers is exactly the quality of source
  dating — garbage dates, garbage windows. `evalset`-style measurement of
  windowed recall on a dated corpus is the natural acceptance gate for the
  implementation issue.
- #421's staleness/contradiction audit gains a well-posed input — dated,
  ordered assertions per `(subject, label)` (§6, §7); the cardinality
  follow-up ADR gains its target invariant and the interval-derivation
  question, both named but neither designed here (§7).
- Implementation splits cleanly: (S1) library — eligibility view on the
  four graph reads, windowed weight; (S2) registry — window→eligible-set
  plumbing, cache params; (S3) HTTP/MCP/SDK surface; (S4) docs +
  `llm-protocol.md` producer guidance ("declare `date` when the document
  has one" is already the ingest discipline; it now pays on the graph
  lanes too). Each stage lands green independently; S1–S2 ship inert
  until S3 exposes the parameters.
