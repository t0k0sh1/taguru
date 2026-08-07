# 0008. Retrieval tracing: span inventory, attribute vocabulary, privacy, and propagation

- **Status**: Accepted
- **Date**: 2026-08-01
- **Issue**: #224
- **Related**: #132, #151, #193, #194, #220, ADR 0005 §4, ADR 0006 §10, ADR 0007 §11
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

Unlike ADRs 0006/0007, this one ships with its own code — #224 is one issue,
not an umbrella — so this document and its implementation land in the same
PR.

## 1. Scope

The span inventory, attribute vocabulary bound to existing metric enums,
event vocabulary, privacy enforcement, error-status rules, and context
propagation (HTTP inbound/outbound, `std::thread::scope`,
`block_in_place`/`Handle::block_on`, router fan-out, stdio MCP) for tracing
Taguru's composed retrieval pipeline end to end, per #224. Also records two
pre-existing defects this work must fix because they make #224's own
acceptance criteria unachievable (§2.5).

Out of scope: the Python/TypeScript SDK's own span emission (a separate,
smaller vocabulary that reuses this document's span/attribute names but ships
no server-side code — recorded here only as a consumer of §5/§6); metrics
(Prometheus counters are unchanged; §6 only reads their label vocabularies,
never forks them); log format changes beyond the existing `trace_id`
correlation (`src/metrics.rs:2125`).

## 2. Context

### 2.1 What exists today

`src/trace.rs` is an opt-in OTLP span pipeline: `enabled() -> bool`,
`provider() -> (Option<SdkTracerProvider>, Option<String>)`, and
`extract_parent(&HeaderMap) -> Context` (W3C `traceparent` first, AWS
`x-amzn-trace-id` fallback). `init_telemetry` (`src/main.rs:1265-1315`) wires
`tracing_opentelemetry::layer()` at a bare `LevelFilter::INFO`, independent of
`RUST_LOG`. Across the entire production codebase there are exactly **three**
spans: `"request"` (`src/metrics.rs:2167-2209`, the HTTP server span, parented
via `extract_parent`), `"embed"` (`src/embedding.rs:178-190`), and `"rerank"`
(`src/api/evidence/rerank.rs:277-283`). **No inject side exists anywhere** —
`TextMapPropagator`/`set_text_map_propagator` appear nowhere in the tree — so
every trace stops dead at a process boundary. Router mode
(`route::track_router_http`, `src/route.rs:619-642`) calls `init_telemetry`
but creates no span at all. The stdio bridge (`src/bin/taguru-mcp.rs`) has no
tracing subscriber whatsoever.

### 2.2 The closed metric vocabularies this must not fork

`src/metrics.rs` already has several closed, exhaustively-matched enums with
an `as_str()` and an `ALL` constant that renders every label from zero:
`SearchOp` (`resolve`/`resolve_label`/`recall`/`query`/`activate`/
`search_passages`/`search_communities`/`explore`), `RetrievalCacheOp`
(`recall`/`query`/`search_passages`/`search_communities`),
`SemanticCacheOutcome` (`hit`/`stale`/`guarded`/`miss`), `ResolveTier`
(`lexical`/`semantic`/`weak_lexical`/`miss`), `RerankOutcomeKind` (`ok` plus
seven failure tokens matched verbatim from `src/api/evidence/rerank.rs`'s
`REASON_*` constants). `src/registry.rs` separately has `PassageSearchLanes`
(`NoQueryTerms`/`ZeroLimit`/`Ran{vector}`) and `VectorLaneStatus`
(`Off{provider_configured}`/`QueryEmbeddingFailed(String)`/`NoVectors`/
`ModelChanged{stored,current}`/`WidthChanged{stored,current}`/`Ran{floor}`),
used for wire diagnostics but with no `as_str()`/`code()` of their own yet.
Span attributes must reuse these strings, not invent a parallel spelling
(§6).

### 2.3 The two composed-retrieval implementations

**Client-composed**: `mcp::run_retrieve_bounded` (`src/mcp/retrieve.rs:63`),
called from `src/remote_mcp.rs:127` (remote MCP, inside
`tokio::task::block_in_place` + `Handle::block_on`) and from
`src/bin/taguru-mcp.rs:499` (stdio bridge, via the unbounded `run_retrieve`
wrapper). It performs Step 1 resolve origins (`retrieve.rs:131-163`), Step 2
describe anchors (`:165-176`, skippable), Step 3a query associations
(`:186-207`, labels-gated) / 3b activate graph (`:208-235`), Step 4 fetch
citations (`:243-280`, skippable, tolerates per-citation 404), Step 5 passage
fallback (`:293-319`, conditional on `text_fallback_only_if_empty`).

**Server-composed**: `assemble_evidence` (`src/api/evidence/assemble.rs:154`,
`POST /contexts/{name}/evidence`), whose `EvidenceLanesPlan`
(`assemble.rs:130-138`) already names its phases `resolve`/`query`/
`activate`/`passages`/`communities`/`citations` — a vocabulary this ADR
reuses rather than renaming (§5).

### 2.4 Existing privacy precedent

`RerankCandidate` "deliberately does not derive `Debug`"
(`src/api/evidence/rerank.rs:299`) so a stray `{:?}` log can never leak
passage text — the same posture this ADR applies to spans (§8), just enforced
at a different layer (no dynamic attribute keys, rather than no `Debug`).

### 2.5 Two live defects that block #224's own acceptance criteria

Found while auditing the code this ADR must build on top of. Both are fixed
in the same PR as this ADR (§9, §12).

**(a) An embedding degrade colors the whole request span ERROR.**
`src/registry/search.rs:91-95`:

```rust
tracing::warn!(
    context = %name,
    error,
    "passage query embedding failed; serving the lexical lane alone"
);
```

`tracing-opentelemetry` maps a field literally named `error` to an exception
event, and `error_events_to_status` defaults to `true`. The event is WARN,
which passes the otel layer's `LevelFilter::INFO` (`src/main.rs:1285`).
Result: #224's "provider degrade 後に成功した retrieval を誤って root
ERROR にしない" fails today, on every semantic-lane degrade.

**(b) `TAGURU_LOG_SEARCHES` plus OTLP exports the raw user question.**
`src/api/sources.rs:707` (and the cached-path twins at `:737`, `:795`,
`:1358`, `:1387`, `:1487`) log `cue = %request.query` at INFO on target
`taguru::search`. An INFO event inside an active span becomes an OTel span
event. Result: #224's privacy criterion — question text never reaches a
default span — fails whenever both `TAGURU_LOG_SEARCHES=1` and OTLP export
are enabled together.

## 3. Options considered

**Span namespace**: (i) `taguru.*` on every span including the HTTP server
span, (ii) `taguru.*` on Taguru-owned internal/client spans only, server span
keeps semconv `{method} {route}`, (iii) no namespace, following the existing
`"request"`/`"embed"`/`"rerank"` precedent. Rejected (i): `otel.name` already
overrides the HTTP span's macro name to `{method} {route}` and a wire test
(`tests/http_api/observability.rs:161`) asserts that literal string;
renaming would break a passing test for zero benefit, since semconv naming is
what every OTel-aware backend already expects for HTTP spans. Rejected (iii):
`embed`/`rerank` are generic nouns that collide with any other instrumented
library in a shared collector, and leaving 3 spans unnamespaced while adding
~15 namespaced ones is worse than a one-time rename of those 3. **Decision:
(ii).**

**Attribute vocabulary**: (i) a fresh `taguru.*` attribute schema, decided
independently of `src/metrics.rs`, (ii) attributes generated from the
existing metric enums' `as_str()`. Rejected (i): would create two vocabularies
for the same underlying state (e.g. resolve tier, cache outcome), free to
drift, doubling the maintenance surface for every future variant. **Decision:
(ii)**, extending the two enums that don't yet have a string form
(`PassageSearchLanes`, `VectorLaneStatus`) with a `code()` method following
the existing `ALL`+`as_str` pattern, rather than inventing new enums.

**Propagator registration**: (i) `opentelemetry::global::set_text_map_propagator`,
(ii) a process-local `OnceLock<TraceContextPropagator>` used directly by
`inject_context`/`extract_parent`. Rejected (i): nothing in this tree reads
the OTel global (no third-party middleware is installed), the global costs an
`RwLock` read per inject, and registering it would create two independent
extraction paths — this file's hand-rolled `parse_traceparent` and the SDK's
own global-propagator extraction — free to disagree about a malformed header.
**Decision: (ii)**, keeping both directions in `src/trace.rs` so they cannot
drift apart.

**Test harness**: (i) enable `opentelemetry_sdk`'s `testing` feature
(`InMemorySpanExporter`), (ii) extend the existing `FakeCollector`
(`tests/http_api/observability.rs:49-128`), a hand-rolled OTLP-over-HTTP
collector. Rejected (i): `tests/http_api/*` drives a **spawned binary**
(`Server::spawn`, `support.rs:38`); an in-process exporter cannot see spans
from a separate process, so it does not solve the actual testing problem.
The feature also pulls in `metrics`/`logs`/`tokio/sync` unconditionally
(`opentelemetry_sdk`'s Cargo.toml), which Cargo's dev/normal feature
unification would add to the shipped binary for a facility that cannot work
here anyway. **Decision: (ii)**; `FakeCollector` already exposes the complete
OTLP wire shape (parent ids, attributes, events, status, resource), strictly
more than `InMemorySpanExporter` proves.

**`taguru.shard_call` shape**: (i) span link, (ii) child span. Rejected (i):
links exist for joining otherwise-unrelated traces (e.g. batch consumers); a
shard call is a causal child of one router request with a fully contained
lifetime, and a link would lose that containment, cost a link record per
call, and hide the router's own elapsed time from the trace. **Decision:
(ii)**.

## 4. Decision

1. Every Taguru-owned span is `taguru.`-prefixed (§5); the HTTP server span
   alone keeps semconv's `{method} {route}`.
2. Span attributes reuse existing metric enums' string form wherever one
   exists; the two that don't (`PassageSearchLanes`, `VectorLaneStatus`) gain
   a `code()` method rather than a parallel vocabulary (§6).
3. No dynamic attribute keys exist anywhere in the design; the complete
   exported key set is the one enumerated in §6. A target-level firewall
   plus `.with_error_events_to_status(false)` on the export layer, and a
   sentinel test, jointly enforce that no question/concept/source/passage
   text ever reaches a span (§8).
4. `otel.status_code = ERROR` is set only by the span whose own operation
   failed to produce its result, never propagated upward (§9).
5. Context propagation is bidirectional and explicit: `trace::inject_context`/
   `inject_current` (new) complete the half `extract_parent` already
   provides, applied at every process/thread boundary the pipeline crosses —
   HTTP outbound (router fan-out, stdio bridge), `std::thread::scope`
   (passage-search lanes), and `block_in_place`/`Handle::block_on` (§10).
2. through 5. are implemented exactly as specified in §5-§10 below; this
   section is the summary, those sections are binding.

## 5. Span inventory

| span | `otel.kind` | created in |
|---|---|---|
| `{method} {route}` | server | `src/trace.rs` (moved from `src/metrics.rs:2167`); called by `metrics::track_http` **and** `route::track_router_http` |
| `taguru.retrieve` | internal | `src/mcp/retrieve.rs::run_retrieve_bounded` |
| `taguru.resolve` / `taguru.describe` / `taguru.query` / `taguru.activate` / `taguru.citations` / `taguru.passage_fallback` | internal | same, one per Step 1–5 |
| `taguru.assemble_evidence` + the same six phase names | internal | `src/api/evidence/assemble.rs` |
| `taguru.passage_search` | internal | `src/api/sources.rs::search_passages` |
| `taguru.search.bm25` / `taguru.search.ann` / `taguru.search.fuse` | internal | `src/registry/search.rs`, `src/embedding.rs::top_matches` |
| `taguru.embed` / `taguru.rerank` | client | renamed from `"embed"`/`"rerank"` |
| `taguru.shard_call` | client | `src/route.rs::call_shard` |
| `taguru.tool_call` | server | `src/bin/taguru-mcp.rs::run_tool_worker` |

**No `taguru.search.embed`.** The query embedding already runs inside
`taguru.embed`, reached through `passage_query_cue` (`src/registry/search.rs:643`)
→ `cue_vector` (`src/registry.rs:2150`). A wrapper span would add exactly one
bit of information — process cue-cache hit vs. miss — which belongs on an
event (`taguru.cache`, §7), not a span. Note for span-tree readers:
`search_passages` calls `cue_vector` **twice** (the semantic-cache probe at
`src/api/sources.rs:727`, then the lane's own call at `src/registry/search.rs:643`);
the second is a process-cache hit, so one search shows exactly one
`taguru.embed` child, not two.

## 6. Attribute registry

Where a Prometheus label vocabulary already exists, the span attribute
reuses the same `as_str()`, made `pub(crate)`:

| span attribute | source of truth | values |
|---|---|---|
| `taguru.op` | `SearchOp::as_str()` (`src/metrics.rs`) | `resolve` `resolve_label` `recall` `query` `activate` `search_passages` `search_communities` `explore` |
| `taguru.cache.result` | `RetrievalCacheOp` hit/miss | `hit` `miss` |
| `taguru.cache.semantic` | `SemanticCacheOutcome::as_str()` | `hit` `stale` `guarded` `miss` |
| `taguru.resolve.tier` | `ResolveTier::as_str()` (already `pub`) | `lexical` `semantic` `weak_lexical` `miss` |
| `taguru.search.lanes` | new `PassageSearchLanes::code()` | `no_query_terms` `zero_limit` `ran` |
| `taguru.search.vector.outcome` | new `VectorLaneStatus::code()` | `off` `query_embedding_failed` `no_vectors` `model_changed` `width_changed` `ran` |
| `taguru.passage.bm25_only` / `.both_lanes` / `.vector_only` | `taguru_passage_lane_contributions_total{lane}` | counts (`i64`) |
| `taguru.rerank.outcome` | `RerankOutcomeKind` / `REASON_*` | `ok` `not_configured` `model_mismatch` `empty_pool` `invalid_permutation` `circuit_open` `timeout` `provider_error` |
| `taguru.embed.model` | operator's `TAGURU_EMBED_MODEL` | the configured model name — operator-bounded config like §8's shard-list rule, never user data (shipping on `taguru.embed` since the span was introduced; registered here with #474's second provider) |
| `taguru.embed.purpose` | `EmbedPurpose::as_str()` | `index` `query` |

The two new `code()` methods follow the file's own `ALL` + exhaustive-`match`
convention: adding a variant without updating `code()` fails to compile.

Count/shape/flag attributes, all `i64`/`bool`/`f64` (a bare `u16` records as
text — `src/metrics.rs:2199` already documents this trap): `taguru.operation`,
`taguru.transport`, `taguru.tool`, `taguru.context.count`,
`taguru.origin.count`, `taguru.anchor.count`, `taguru.association.count`,
`taguru.activation.count`, `taguru.citation.requested` / `.returned` /
`.missing`, `taguru.passage.hit_count`, `taguru.fallback.ran`,
`taguru.fallback.reason`, `taguru.search.terms` / `.pool` / `.hits` /
`.floor` / `.exact` / `.rows`, `taguru.filter.eligible` / `.total`,
`taguru.limit`, `taguru.dispatch.bytes`, `taguru.result.bytes`,
`taguru.embed.inputs`, `taguru.shard.index` / `.outcome`, `taguru.error.kind`.

Semconv wins where it already exists — `http.request.method`, `http.route`,
`http.response.status_code` — and is never duplicated under `taguru.*`.
Deliberately not `url.path`: unlike `http.route`'s templated shape, the
literal request path can carry a caller-chosen context name
(`/contexts/{name}/...`), which §8 promises never lands on any span.

`taguru.result.bytes` — the composed retrieval's true byte length, only
known after the transport serializes and caps it, well after
`taguru.retrieve` has already closed — is recorded on the ambient
request span instead: the `POST /mcp` HTTP span for the remote MCP
transport, `taguru.tool_call` for the stdio bridge.

## 7. Event vocabulary

`tracing-opentelemetry` maps a `tracing` event's `message` field to the OTel
event **name**; every other field becomes an event attribute. Three event
names, one stable-code attribute:

| event name | meaning |
|---|---|
| `taguru.skip` | a planned step did not run |
| `taguru.degrade` | a step ran in reduced form |
| `taguru.cache` | a cache decided the outcome |

carrying `taguru.reason` from this closed vocabulary: `describe_disabled`,
`no_anchors`, `labels_absent`, `citations_disabled`, `citation_passage_missing`,
`fallback_not_requested`, `fallback_suppressed`, `budget_exhausted`,
`deadline_exceeded_before_start`, `retrieval_cache_hit`,
`semantic_cache_hit`/`_stale`/`_guarded`/`_miss`, `cue_cache_hit`,
`zero_limit`, `no_query_terms`, `vector_off`, `vector_query_embedding_failed`,
`vector_no_vectors`, `vector_model_changed`, `vector_width_changed`,
`bridge_unreachable`.

**No span-event field is ever named `error`.** `tracing-opentelemetry`
special-cases exactly that name into an exception event and (by default) an
ERROR status; §9's `.with_error_events_to_status(false)` makes an accidental
violation harmless, but the naming rule itself is binding.

## 8. Privacy and cardinality

Forbidden by default, verbatim from #224: user question / fallback query
text, concept names, relation labels, source IDs, file paths, passage/citation
text, credentials, embedding vectors. Enforced by three independent
mechanisms:

1. **No dynamic attribute keys.** `tracing::info_span!`/`tracing::info!`
   require field names at macro-expansion time; nothing in this design calls
   a `set_attribute(key, value)`-shaped API. The exported key set is exactly
   §6's table — `grep -rn 'taguru\.' src/` is exhaustive.
2. **A target-level firewall on the export layer.** `init_telemetry`
   (`src/main.rs:1279-1286`) changes its otel-layer filter from bare
   `LevelFilter::INFO` to a `Targets` filter that turns `taguru::search`
   **off**, and sets `.with_error_events_to_status(false)`:
   ```rust
   tracing_opentelemetry::layer()
       .with_tracer(provider.tracer(env!("CARGO_PKG_NAME")))
       .with_error_events_to_status(false)
       .with_filter(
           tracing_subscriber::filter::Targets::new()
               .with_default(tracing::Level::INFO)
               .with_target("taguru::search", tracing::level_filters::LevelFilter::OFF),
       )
   ```
   This closes defect (b) structurally (the log target that carries the raw
   cue never reaches the export layer) and defect (a)'s whole class (no log
   field, `error`-named or otherwise, can set span status behind the code's
   back).
3. **A sentinel test**
   (`tests/http_api/tracing_pipeline.rs::no_question_concept_source_or_passage_text_reaches_the_collector`)
   seeds a context whose concept name, source id, passage text, and query
   each contain a unique nonce, runs a full retrieve with
   `TAGURU_LOG_SEARCHES=1`, and asserts none of the nonces appear anywhere in
   the raw OTLP request bodies `FakeCollector` stored — a substring check
   over the whole payload (attributes, events, resource alike).

Per-item rule: **aggregate by default; per-item child spans only where the
item identity is already non-sensitive *and* operator-bounded.** The only
qualifying case is the shard list (`src/route.rs:733`, an operator's own
config). Cues, contexts, citations, associations, and passages get counts
only. Context **names** are never recorded on any span — consistent with
`taguru_searches_total{op,outcome}` already carrying no context label.

## 9. Error semantics

1. `otel.status_code = ERROR` is set only by the span whose own operation
   failed to produce its result — never propagated to a parent. `UNSET`
   means success in OTel, so a degraded-but-completed retrieval is left
   alone. This is what makes "provider degrade 後に成功した retrieval を誤って
   root ERROR にしない" a property of the code, not a hope.
2. `.with_error_events_to_status(false)` (§8.2) removes the only path by
   which a log line could set a span's status behind the code's back.
3. The HTTP server span keeps its existing rule unchanged
   (`src/metrics.rs:2202-2204`): 5xx is ERROR, 4xx is not. A JSON-RPC tool
   error rides a 200, so `POST /mcp` stays `UNSET` while `taguru.retrieve`
   goes `ERROR` — the discrimination #224 asks for.
4. `taguru.error.kind` is a closed vocabulary: `deadline_exceeded`,
   `result_too_large`, `cancelled`, `invalid_argument`, `not_found`,
   `unauthorized`, `upstream_error`, `transport`, `provider_error`. Its
   classifier, `pub(crate) fn error_kind(message: &str) -> &'static str` in
   `src/mcp/retrieve.rs`, string-matches the two transports' own error text —
   acceptable here specifically because `src/remote_mcp.rs:183-186`'s doc
   already makes byte-identical error text across both transports a
   maintained invariant. A unit test feeds every named error constant
   through and asserts its mapped code, so a reworded constant fails the
   build's tests rather than silently collapsing to `provider_error`.

## 10. Context propagation

- **HTTP inbound**: unchanged, `trace::extract_parent` (existing).
- **HTTP outbound**: new `trace::inject_context`/`inject_current`, built on
  `opentelemetry_sdk::propagation::TraceContextPropagator` (already available
  under the enabled `trace` feature — no new dependency) and a
  `HeaderInjector` wrapping `http::HeaderMap` — the one type axum, reqwest,
  and ureq 3 share, so a single `Injector` implementation serves the
  router's fan-out and the stdio bridge's outbound calls alike. Applied at:
  `route::forward_headers` (from the router's own current span, replacing a
  bare header pass-through), `taguru-mcp.rs`'s `Bridge::call` (from the
  active tool-call/phase span).
- **`std::thread::scope`** (`src/registry/search.rs:202-227`, the BM25/
  semantic lane split): the calling thread's `tracing::Span::current()` is
  captured before `scope.spawn` and re-entered inside each lane via
  `parent.in_scope(...)`. A `tracing::Span` is handed across, not a bare
  `opentelemetry::Context` — `tracing-opentelemetry` derives the OTel parent
  from the *tracing* parent, so handing the `Span` keeps both the trace tree
  and log correlation intact; a bare `Context` would need a `set_parent` on
  every child and leave the tracing tree flat.
- **`block_in_place`/`Handle::block_on`** (`src/remote_mcp.rs:125-141` and
  the passage-search/evidence-assembly `block_in_place` sites): confirmed
  that the parent is not actually lost today, because `block_in_place` runs
  its closure on the same thread, inside the same `.instrument`ed poll, so
  `tracing`'s thread-local span stack is intact. `call_inner`'s future
  nonetheless gets an explicit `.instrument(tracing::Span::current())` at
  construction, so a future refactor that adds a `spawn` cannot silently
  detach it.
- **stdio MCP**: `params._meta.traceparent`/`.tracestate`, read via
  `mcp::protocol::meta_trace_headers` (new) which builds an `http::HeaderMap`
  and feeds `extract_parent` unchanged — one parser, shared by both
  transports. Optional, read-only; absent/malformed is "no parent," never an
  error.

## 11. Sampling and overhead

No custom sampler is introduced; the SDK's own `OTEL_TRACES_SAMPLER`/
`OTEL_TRACES_SAMPLER_ARG` env defaults (`parentbased_always_on`) are
respected unchanged. `Span::none()` (the `span!` macro's disabled-export arm,
§12) is the only genuinely zero-cost mode. **Sampled-out is not disabled**:
`tracing-opentelemetry` still creates the tracing span and the SDK still runs
the sampler even for a span that will not be exported; this is a real,
documented boundary, not an oversight. `taguru-mcp`'s stderr log level
defaults to `warn` (not `info`) specifically to avoid a per-tool-call log
line while the OTLP layer keeps exporting regardless.

## 12. Compatibility

Span names, attribute keys, and event codes are **not** part of ADR 0005's
wire contract — that ADR governs the HTTP/JSON-RPC wire shape, and spans are
telemetry, not response data. Renaming `"embed"`→`taguru.embed` and
`"rerank"`→`taguru.rerank` is therefore not a breaking wire change; no test
asserts either old name, and this is pre-1.0 telemetry rather than a
maintained contract. An operator with a dashboard keyed on the old span name
re-keys once.

`Transport` (new, `src/mcp/retrieve.rs`) is a parameter of
`run_retrieve_bounded`, which is not part of `taguru`'s library surface
(`src/mcp.rs:9-10` — `mcp` is dual-included into two binaries, not exported
from `src/lib.rs`), so this signature change is not an API-compatibility
concern under ADR 0005 either.

## 13. Consequences and follow-up

| Item | Disposition |
|---|---|
| `assemble_evidence` phase spans | Included in this PR; the one item to drop first if scope must shrink — every other section is load-bearing for #224's acceptance criteria. |
| SDK (Python/TypeScript) span emission | Separate implementation, reuses this document's span/attribute/event names; no server-side code. |
| A ratio/custom sampler, tail-sampling policy | Left to operator/collector configuration; out of scope. |
| Baggage propagation | Forwarded as-is when an application configures the baggage propagator; not filtered, since it is the application's own data and decision. |

## 14. Documentation impact

New `docs/tracing.html`: a Jaeger/Tempo docker-compose walkthrough, the
expected span tree, §6's attribute table, §8's privacy policy, and the
`trace_id`-from-JSON-log correlation recipe. Trace sections added to
`docs/architecture.html` and `docs/troubleshooting.html`; links added from
`docs/index.html`, `README.md`, `docs/docker-compose.html`,
`docs/kubernetes.html`. Each SDK README gains a `## Tracing` section. This
ADR's own PR ships the documentation — unlike ADR 0006/0007, there is no
separate follow-up issue for it.

## Appendix: requirement traceability

| #224 acceptance criterion | Section |
|---|---|
| 一回の `retrieve` が一つの論理 parent span としてexport される | §5 |
| resolve / describe / query / activate / citations / passage fallback が child span で区別される | §5 |
| BM25 / embedding / ANN / fusion の latency と outcome を追える | §5, §6 |
| remote MCP、stdio MCP、HTTP/SDK で trace context が連続する | §10 |
| router / shard / cross-context fan-out へ W3C context が伝播する | §10 |
| fallback / degrade / skip 理由が stable event code で分かる | §7 |
| cache hit/miss と result counts/bytes を低 cardinality 属性で確認できる | §6 |
| provider degrade 後に成功した retrieval を誤って root ERROR にしない | §2.5(a), §9 |
| question、concept、source、passage、credential が default span へ含まれない | §2.5(b), §8 |
| OTEL disabled 時の zero/near-zero overhead を維持する | §11 |
| parent-based sampling を尊重する | §11 |
| in-memory exporter による span tree / propagation test がある | §3 (`FakeCollector` decision) |
| tracing overhead benchmark がある | §11 |
| Jaeger または Tempo で確認できる local walkthrough がある | §14 |
| JSON log の trace ID から対応 trace を開ける | §5 (router request span), existing `trace_id` correlation |
