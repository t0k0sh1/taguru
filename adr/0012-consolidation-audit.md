# 0012. Consolidation audit — merge, contradiction, and staleness candidates, proposal-only

- **Status**: Accepted
- **Date**: 2026-08-07
- **Issue**: #421
- **Related**: #420 / ADR 0011, #423, ADR 0009 §9.2 / §9.5 / §10, ADR 0005
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

A long-lived context accumulates spelling-twin concepts, conflicting
associations, and facts the corpus has moved past — and today nothing
surfaces them as work. This ADR designs the audit lane that does: what the
server detects (three candidate classes), what shape the report takes, how
the LLM-judgment half stays incremental behind content fingerprints, and
which existing write APIs apply an accepted proposal. The posture is fixed
before the mechanics: **candidates for review, not verdicts** — the same
framing `audit_vocabulary` (`src/api/vocabulary.rs:38`) and ADR 0009 §10's
schema audit already carry, and nothing in this design auto-applies
anything.

Design-first, the #370 / #225 shelf. Out of scope, answered in §9: a
first-class concept-merge write operation, automatic application of any
proposal, server-side LLM calls, and cross-context consolidation.

## 2. Context: the machinery this composes, none of it new

Five shipped mechanisms are load-bearing here:

- **The communities artifact pattern** — the template the issue names.
  Server-side deterministic detection (`GET /contexts/{name}/communities`,
  algorithm-stamped `louvain-cc/1`), per-item **content fingerprints**
  (FNV-1a over sorted members + induced edges,
  `src/context/community.rs:36`), a client verb (`taguru communities`)
  that diffs against the previous artifact and calls the LLM **only for
  changed items** — an unchanged corpus costs zero LLM calls — and
  judgments written back through plain `POST /import` into a derived
  context (`{name}::communities`) under a `taguru_communities: u64`
  format stamp. Detection is the server's; meaning is the client's.
- **Twin detection already exists for names.** `vocabulary_audit`
  (`src/api/vocabulary.rs:83`) sweeps lexical twins (bigram-Dice via
  `EntryIndex::twins`, `src/context/entry_index.rs:190`) and semantic
  twins (gloss-embedding cosine, when an embedder is configured), shared
  with `drift/audit`'s `include_twins` extension. What no audit reads yet
  is **structure** — whether two similarly-spelled concepts also attest
  the same facts.
- **Assertion time (ADR 0011).** An edge's temporal position is its
  attributing sources' effective time (`date.or(stored_at)`), joined at
  the registry layer where the passage store's `SourceMeta` lives.
  ADR 0011 §7 explicitly hands this audit its input: dated, ordered
  assertions per `(subject, label)`.
- **The liveness conventions.** `query`/`query_any` materialize dead
  edges; every consumer filters `count > 0` itself (the convention
  `community.rs`, `traverse.rs`, `gloss.rs` follow). A negative-sum edge
  is *contested*, not dead; a sign-conflicted edge whose attributions
  disagree is its own signal, distinct from two objects disagreeing.
- **The audit frame.** Read-only heavy analyses run as `Role::Read`
  behind the heavy-ops ceiling (`src/main.rs:767`), deadline-prechecked,
  CPU-bound scans under `block_in_place` — the frame every existing
  audit route already follows.

And one hard boundary, discovered rather than chosen: **concept merge
does not exist as a write.** `AliasError::Conflict`'s own doc
(`src/context.rs:129`) refuses aliasing two spellings that both live as
concepts — "that is a merge, which is not a feature; rebuild the
context." §7 designs around this honestly instead of pretending an alias
is a merge.

## 3. Options considered — one audit or three

**A. Three endpoints, one per candidate class.** Rejected. The three
detections share one O(edges) grouped pass (contradiction and staleness
read the same per-`(subject, label)` grouping; merge shares the twin
sweep with `vocabulary_audit`), share one report vocabulary, and share
one consumer — the judging client. Three routes mean three role rows,
three heavy-op entries, and three places for the framing ("candidates,
not verdicts") to drift.

**B. Extend `drift/audit` with three more sections.** Rejected on
ownership: drift's checks are *bookkeeping* health (unsourced edges, dead
aliases) with a cheap default path and one conditional heavy extension.
Consolidation is unconditionally heavy, fingerprint-bearing, and exists
to feed a judgment artifact — a different consumer and lifecycle bolted
onto a route whose contract is "cheap unless asked."

**C. One endpoint, three sections, caller-selected.** Chosen; §4. The
schema audit's shape (ADR 0009 §10: independent checks, one response,
paged sections) with drift's lesson applied as a `checks` selector
rather than a conditional flag.

## 4. Decision

**One route, `POST /contexts/{name}/consolidation/audit`, returning up
to three independently paged sections — `merge`, `contradiction`,
`staleness` — selected by a `checks` array (required, non-empty: the
caller names what it pays for). Every candidate carries a content
fingerprint over its own evidence; the LLM-judgment half is a client
verb that reuses judgments for unchanged fingerprints, communities-style.
The server never judges, never applies, and never calls an LLM.**

Detection, per section — all over live edges only (`count > 0`, the §2
convention):

- **`merge`** — candidate pairs of concepts, generated by the existing
  twin sweep (lexical Dice and, when embeddings are configured, semantic
  twins — the `vocabulary_audit` core, reused not reimplemented) and
  **corroborated structurally**: the overlap of the two concepts'
  neighborhoods, as Jaccard over their live `(label, object)` (and
  incoming `(subject, label)`) pairs. Each pair reports both scores and
  the evidence itself — shared pairs, and each side's distinct pairs —
  because the judge's question ("same thing, differently spelled?") is
  answered by exactly that list. Name similarity without shared structure
  ranks low but still reports (young twins share little yet); structure
  without name similarity is out of scope for v1 (it detects *duplicate
  entities*, a semantic judgment this audit only reaches via the
  embedding tier, not a lexical one).
- **`contradiction`** — two kinds, both named by the issue, kept as one
  section with a `kind` discriminator. **Between objects**: a
  `(subject, label)` holding ≥ 2 live distinct objects, each row carrying
  weight, count, and its latest assertion time (the ADR 0011 join) —
  ordered rows, so "supersession or conflict" is askable at a glance.
  Ranked by the label's *functional tendency*, measured not declared:
  the fraction of this label's subjects that hold exactly one live
  object, computed in the same grouped pass. A label that is one-object
  for most subjects makes its multi-object outliers strong candidates; a
  naturally many-valued label (`contains`) sinks. **Within an edge**:
  sign-conflicted attributions — an edge whose per-source sums disagree
  in sign (the store's own "contested" vocabulary) — reported with both
  sides' sources. Schema presence changes nothing here in v1: cardinality
  is still ADR 0009 §9.2's deferral, and type-level violations already
  belong to `schema/audit` — this section is value-level, that one is
  type-level, and the two must not duplicate each other.
- **`staleness`** — edges left behind by their own neighborhood: the gap
  between an edge's latest attesting effective time and the maximum
  effective time across its subject's live edges, reported ranked by gap
  (a `floor_secs` parameter trims noise; no default pretends to know the
  corpus's tempo). Undated sources are invisible here — ADR 0011 §4's
  rule, inherited unchanged, and reported honestly: the section carries
  the count of edges it could not date, so "no candidates" and "nothing
  was datable" never look alike.

Report shape: sections are paged (the `page_by` mirror ADR 0009 §10
already uses), every section's caps are explicit — reported totals
alongside returned counts, never a silent top-N (the house rule) — and
every candidate carries `fingerprint: u64` (FNV-1a, the communities
choice) over its own sorted evidence: the pair's names + neighbor sets
for `merge`, the ordered object rows for `contradiction`, the edge
identity + both timestamps for `staleness`.

## 5. The judgment artifact: fingerprints are the increment and the staleness

The client half mirrors `taguru communities` deliberately — same
division of labor, same storage substrate:

- A client verb runs the audit, loads the previous judgment artifact,
  and calls the LLM **only for candidates whose fingerprint it has never
  judged**. Unchanged corpus, zero calls; a graph edit reshapes only the
  candidates it touched, and only those re-judge.
- Judgments — accept (with the proposed action), or **dismiss** — are
  written through plain `POST /import` into a derived context
  (`{name}::consolidation`), one source per candidate keyed by its
  fingerprint, under a `taguru_consolidation: u64` stamp
  (mismatch-checked like `taguru_communities`,
  `src/communities.rs:747`). Dismissals are first-class and are the
  point of the memory: a candidate the operator has judged benign must
  not re-cost an LLM call every audit — until its evidence changes,
  which its fingerprint detects by construction.
- **Fingerprint identity is the staleness mechanism.** A judgment binds
  to the evidence it judged; if the candidate reappears with a different
  fingerprint, the judgment simply does not match and the candidate is
  new again. No revision-comparison endpoint is needed in v1 — candidate
  granularity is strictly finer than the graph-revision granularity
  `communities/search` uses, and the derived context is a normal context
  readable by every existing means.

## 6. Applying a proposal: existing writes, spelled out per class

The audit proposes; the client applies, explicitly, through APIs that
all exist today:

- **Merge** — two cases, and the audit report says which one a pair is
  in. If one spelling has no live concept of its own (aliasing it is
  legal), the fix is an ordinary alias (`POST /contexts/{name}/aliases`)
  — ADR 0009 §9.5's own answer to spelling drift. If **both** spellings
  live, aliasing is refused by design (`AliasError::Conflict`, §2), and
  the honest application is re-attribution: retract the sources that
  attest the duplicate and re-import their batches under the canonical
  spelling — exact by the batch contract's retract-then-apply
  idempotency, and expensive in proportion to how real the duplication
  is. A first-class merge operation stays a named follow-up (§9), and
  this audit is deliberately the thing that will measure whether demand
  justifies its cost.
- **Contradiction** — the store's existing vocabulary, chosen by the
  judgment: a wrong assertion retracts (`retract_association` /
  `retract_source`); a contested-but-standing one takes a negative-weight
  assertion; a superseded-but-was-true one **stays**, dated — ADR 0011
  §6's posture, where as-of queries and the eventual interval derivation
  (the cardinality follow-up) give old truths their place instead of
  erasing them.
- **Staleness** — re-ingest the source's current truth (the connector
  path exists), or retract it if it has none. Never automatic expiry:
  #423's lifecycle umbrella owns forgetting, and this store's posture is
  explicit writes only.

## 7. What the schema adds, and what its absence costs

Detection is deliberately schema-independent in v1: twins, structural
overlap, liveness, per-label functional tendency, and assertion times all
exist without a schema document. A schema sharpens *judgment*, not
detection — `types` tell the judge two twin concepts are same-typed
(or that a merge would cross types, a strong dismissal signal), and the
report attaches each concept's live `schema:type` objects when present.
The one schema field that would change detection itself — declared
cardinality — does not exist yet (ADR 0009 §9.2); when its follow-up ADR
lands, the contradiction section's measured functional tendency gets a
declared override, and nothing else in this design moves. Type-name
concepts stay excluded from merge candidates, the same ADR 0009 §6.3
exclusion `vocabulary_audit` already applies — a type name's spelling is
the schema author's business, not drift.

## 8. Surface and compatibility

Additive throughout; `http_contract` stays 1 (ADR 0005 §4).

- **HTTP**: one new route, `POST /contexts/{name}/consolidation/audit`.
  `Role::Read` beside the other audits (`src/auth.rs:808`); the
  **unconditional** heavy-ops group (`src/main.rs:767`) — every section
  is O(edges) or worse, so there is no cheap default to protect, which
  is exactly the distinction that put `drift/audit` outside that group.
  Deadline prechecked, scans under `block_in_place`, the standing frame.
- **Report and artifact stamps**: the response carries the detector's
  algorithm/version stamp (the `louvain-cc/1` precedent — fingerprints
  are only comparable within one detector version, and a detector change
  invalidates every stored judgment *loudly*, the `communities.rs:675`
  behavior); the judgment artifact carries `taguru_consolidation: u64`,
  equality-checked. Both are new dimensions under ADR 0005 §3's
  "different owners, different consumers" rule — neither touches
  `http_contract`, `batch_formats`, or `image_formats`.
- **MCP / SDK**: one new tool / method, pass-through of the HTTP shape
  (ADR 0005 §3 boundary); additive.
- **Library**: the detection core that is graph-only (twin reuse,
  structural overlap, grouped contradiction pass) lands beside
  `Context::communities` as lib methods; the staleness join stays in the
  registry where `SourceMeta` lives — the same layering ADR 0011 §4
  fixed, and the lib surface grows additively.
- **On-disk / WAL / batch**: untouched. The judgment artifact is an
  ordinary derived context written through the ordinary import door —
  the same reason the communities artifact needed no format change.
- **Caching**: uncached, like every audit; nothing enters the retrieval
  cache and nothing invalidates.

Cost note: merge's structural corroboration is bounded by the twin
sweep's output (pairs, not all-pairs — the Dice floor and embedding
floors gate it), each pair costing its two adjacency walks;
contradiction and staleness share one grouped O(edges) pass plus the
registry's O(sources) metadata join. The heavy-ops permit is the
backstop, as everywhere.

## 9. Non-goals

- **A concept-merge write operation** — refused today by
  `AliasError::Conflict`'s design; re-attribution (§6) is the exact,
  existing path. If this audit demonstrates recurring demand, a merge op
  is its own ADR (WAL op, image semantics, attribution rewrite — none of
  it trivial), never a quiet extension.
- **Auto-application of any proposal** — the issue's own line, restated:
  detection is the server's, judgment is the client's, application is an
  explicit write.
- **Server-side LLM calls** — the README boundary holds; the server
  computes candidates and fingerprints, nothing more.
- **Declared cardinality** — still ADR 0009 §9.2's follow-up; §7 fixes
  the seam it will slot into.
- **Cross-context consolidation** — one context per audit; federating
  candidates across shards is #370's territory if it is ever anyone's.

## 10. Consequences and follow-ups

- #423's promotion workflow gains its missing prerequisite: promotion
  into a permanent context can now be preceded by an audit that says
  what the promotion would collide with.
- The audit will produce the first honest data on whether a real merge
  operation is worth its cost (§9) — the same measure-then-build gate
  ADR 0010 §9 applied to the prefix index.
- Implementation splits cleanly: (S1) lib — structural overlap +
  grouped contradiction pass beside the twin sweep; (S2) registry +
  HTTP — the staleness join, the route, paging, fingerprints; (S3) the
  client judge verb + judgment artifact, communities-patterned; (S4)
  MCP/SDK surface + docs. S1–S2 are inert until S3 gives the report a
  consumer; each lands green independently.
- The `evalset`-style acceptance question for the implementation issue:
  seeded twin/contradiction/staleness corpora with known ground truth,
  measuring candidate recall and the incremental property (second audit
  over an unchanged corpus must cost zero LLM calls).
