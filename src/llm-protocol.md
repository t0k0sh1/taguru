# Taguru client protocol

You (an LLM) are the intended client. This is the discipline for
ingesting knowledge into, and retrieving it from, an association
network. The server handles structure only — understanding language,
choosing contexts, decomposing documents into facts, and composing
answers back into prose are your job.

## Model

- Knowledge is **(subject, label, object, weight, source)**
  associations. Weight is signed; negative asserts "not" (「大量生産を
  行わない」→ `{"subject":"青嶺酒造","label":"行う","object":"大量生産","weight":-1.0}`).
- Re-asserting a triple adds weight and keeps per-source attributions:
  2 sources × 1.0 (independent corroboration) stays distinguishable
  from 1 source × 2.0 (one emphatic claim).
- **One context = one 文脈.** One spelling means one referent. Apple
  the fruit and Apple the company belong in different contexts.
- The graph is an index, not an archive. Register originals through
  the sources API and reach them back via attribution source ids.

## Retrieval loop

1. **Pick a context**: `GET /contexts` lists names, human-written
   descriptions, mechanical stats (association counts, top concepts,
   label sample — these never go stale), and usage counters (reads,
   empty reads, writes, last-read/write unix seconds). Torn between a
   few candidates? Search them together: `POST /recall`, `/query`, and
   `/sources/search` take `contexts: [full names]` and/or
   `groups: [group names]` (a group searches every context it reaches,
   nested children included; overlaps dedupe) and tag every match with
   its context — shortlist, search once, then continue inside the
   context that answered.
2. **Resolve cues**: extract entity and relation candidates from the
   question; `resolve` (concepts) / `resolve_label` (relations). The
   entry is normalized — width, case, katakana/hiragana, light typos
   all land. On a miss: reword, or lower `dice_floor` (default 0.3 →
   e.g. 0.2) to widen fuzzy matching for one call. If the server has
   embeddings, a semantic tier joins whenever lexical candidates are
   absent or weak (best score < 0.5): appended with `tier:"semantic"`,
   scored by cosine — never compare scores across tiers. Names are
   embedded as graph-context glosses, so paraphrases (醸造責任者→杜氏)
   and question-shaped cues land too. Still empty → probably the wrong
   context; try the next candidate.
   Lexical candidates carry `kind`: `exact`/`alias` mean the cue IS a
   stored spelling; `containment`/`fuzzy` mean it merely overlaps one —
   a high score there can be a lookalike, not the thing (京都 scores
   0.67 against 東京都, `possible` 0.8 against `impossible`). The top
   candidates carry `gloss`, the name plus its heaviest facts: read it
   before adopting a containment/fuzzy hit — string overlap says two
   names are near, the glosses say whether they are the same thing.
   Never adopt a lookalike on score alone.
   When a name you EXPECTED still fails to appear, don't bisect floors
   by hand: `POST /contexts/{name}/resolve/explain` (`explain_resolve`,
   `resolve_label/explain` for labels) with `{cue, expected}` answers
   why in one call — down to "not in the vocabulary at all; nearest
   stored spellings attached" (register an alias?) or "the cue is an
   exact spelling of something else, so nothing else was ever scored".
3. **Outline, then narrow**: `describe` a hub concept first (which
   labels, how many, per role), then `query` just the facets you need
   (`"label": ["住所","職歴"]`). Don't pull whole profiles.
4. **Expand and rank**: `activate` spreads from origins (strongest
   first, `path` shows the route; strength is an ordering within one
   call — never compare across calls). `explore` walks structure
   exhaustively with hop-distance annotations.
5. **Answer from the originals**: attributions from `recall`, `query`,
   `explore`, `activate`, and `unreachable_from` already carry a
   resolved `section` label — enrichment on the graph read, not
   something you need to fetch (`null` with no `paragraph` locator, or
   when the locator falls outside every section the source has
   stored). For the verbatim text itself, call
   `POST /contexts/{name}/citations` with `{source, paragraph}`
   when a `paragraph` locator is present — one excerpt, with the same
   `section` alongside it. Without a `paragraph`, there is no located
   excerpt; feed the source id to `POST /contexts/{name}/sources/lookup`
   instead and ground your wording in the whole passage. Reflect
   negative weights as negation and attribution counts as strength of
   support.
6. **Switch to the text lane**: knowledge that never fit a triple
   (procedural detail, conditions, discourse) was never in the graph.
   When graph results can't compose the answer, run
   `POST /contexts/{name}/sources/search` — paragraph search with a
   lexical lane (BM25) fused with a semantic lane (paragraph
   embeddings) where the server has them. Graph first, text as the
   safety net.
   - **Phrase the query as an answer, not a question.** Most embedding
     models place a question ("What plan includes SSO?") measurably
     farther from its answer ("SSO is available on the Enterprise
     plan.") than two independently phrased statements sit from each
     other. Guess a plausible declarative sentence and search with
     THAT — the guess does not need to be correct, only shaped like
     the text you hope to find. It costs nothing when the lexical lane
     would have found it anyway, and recovers what the lexical lane
     alone would miss.
   - Each hit's `lanes` field says which lane surfaced it at what
     rank. A vector-only hit is the paraphrase case; a BM25-only hit
     matched wording. Both are evidence, not verdicts — read the text.
   - The response's `plan` says what the search chose to do at all:
     which contexts were searched and, per context, whether the
     semantic lane actually ran — and why not when it did not
     (embeddings off, nothing embedded yet, model or vector width
     changed, provider refused) — plus the effective cosine floor
     when it did. Check
     BOTH lanes before concluding "not in the corpus": empty `hits`
     with `bm25: {ran: true}` and `vector: {ran: false}` is a
     lexical-only answer, not a semantic miss — and when both lanes
     say `ran: false`, nothing searched at all (a no-term query, or
     `limit: 0`).
   - `semantic_floor` (0–1) overrides the vector lane's cosine floor
     for one call — raise it to keep only strong paraphrase matches,
     lower it to let weak ones through. It floors only that lane:
     BM25-only hits still return (the fused score is rank arithmetic
     and has no floorable scale).
   - **When a search misses something you know is there**, ask the
     server why instead of re-searching with varied wording:
     `POST /contexts/{name}/sources/search/explain` (`explain_search`)
     with `{query, source}` names the first reason that applies —
     never stored (or retracted), no shared term (the spelling
     mismatch shown as strings on both sides), or ranked past your
     limit (with a limit that verifiably reaches it). Report the
     verdict, or repair it: register an alias, re-import the source,
     or widen the limit.

## Ingest loop

1. Decompose the document into (subject, label, object, weight).
   - **Check before mint**: `resolve` / `resolve_label` before coining
     any spelling; reuse what exists. `GET /contexts/{name}/labels`
     lists the relation vocabulary. A near-hit whose gloss shows a
     DIFFERENT thing (a lookalike, not your entity) → keep your own
     spelling and record the distinction (step 5), so the collision
     warns instead of confusing from then on.
   - Don't re-assert paraphrases within one document (inflates
     weight). DO re-assert across documents (that's corroboration).
   - Negation: positive label, negative weight.
   - Make implicit membership explicit (whose 杜氏 is 高瀬? — add the
     edge).
2. `POST /contexts/{name}/associations` in batches — one document per
   request, up to 10,000 associations, with a `source` on every element.
   Split a larger document across requests; for corpus-scale ingestion,
   use `POST /import` or `taguru import` instead.
   A single-association request still pays for a full durable write —
   roughly two orders of magnitude more per association than a batched
   request — and stalls that context's readers while its fsync lands.
   Batching, not concurrency, is the lever: writes to one context
   serialize by design; writes to different contexts run in parallel.
3. Register originals: `POST /contexts/{name}/sources` (source id →
   passage). Store the document's full text as-is: the server splits it
   into paragraphs internally (blank-line boundaries) and searches at
   paragraph granularity, so a long document does not bury its best
   paragraph. Blank lines between logical units are what make that
   split work — keep them. Optionally attach doc2query `questions` —
   per paragraph, the questions a user would type whose answer is that
   paragraph, phrased AWAY from its wording. They index INTO the
   paragraph on every server: their terms join its BM25 postings (so a
   question-shaped search lands lexically even with no embedding
   provider), and on servers with passage embedding they also embed
   beside the paragraph and catch what its own vector misses.
4. Audit reachability: `POST /contexts/{name}/unreachable_from` with
   the document's main entities. Non-empty = membership edges are
   missing. If embeddings are configured, finish with
   `POST /contexts/{name}/embeddings/refresh` (diff-only, idempotent;
   unnecessary when the server runs `TAGURU_EMBED_AUTO`).
5. At milestones, `POST /contexts/{name}/vocabulary/audit` lists fork
   candidates (lexical twins = spelling drift, semantic twins =
   synonym forks). Candidates, not verdicts — adjudicate each pair:
   - Same referent → pick one canonical, point an alias at it, use
     the canonical from then on. (Forks that already accumulated
     facts cannot be merged — that's rebuild territory.)
   - Different things that will keep colliding (前株/後株 company
     names, 東京都/京都) → record the distinction as an ordinary
     fact: `{"subject": "株式会社青嶺", "label": "別物", "object":
     "青嶺株式会社", "weight": 1.0}`. One direction is enough —
     glosses carry incoming edges too, so both names warn in
     resolve's evidence from then on, even before either concept has
     any other fact. Use one label consistently per context (`別物`,
     or `distinct_from` in English vocabularies).
6. When live wording misses, register alternate spellings:
   `POST /contexts/{name}/aliases`. Aliases are entry-only; results
   always return the canonical. An alias cannot join two existing
   concepts (that would be a merge — rebuild territory). A
   mis-registered spelling is withdrawn with `DELETE` on the same
   path (exact spellings; canonical names are refused — removal
   cannot unname a record), which frees it to point elsewhere.
7. **Document updated? Sync the diff**:
   `POST /contexts/{name}/sources/retract` withdraws the old version's
   contributions (weights, attributions, passage), then ingest the new
   version normally. Concepts and edges remain; only weights come
   down.
8. **One fact wrong?** Pick the correction that matches what happened:
   an extraction error or merge mistake — a fact that should never
   have been asserted — is withdrawn outright with
   `POST /contexts/{name}/associations/retract` `{subject, label,
   object}` (every source's contribution to that one edge; the rest of
   each document stays). A fact the world CONTESTS is asserted with
   negative weight instead, which preserves the dispute as evidence.

## Procedures (ordered knowledge)

Steps become concept nodes woven with three kinds of edges — no new
machinery, same rank as membership edges and negative weights:

```json
[{"subject":"日本酒の醸造","label":"最初の工程","object":"洗米","weight":1.0,"source":"工程書"},
 {"subject":"洗米","label":"次の工程","object":"浸漬","weight":1.0,"source":"工程書"},
 {"subject":"日本酒の醸造","label":"工程","object":"洗米","weight":1.0,"source":"工程書"}]
```

- **Order** is the `次の工程` chain (fix ONE canonical label; branches
  are just several `次の工程` edges — a DAG). **Start** is
  `最初の工程`. **Membership** (`工程`) ties every step to the hub,
  for coverage audits.
- **Replay**: query the start, then `query {label:"次の工程"}` for all
  pairs at once and sort them (or step along with the subject pinned).
  **Never use `explore` distance for order** — membership edges create
  hub shortcuts that disagree with chain position.
- Same-named steps shared across procedures → qualify the name
  (「醸造の蒸米」). One spelling, one referent applies to steps too.
- Sources disagreeing about order surface as low-weight `次の工程`
  edges — normal weight arithmetic.
- Step detail (amounts, conditions, tips) stays in sources, found via
  `sources/search` — don't force it into triples.

## Causality

Cause → effect as directed edges, with canonical labels (`引き起こす`
/ `高める` / `防ぐ` / `要因`… — `resolve_label` before minting):

```json
[{"subject":"ストレス","label":"引き起こす","object":"不眠","weight":1.0,"source":"論文A"},
 {"subject":"カフェイン","label":"引き起こす","object":"不眠","weight":-0.8,"source":"論文C"},
 {"subject":"運動","label":"防ぐ","object":"不眠","weight":1.0,"source":"論文D"}]
```

- "Why X" = `query {label:["引き起こす","高める","要因"], object:"X"}`;
  "what does X lead to" pins the subject instead. Chains come back
  through `activate` paths — the system shows A→B→C but never asserts
  A→C; transitivity is your judgment.
- Disputed causality = small net weight + split attributions on one
  edge. Say so in the answer.
- Denial ("does not cause") is negative weight on the causal label;
  prevention is its own positive label. Don't conflate them.
- Weight is evidence mass, never effect size. "2× risk" goes in the
  object (`喫煙 →リスク倍率→ 2倍`) or stays in the passage.
- Correlation stays `相関する`; don't promote it to causation.
- Conditional ("only when fasting") or compound ("A and B together")
  causes: reify an event/compound node, or leave them to the text
  lane.

## Code

Source code takes the same discipline; only the naming changes.

- Concepts are identifiers in their exact spelling, SHORT names:
  `fetch_block`, not `CacheStore::fetch_block` — qualification dilutes
  entry scores. Namespace with edges (`defined_in` → `src/store.rs`);
  files and paths are concepts too.
- Fix a small label vocabulary up front and reuse it: `kind`,
  `defined_in`, `calls`, `field`, `variant`, `returns`, `invariant`,
  `purpose` cover most code facts.
- Case twins (`Frame` the type, `frame` the accessor) are separate
  concepts and `resolve` returns both at 1.0 — fan out over top-score
  ties and disambiguate with `describe`. Never merge them.
- The entry absorbs camelCase/snake_case/case/typos: `fetchBlock` and
  `evict_cld` land on the snake_case originals. Add natural-language
  aliases onto identifiers (`退避ループ` → `evict_cold`) for language
  entry.
- Passages: one function or type per source id
  (`src/wire.rs:seal_frame`), raw code as the text. Passage search
  matches whole identifiers and their camelCase/snake_case pieces;
  crossing languages (Japanese cue → English comment) is the semantic
  tier's job, not the text lane's.
- The highest-value facts are what grep cannot answer: invariants,
  purposes, design constraints — store them as edges on the
  identifier.

## API

| Method | Path | Body / returns |
|---|---|---|
| GET | `/contexts` | `?limit=1000&after=name` → `{total, contexts:[{name, description, pinned, loaded, dice_floor, semantic_floor, stats, usage, revision}]}` (keyset paging by name; `revision` = change counters `{graph, passages, config}` — graph writes, passage writes, and config/embedding changes respectively; equal counters ⇒ that lane's answers are unchanged since you last looked, so a cache can key on them — compare for EQUALITY only, and re-check after a server restart: a crash can lag a cold context's counters until its first load, and delete-recreate restarts them; the server itself already runs an exact-match result cache keyed this way, so repeating an identical recall/query/search is cheap without any client-side cache — and, where the operator enabled it, a guarded semantic tier that answers paraphrased passage searches from an equivalent earlier query's entry) |
| GET | `/contexts/{name}` | one directory row / 404 |
| PUT | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → create |
| PATCH | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → update metadata |
| DELETE | `/contexts/{name}` | delete, files included |
| GET | `/groups` | `?limit=1000&after=name` → `{total, groups:[{name, description, contexts, groups, fingerprint}]}` (keyset paging by name; a group bundles contexts many-to-many and may nest child groups — `groups` — at most 3 tall, cycles refused; `fingerprint` = one change token over the transitive member contexts' `revision` counters — it moves exactly when a member you can see changed: a write, an embedding refresh, a rename, or a membership edit — same equality-only, re-check-after-restart contract as `revision`) |
| GET | `/groups/{name}` | one group row / 404 |
| PUT | `/groups/{name}` | `{description?, contexts?:[name], groups?:[name]}` → create (groups and contexts are separate namespaces; every listed member — context or child group — must exist) |
| PATCH | `/groups/{name}` | `{description?, add_contexts?, remove_contexts?, add_groups?, remove_groups?}` → the updated row (deltas, not a replacement list; removals apply first; added members must exist, removing a non-member is a no-op; the result holds at most 1000 member contexts and 1000 child groups — `over_limit` past that; split into nested child groups) |
| DELETE | `/groups/{name}` | delete the bundling only — member contexts and child groups are untouched (deleting a context or a group also drops it from every group) |
| GET | `/groups/{name}/export` | the group as one import-stream record (a `taguru_group` JSON Lines line, not the JSON envelope) — `POST /import` (or `taguru import`) restores it as a create-or-replace of the WHOLE record; batches in the same stream apply first, so a group and its member contexts can travel together in any order |
| POST | `/contexts/{name}/associations` | `[{subject,label,object,weight,source?,paragraph?}]` → applied count (`paragraph` locates the fact within `source` and is ignored without one) |
| POST | `/contexts/{name}/recall` | `{cue, limit?, after?}` → `{total, matches, plan}` (`plan.contexts` = the contexts actually searched — trivially `[name]` here; the cross variants are where it earns its place) |
| POST | `/recall` | `{contexts?:[name], groups?:[group], cue, limit?, after?}` → `{total, matches, plan}` — recall across several contexts at once (full names, and/or groups: each searches every context it reaches, nested children included, overlaps deduped; every match tagged with its `context`; past the limit the strongest \|weight\| survives, one scale across contexts; `plan.contexts` = the RESOLVED target list in effective order — groups expanded, your key's grants applied — so a target that came back empty is still visibly distinct from one your grant dropped) |
| POST | `/contexts/{name}/query` | `{subject?, label?, object?, limit?, after?}` — each position a string or an array → `{total, matches, plan}` |
| POST | `/query` | `{contexts?:[name], groups?:[group], subject?, label?, object?, limit?, after?}` → `{total, matches, plan}` — query across several contexts at once, same contract as `POST /recall` |
| POST | `/contexts/{name}/describe` | `{concept}` → label outline (counts per role) / null |
| POST | `/contexts/{name}/explore` | `{origins, max_depth?, limit?, after?}` → `{total, matches:[{distance, path, association}]}` (hop cap 10, applied when omitted; truncation keeps the nearest) |
| POST | `/contexts/{name}/activate` | `{origins, decay?=0.5, limit?=20}` → `{total, matches:[{strength, path, association}]}` |
| POST | `/contexts/{name}/resolve` | `{cue, dice_floor?, semantic_floor?, limit?}` → `[{name, score, tier, kind?, gloss?}]` concept candidates (limit default/ceiling 1000) |
| POST | `/contexts/{name}/resolve_label` | `{cue, dice_floor?, semantic_floor?, limit?}` → `[{name, score, tier, kind?, gloss?}]` relation candidates (limit default/ceiling 1000) |
| POST | `/contexts/{name}/resolve/explain` | `{cue, expected, dice_floor?, semantic_floor?, limit?}` → one verdict for "why didn't (or did) `expected` come back for `cue`", first that applies: `not_in_vocabulary` (nearest stored spellings attached — register an alias?) / `cue_resolved_exactly` (the cue IS another stored spelling; the exact tier answers alone) / `below_floor` (its actual score vs the floor in effect) / `below_cutoff` (rank, plus a `limit_to_reach` verified by rerunning the serve) / `semantic_not_run` / `semantic_below_floor` (gloss cosine vs the semantic floor, or which precondition failed) / `served` — same floors and limit as the resolve call being explained |
| POST | `/contexts/{name}/resolve_label/explain` | the same, for relation labels |
| POST | `/contexts/{name}/embeddings/refresh` | re-embed new/changed concept and label glosses (run after ingest) |
| GET | `/contexts/{name}/embeddings` | `{provider_model, glosses?:{model, width, concepts, labels}, passages?:{model, width, rows}}` — the embedding identity: the provider configured now beside the (model, width) each vector sidecar was actually built with (`provider_model` null = embeddings off; a sidecar absent = nothing embedded yet; the two models disagreeing = refresh needed) |
| GET | `/contexts/{name}/labels` | `?limit=1000&after=label` → `{total, labels:[...]}` relation vocabulary (canonical only, keyset-paged by label) |
| GET/POST/DELETE | `/contexts/{name}/aliases` | `?limit=1000&after=concept:x\|label:x` → `{total, concepts:{alias:canonical}, labels:{...}}` (one page across both namespaces, concepts first; `after` = the last entry shown) / register `{concepts:{alias:canonical}, labels:{...}}` / withdraw `{concepts:[alias], labels:[...]}` |
| GET/POST | `/contexts/{name}/sources` | `?limit=1000&after=id` → `{total, sources:[...]}` registered source ids (keyset-paged) / `{passages:{source:text}, questions?:{source:[{paragraph, question}]}, sections?:{source:[{paragraph, section}]}}` → `{stored, questions_stored, questions_dropped, sections_stored, sections_dropped}` (a dropped question or section named a paragraph its text's blank-line split does not have) |
| POST | `/contexts/{name}/sources/lookup` | `{sources:[...]}` → `{passages, missing}` |
| POST | `/contexts/{name}/sources/search` | `{query, limit?=5, semantic_floor?}` → `{plan, hits:[{source, paragraph, score, text, lanes}]}` best PARAGRAPHS across passages (`paragraph` = its position in the source; `text` = that paragraph alone; `lanes.bm25`/`lanes.vector` = per-lane `{rank, score}`; `score` is rank-fused when the vector lane ran, raw BM25 otherwise; `semantic_floor` (0–1) overrides the vector lane's cosine floor for this call — context setting, then server default, otherwise — flooring only that lane). `plan.contexts` = one `{context, lanes:{bm25:{ran, reason?}, vector:{ran, reason?, floor?}}}` per context actually searched: whether each lane ran there and why not when it did not (the same wording `search/explain` uses), the vector lane's effective floor when it did — so zero hits with a skipped semantic lane no longer looks like "nothing matched" |
| POST | `/sources/search` | `{contexts?:[name], groups?:[group], query, limit?=5, semantic_floor?}` → the same `{plan, hits}` wrap, each hit tagged with its `context`, across several contexts at once (groups resolve as in `POST /recall`) — merged by per-context rank (every context's best hit first); `score` compares within one context only; `plan.contexts` carries one entry per resolved target in effective order (per-context floors included — a context's own `semantic_floor` setting shows here) |
| POST | `/contexts/{name}/sources/search/explain` | `{query, source, paragraph?, limit?=5, semantic_floor?}` → one verdict for "why didn't (or did) this source appear for this query" — same floor override as the search call being explained — first that applies: `not_stored` (never stored here, or retracted — the store keeps no tombstone history to tell which) / `paragraph_out_of_range` / `no_query_terms` / `no_term_overlap` (the query's terms and the paragraph's terms side by side, AS STRINGS — the spelling-mismatch case: stored 酒蔵, searched 酒造) / `below_cutoff` (its rank, the cutoff score at your `limit`, and a `limit_to_reach` verified by rerunning the real serve computation, pool caps included) / `served` — evidence carries per-term BM25 tf/df/idf/contribution (the very addends search summed) and the vector lane's cosine, or the reason that lane never ran. `paragraph` omitted picks the source's best showing |
| POST | `/contexts/{name}/citations` | `{source, paragraph}` → `{text, source, section}` one verbatim paragraph by source and paragraph — the same paragraph `sources/search` would show at that paragraph (`section` is the label governing that paragraph, `null` outside every section the source has stored; `recall`/`query`/`explore`/`activate`/`unreachable_from` resolve the same label onto each attribution as `attributions[].section`) |
| POST | `/contexts/{name}/sources/retract` | `{source}` → withdraw that source's contributions (diff sync) |
| POST | `/contexts/{name}/associations/retract` | `{subject, label, object}` → `{retracted, attributions_removed}` — withdraw ONE association outright, every source's contribution to that edge (names resolve through aliases; `retracted: false` = no live edge, nothing changed; the edge row stays visible at weight 0 until compaction, and re-asserting later just works). For a fact that should never have been asserted; a fact that is merely CONTESTED wants a negative-weight assertion instead |
| POST | `/contexts/{name}/unreachable_from` | `{origins, limit?, after?}` → `{total, matches}` unreachable associations |
| POST | `/contexts/{name}/vocabulary/audit` | `{dice_floor?=0.6, cosine_floor?=0.6}` → spelling/synonym fork candidates |
| GET | `/contexts/{name}/export` | the context as an import batch stream (JSON Lines body, not the JSON envelope) — one batch per source, create block first, aliases last; `POST /import` (or `taguru import`) restores it, per-source retract-then-apply, answering `{batches: [...]}` in stream order (`taguru_group` records ride the same stream, restore after every batch as whole-record replaces, and answer under `groups: [...]`) |
| POST | `/contexts/{name}/compact` | rebuild the image without dead records (admin; the context's requests wait out the rebuild) → `{bytes_before, bytes_after, dead_edges, aliases_dropped}` |
| POST | `/maintenance/compact` | `?min_dead_ratio=0.0` (default; any dead weight at all) → sweep every context whose live dead ratio strictly exceeds it, worst ratio first, each rebuilt like `/contexts/{name}/compact`; admin, server-wide (refused for a context-scoped key, like `/flush`) — closes the server to ordinary traffic for the sweep (`/health` answers `503 maintenance` meanwhile, distinct from an actual fault) and reopens when it ends or the deadline cuts it short → `{contexts:[{name, bytes_before, bytes_after, dead_edges, aliases_dropped}], deadline_exceeded}` |

## Auth

- If the server sets `TAGURU_API_TOKEN`, every request except
  `/health`, `/live`, and `/metrics` needs
  `Authorization: Bearer <token>`; missing or wrong → `401` in the
  error shape below.
- The MCP bridge (taguru-mcp) reads its own `TAGURU_API_TOKEN` and
  attaches it to every request — when the server turns auth on, set
  the same value on the bridge.
- Unset = auth disabled (dev mode; never expose beyond localhost).
- Keys may carry a scope (`TAGURU_KEY_SCOPES`): a role — read (the
  retrieval loop) ⊂ write (+ the ingest loop, group create/update) ⊂
  admin (+ context and group deletion, `/import`, `/flush`,
  `/maintenance/compact`) — and
  optionally a context list. Out of scope → `403` in the error shape,
  naming what the key lacks; a context-scoped key sees only its grant
  in `GET /contexts`, group listings — and the group export — show it
  only the members it may see (child group names stay visible — they
  are labels, not content), and a cross-context search naming a context beyond the
  grant in `contexts` — or a group write touching one, counted
  through nested children — is refused whole. A cross-search `groups`
  entry instead resolves to just the members the grant covers, the
  same slice the listing shows: a refusal there would name what the
  listing hides. Scopes bind MCP tool calls exactly as raw HTTP.

## Errors and limits

Every JSON error answers ONE shape:
`{"status": "error", "code": "<kind>", "error": "<text>", "time": <s>}`.
`error` is prose for you to read; `code` is the STABLE machine
vocabulary to branch on (never match on message wording):
`malformed_request` (the request never parsed: broken JSON, wrong
Content-Type, mistyped shape) / `invalid_argument` (parsed, but a
value was refused: empty or oversized name, bad weight, bad cursor) /
`over_limit` (a batch or list over its per-request cap — split and
resend) / `unauthorized` / `forbidden` / `no_context` / `no_source` /
`no_paragraph` / `no_group` / `unknown_path` / `method_not_allowed` / `timeout` /
`already_exists` / `conflict` / `payload_too_large` / `rate_limited` /
`internal` / `embeddings_unconfigured` / `embeddings_failed` /
`overloaded` (shed at the global in-flight ceiling or the shared
heavy-operation ceiling for vocabulary audits/context compactions;
wait `Retry-After`) /
`unhealthy` (the write path is degraded) / `maintenance` (a
`POST /maintenance/compact` sweep is running — wait `Retry-After` and
retry) / `storage_full` / `read_only_replica` (403: this server is a
read replica — do NOT retry here; send the write to the writer the
message names) / `shard_unreachable` (502 from a `taguru route`
router: a shard this request needs did not answer — retry once the
shard or its load balancer does).

- `401` auth (above). `404` unknown context or group. `409` duplicate
  create / alias conflict / a `POST /maintenance/compact` overlapping
  one already running.
- `507` context full (`storage_full`) — the write was NOT applied.
  Two ways here: the context hit the library's own capacity cap
  (further knowledge goes to a new context), or it reached a
  per-context storage quota the operator declared
  (`TAGURU_CONTEXT_QUOTAS`) — the message says which. At a quota,
  retractions, alias removals, `DELETE`, and compaction still work:
  shrink the context (or have the operator raise its quota) and
  retry; do not blindly re-send the refused write.
- `501` `/embeddings/refresh` without a provider configured
  (server-side TAGURU_EMBED_*). `502` embedding provider failure
  (refresh, or the semantic fallback inside resolve) — retry later.
- `400`: association batch over 10,000 per request (nothing applied —
  split and resend; alias batches and removals share the same cap) /
  list-shaped read input over 1,000 items (origins, query terms,
  `sources/lookup` sources, cross-search `contexts` and `groups` —
  split the request) / weight not finite
  or |weight| > 1,000,000
  (whole batch refused) / name too long (subject, label, object,
  source, alias ≤ 1024 bytes — names are headings, not bodies:
  passages go to sources, long knowledge gets decomposed; context or
  group name ≤ 64, description ≤ 4096) / group nesting over the cap
  (`over_limit`, at most 3 groups tall) or closing a cycle
  (`invalid_argument`). `408` timeout (default 30 s —
  narrow the query and retry). `413` body over the cap (default
  8 MiB).
  `429` this key is over its request budget — wait the `Retry-After`
  seconds and continue; prefer batching writes over rapid-fire calls.
- Off-axis errors answer in the same shape: unknown path `404`, right
  path wrong verb `405`, broken JSON `400`, wrong Content-Type `415`,
  well-formed but mistyped JSON `422`.
- recall / query / explore / unreachable_from (and the cross-context
  `POST /recall` / `POST /query`) default `limit` to 100. `total`
  above the returned count = truncation (recall/query/
  unreachable_from keep the strongest |weight|, explore keeps the
  nearest hops). Narrow or raise `limit` — capped at 1000 everywhere —
  or page past it with `after`: copy `weight`/`subject`/`label`/
  `object` (plus `context` too for the cross-context forms) from the
  last match, or `distance`/`subject`/`label`/`object` from explore's
  last recollection, verbatim from the previous page's last row.
  `total` stays constant across pages; stop once `matches` comes back
  empty.
- The keyset-paged listings — `/contexts` and `/groups` (`after` = the
  last `name`), `/contexts/{name}/labels` (`after` = the last `label`),
  `/contexts/{name}/sources` (`after` = the last `id`), and
  `/contexts/{name}/aliases` (`after` = the last `concept:x|label:x`) —
  page by the last row's key, not by rank, and their `total` is a live
  count independent of your cursor. As with the match endpoints above,
  a page can come back shorter than `limit` — a row deleted or
  retracted in the instant it is read drops from that page while the
  rows after it still follow — so a short page is not the last one.
  Stop only once a page comes back empty.
- Behind a `taguru route` router (sharded deployments), the
  cross-context searches and the `/contexts` listing may answer 200
  with an extra top-level `unreached` array —
  `[{shard, contexts, error}]` — when a shard could not be REACHED:
  the results are real but partial (that shard's contexts are
  missing). Treat a non-empty `unreached` as a partial view; retry for
  the full one. A shard that answered an error fails the request whole
  instead, exactly as one failing context does on a single server.
- A write that returned 200 is durable via the WAL (it survives a
  crash and replays on restart). Only when the server runs
  `TAGURU_WAL=0` can writes inside the flush interval (default 5 s)
  be lost.

## Compatibility

- This protocol travels WITH the server — read it from the deployment
  you target (`GET /protocol`, or the MCP instructions, which carry
  the same text). There is deliberately no `/v1` path prefix: one
  server serves one protocol version, its own.
- Parse responses tolerantly: new fields may appear in any release
  (additive), and absent optional fields are omitted rather than
  null. Pre-1.0, shapes may also change between minor versions —
  every break is named in the CHANGELOG's "Changed" section before it
  ships.
- The batch format (`taguru_batch: 1`) and the image format are
  versioned independently of the API: old batch files stay readable,
  and images migrate forward on load. Rolling a server BINARY back
  past an image-format bump needs the data rolled back with it — the
  release notes flag format bumps.
