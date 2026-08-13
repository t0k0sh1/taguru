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
   exhaustively with hop-distance annotations. When the question is
   "how are these two concepts related?", use `paths` instead of
   eyeballing either: it returns the concrete trails between origins
   and targets, shortest first, each hop carrying its association and
   citations — recompose the connection from the trail, and treat a
   trail as a chain of stored assertions, never as one asserted fact.
   - Time-scoped questions ("as of 2023", "what changed since"):
     `recall`/`query`/`explore`/`activate` take the same `since`/
     `until` window passage search does (epoch seconds, half-open,
     over each source's `date ?? stored_at`), one context at a time.
     A windowed read serves only facts an in-window source attests,
     with weight and citations re-derived from those attributions —
     `until` alone reads *as-of*. This is assertion time, not a
     validity interval: a superseded fact still answers unwindowed,
     dated; judging which of two dated answers still holds is your
     job, from the dates on their citations.
5. **Answer from the originals**: attributions from `recall`, `query`,
   `explore`, `activate`, and `unreachable_from` already carry a
   resolved `section` label and typed citation `locator` (a
   page/slide/sheet/table position) — enrichment on the graph read, not
   something you need to fetch (each `null` with no `paragraph`
   locator, or when the paragraph falls outside every marker the
   source has stored). For the verbatim text itself, call
   `POST /contexts/{name}/citations` with `{source, paragraph}`
   when a `paragraph` locator is present — one excerpt, with the same
   `section`/`locator` alongside it. Without a `paragraph`, there is no
   located excerpt; feed the source id to
   `POST /contexts/{name}/sources/lookup` instead and ground your
   wording in the whole passage. Reflect negative weights as negation
   and attribution counts as strength of support.
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
   - "Only documents tagged X", "only documents from the last year":
     pass `tags` (any-of, matched against tags stored with each
     source) and/or `since`/`until` (epoch seconds, half-open, over
     each source's `date ?? stored_at`) — the filter runs BEFORE the
     lanes, so `limit` counts eligible hits and the semantic lane is
     never spent on sources that could not qualify. The plan's
     `filter` block says how many sources were eligible; a source
     stored without metadata never matches a filter.
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
   Atomicity boundary: this call and step 3's `sources` call are
   SEPARATE writes — a document's facts can land while its passage
   store still fails, or vice versa. When a single all-or-nothing
   write across facts, aliases, and passage for one source matters,
   use `POST /import`/`taguru import` instead (retract-then-apply,
   whole batch or nothing).
3. Register originals: `POST /contexts/{name}/sources` (source id →
   passage). Store the document's full text as-is: the server splits it
   into paragraphs internally (blank-line boundaries) and searches at
   paragraph granularity, so a long document does not bury its best
   paragraph. Blank lines between logical units are what make that
   split work — keep them. Declare `dates` whenever the document has
   one: a source's `date ?? stored_at` is what time-windowed search —
   passage AND graph lanes alike — filters on, so a dated corpus
   answers "as of 2023" for free, and a source with no stored passage
   has no metadata and is invisible to every window. Optionally attach doc2query `questions` —
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
| GET | `/contexts` | `?limit=1000&after=name` → `{total, contexts:[{name, description, pinned, loaded, dice_floor, semantic_floor, stats, usage, revision, schema_mode}]}` (keyset paging by name; `schema_mode` = `"off"`/`"warn"`/`"strict"` echoed read-only from the installed schema's own `mode` so a client can route without a second `GET /schema` call, `null` for a context that never installed one — never a bare `"off"` standing in for "no document"; `revision` = change counters `{graph, passages, config}` — graph writes, passage writes, and config/embedding changes respectively; equal counters ⇒ that lane's answers are unchanged since you last looked, so a cache can key on them — compare for EQUALITY only, and re-check after a server restart: a crash can lag a cold context's counters until its first load, and delete-recreate restarts them; the server itself already runs an exact-match result cache keyed this way, so repeating an identical recall/query/search is cheap without any client-side cache — and, where the operator enabled it, a guarded semantic tier that answers paraphrased passage searches from an equivalent earlier query's entry) |
| GET | `/contexts/{name}` | one directory row / 404 |
| PUT | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → create |
| PATCH | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → update metadata |
| DELETE | `/contexts/{name}` | delete, files included |
| POST | `/contexts/{name}/rename` | `{to}` → rename (admin): the whole file family moves to `to` and every group naming `name` is rewritten to match; refused when `to` already exists, and while either name is mid-rename, -create, or -delete (retry shortly). The destination lives in the body, so a context-scoped key needs `to` in its grant too, like `/import` |
| GET | `/groups` | `?limit=1000&after=name` → `{total, groups:[{name, description, contexts, groups, fingerprint}]}` (keyset paging by name; a group bundles contexts many-to-many and may nest child groups — `groups` — at most 3 tall, cycles refused; `fingerprint` = one change token over the transitive member contexts' `revision` counters — it moves exactly when a member you can see changed: a write, an embedding refresh, a rename, or a membership edit — same equality-only, re-check-after-restart contract as `revision`) |
| GET | `/groups/{name}` | one group row / 404 |
| PUT | `/groups/{name}` | `{description?, contexts?:[name], groups?:[name]}` → create (groups and contexts are separate namespaces; every listed member — context or child group — must exist) |
| PATCH | `/groups/{name}` | `{description?, add_contexts?, remove_contexts?, add_groups?, remove_groups?}` → the updated row (deltas, not a replacement list; removals apply first; added members must exist, removing a non-member is a no-op; the result holds at most 1000 member contexts and 1000 child groups — `over_limit` past that; split into nested child groups) |
| DELETE | `/groups/{name}` | delete the bundling only — member contexts and child groups are untouched (deleting a context or a group also drops it from every group) |
| POST | `/groups/{name}/rename` | `{to}` → rename the bundling (admin): the group's file moves to `to` and every OTHER group naming `name` as a child is rewritten to match, member contexts untouched; refused when `to` already exists. Renaming touches every member's grant — nested included — so a context-scoped key needs them all, exactly like DELETE |
| GET | `/groups/{name}/export` | the group as one import-stream record (a `taguru_group` JSON Lines line, not the JSON envelope) — `POST /import` (or `taguru import`) restores it as a create-or-replace of the WHOLE record; batches in the same stream apply first, so a group and its member contexts can travel together in any order |
| POST | `/contexts/{name}/associations` | `[{subject,label,object,weight,source?,paragraph?}]` → applied count (`paragraph` locates the fact within `source` and is ignored without one) |
| POST | `/contexts/{name}/recall` | `{cue, limit?, after?, since?, until?}` → `{total, matches, plan}` (`plan.contexts` = the contexts actually searched — trivially `[name]` here; the cross variants are where it earns its place). `since`/`until` (epoch seconds, half-open, over each source's `date ?? stored_at`) window the graph by assertion time on all four single-context graph lanes — only facts an in-window source attests, weights/citations from those attributions alone; refused on the cross variants |
| POST | `/recall` | `{contexts?:[name], groups?:[group], cue, limit?, after?}` → `{total, matches, plan}` — recall across several contexts at once (full names, and/or groups: each searches every context it reaches, nested children included, overlaps deduped; every match tagged with its `context`; past the limit the strongest \|weight\| survives, one scale across contexts; `plan.contexts` = the RESOLVED target list in effective order — groups expanded, your key's grants applied — so a target that came back empty is still visibly distinct from one your grant dropped) |
| POST | `/contexts/{name}/query` | `{subject?, label?, object?, subject_types?, object_types?, limit?, after?, since?, until?}` — each position a string or an array → `{total, matches, plan}` (`subject_types`/`object_types` filter by declared entity type, `is_a`-expanded, when the context has an installed schema — a filter, never an anchor, applied after the position pins and before paging, so `total`/`after` see the post-filter count; a schema-free context answers empty for a non-empty filter) |
| POST | `/query` | `{contexts?:[name], groups?:[group], subject?, label?, object?, subject_types?, object_types?, limit?, after?}` → `{total, matches, plan}` — query across several contexts at once, same contract as `POST /recall`; the type filters evaluate per target against that target's own schema, same paging order as the single-context route |
| POST | `/contexts/{name}/describe` | `{concept}` → label outline (counts per role) plus declared types (`types?`, absent both without an installed schema and for an untyped concept — the two are indistinguishable on purpose) / null |
| POST | `/contexts/{name}/explore` | `{origins, max_depth?, limit?, after?, since?, until?}` → `{total, matches:[{distance, path, association}]}` (hop cap 10, applied when omitted; truncation keeps the nearest) |
| POST | `/contexts/{name}/activate` | `{origins, decay?=0.5, limit?=20, since?, until?}` → `{total, matches:[{strength, path, association}]}` |
| POST | `/contexts/{name}/paths` | `{origins, targets, max_depth?, limit?=10}` → `{total, capped, matches:[{distance, path, strength, associations}]}` every simple path from an origin to a target, shortest first (hop cap 10, applied when omitted; limit capped at 100); within one length the largest weakest-link \|sum\| ranks first; `capped: true` means enumeration hit the server budget, so `total` is a lower bound |
| POST | `/contexts/{name}/resolve` | `{cue, dice_floor?, semantic_floor?, limit?}` → `[{name, score, tier, kind?, gloss?, types?}]` concept candidates (limit default/ceiling 1000; `types` rides the top 8 candidates only, and is absent both without an installed schema and for a candidate with no type assertion) |
| POST | `/contexts/{name}/resolve_label` | `{cue, dice_floor?, semantic_floor?, limit?}` → `[{name, score, tier, kind?, gloss?}]` relation candidates (limit default/ceiling 1000) |
| POST | `/contexts/{name}/resolve/explain` | `{cue, expected, dice_floor?, semantic_floor?, limit?}` → one verdict for "why didn't (or did) `expected` come back for `cue`", first that applies: `not_in_vocabulary` (nearest stored spellings attached — register an alias?) / `cue_resolved_exactly` (the cue IS another stored spelling; the exact tier answers alone) / `below_floor` (its actual score vs the floor in effect) / `below_cutoff` (rank, plus a `limit_to_reach` verified by rerunning the serve) / `semantic_not_run` / `semantic_below_floor` (gloss cosine vs the semantic floor, or which precondition failed) / `served` — same floors and limit as the resolve call being explained |
| POST | `/contexts/{name}/resolve_label/explain` | the same, for relation labels |
| POST | `/contexts/{name}/embeddings/refresh` | re-embed new/changed concept and label glosses (run after ingest) → `{embedded, total, glosses?:{embedded, total, skipped_over_limit?}, passages?:{embedded, total, skipped_over_limit?}}` — the per-lane breakdown appears when that lane ran, and `skipped_over_limit` counts passage rows past the server's vector-store ceiling (`TAGURU_PASSAGE_VECTOR_LIMIT`) — they stay un-embedded (BM25 still serves them) until the ceiling is raised |
| GET | `/contexts/{name}/embeddings` | `{provider_model, glosses?:{model, width, concepts, labels}, passages?:{model, width, rows}}` — the embedding identity: the provider configured now beside the (model, width) each vector sidecar was actually built with (`provider_model` null = embeddings off; a sidecar absent = nothing embedded yet; the two models disagreeing = refresh needed) |
| GET | `/contexts/{name}/labels` | `?limit=1000&after=label` → `{total, labels:[...]}` relation vocabulary (canonical only, keyset-paged by label) |
| GET | `/contexts/{name}/changes` | `?since=&limit=100` → `{events:[{seq, kind, ...}], next, more}` the polling change feed: content-change events after the opaque cursor `since` (kinds: `associations_added{count}`, `association_retracted{subject,label,object}`, `aliases_added{count}`, `aliases_removed{count}`, `source_stored{source}`, `source_retracted{source}`, `schema_updated{mode}` — events aggregate per write call, so a bulk import is one `associations_added` however many lines it carried, never one event per line; one call can still emit several KINDS, e.g. an import's per-source replace is `source_retracted` + `associations_added` + `source_stored`). Omit `since` to start tailing: an empty page whose `next` is the current position — the bootstrap after a full sync. `more: true` means events past `limit` are already waiting; poll again immediately. The feed is a bounded in-memory ring, deliberately not persisted history: a server restart, a delete-and-recreate, or falling further behind than the ring retains answers 410 `stale_cursor` — run a full resync, then tail again from a fresh cursor. Cursors are opaque and node-local (a replica mints its own) |
| GET/POST/DELETE | `/contexts/{name}/aliases` | `?limit=1000&after=concept:x\|label:x` → `{total, concepts:{alias:canonical}, labels:{...}}` (one page across both namespaces, concepts first; `after` = the last entry shown) / register `{concepts:{alias:canonical}, labels:{...}}` / withdraw `{concepts:[alias], labels:[...]}` |
| GET/POST | `/contexts/{name}/sources` | `?limit=1000&after=id` → `{total, sources:[...], entries:[{name, stored_at?, date?, tags?}]}` registered source ids with their metadata (keyset-paged; absent metadata omits its key) / `{passages:{source:text}, questions?:{source:[{paragraph, question}]}, sections?:{source:[{paragraph, section}]}, locators?:{source:[{paragraph, locator:{kind, value}}]}, tags?:{source:[tag]}, dates?:{source:epoch_secs}}` → `{stored, questions_stored, questions_dropped, sections_stored, sections_dropped, locators_stored, locators_dropped}` (a dropped question, section, or locator named a paragraph its text's blank-line split does not have; a locator is independent of `section` and does not extend to the next paragraph; `stored_at` is stamped by the server, `date` is the document's own time; storage replaces per source wholesale, metadata included) |
| POST | `/contexts/{name}/sources/lookup` | `{sources:[...]}` → `{passages, missing}` |
| POST | `/contexts/{name}/sources/search` | `{query, limit?=5, semantic_floor?, tags?, since?, until?}` → `{plan, hits:[{source, paragraph, score, text, lanes}]}` best PARAGRAPHS across passages (`paragraph` = its position in the source; `text` = that paragraph alone; `lanes.bm25`/`lanes.vector` = per-lane `{rank, score}`; `score` is rank-fused when the vector lane ran, raw BM25 otherwise; `semantic_floor` (0–1) overrides the vector lane's cosine floor for this call — context setting, then server default, otherwise — flooring only that lane; `tags` (any-of) and the half-open `[since, until)` epoch-second window over each source's `date ?? stored_at` pre-filter which sources may answer BEFORE the lanes run — a source with no tags, or with neither timestamp, never matches the respective filter kind). `plan.contexts` = one `{context, lanes:{bm25:{ran, reason?}, vector:{ran, reason?, floor?}}, filter?:{eligible_sources, total_sources}}` per context actually searched: whether each lane ran there and why not when it did not (the same wording `search/explain` uses), the vector lane's effective floor when it did, and — under a filter — how many sources were eligible of how many stored, so zero hits under a narrow filter no longer looks like "nothing matched" |
| POST | `/sources/search` | `{contexts?:[name], groups?:[group], query, limit?=5, semantic_floor?, tags?, since?, until?}` → the same `{plan, hits}` wrap, each hit tagged with its `context`, across several contexts at once (groups resolve as in `POST /recall`) — merged by per-context rank (every context's best hit first); `score` compares within one context only; the filter applies identically to every target; `plan.contexts` carries one entry per resolved target in effective order (per-context floors and filter counts included — a context's own `semantic_floor` setting shows here) |
| POST | `/contexts/{name}/sources/search/explain` | `{query, source, paragraph?, limit?=5, semantic_floor?, tags?, since?, until?}` → one verdict for "why didn't (or did) this source appear for this query" — same floor override and source filter as the search call being explained — first that applies: `not_stored` (never stored here, or retracted — the store keeps no tombstone history to tell which) / `paragraph_out_of_range` / `filtered_out` (the request's source filter excludes it — the search never considered it) / `no_query_terms` / `no_term_overlap` (the query's terms and the paragraph's terms side by side, AS STRINGS — the spelling-mismatch case: stored 酒蔵, searched 酒造) / `below_cutoff` (its rank, the cutoff score at your `limit`, and a `limit_to_reach` verified by rerunning the real serve computation, pool caps included — `limit_to_reach_reason: "unreachable"` names the rare case where no limit up to the raw row ceiling reaches it, distinct from the field simply being absent) / `served` — evidence carries per-term BM25 tf/df/idf/contribution (the very addends search summed) and the vector lane's cosine, or the reason that lane never ran. `paragraph` omitted picks the source's best showing |
| POST | `/contexts/{name}/citations` | `{source, paragraph}` → `{text, source, section, locator}` one verbatim paragraph by source and paragraph — the same paragraph `sources/search` would show at that paragraph (`section` is the label governing that paragraph, `null` outside every section the source has stored; `locator` is the typed citation locator — page/slide/sheet/table position, ADR 0007 §7 — named for that exact paragraph, `null` if none was stored, and unlike `section` it never extends to the next paragraph; `recall`/`query`/`explore`/`activate`/`unreachable_from` resolve the same section label and locator onto each attribution as `attributions[].section`/`attributions[].locator`) |
| POST | `/contexts/{name}/sources/retract` | `{source}`, `?dry_run=true` to preview → `{associations_touched, passage_removed}` withdraw that source's contributions (diff sync; the dry run reports the same shape with nothing written). Storage is append-only, so retraction alone leaves the withdrawn records and passage bytes on disk behind tombstones — full erasure is retract, then `POST /contexts/{name}/compact` |
| POST | `/contexts/{name}/associations/retract` | `{subject, label, object}` → `{retracted, attributions_removed}` — withdraw ONE association outright, every source's contribution to that edge (names resolve through aliases; `retracted: false` = no live edge, nothing changed; the edge row stays visible at weight 0 until compaction, and re-asserting later just works). For a fact that should never have been asserted; a fact that is merely CONTESTED wants a negative-weight assertion instead |
| POST | `/contexts/{name}/unreachable_from` | `{origins, limit?, after?}` → `{total, matches}` unreachable associations |
| GET | `/contexts/{name}/communities` | community detection on the live graph (JSON Lines body, not the JSON envelope): a header line `{taguru_communities:1, context, algorithm, revision, concept_count, edge_count, levels, communities}` — `revision` is the snapshot the analysis was cut at — then one line per community `{id, level, parent?, fingerprint, concept_count, members?:[{name, strength}], children?, top_associations?}`, leaves (level 0) first. Compute-heavy (heavy-ops gated); this is the derivation half `taguru communities` orchestrates — most callers want `communities/search` below instead |
| POST | `/contexts/{name}/communities/search` | `{query, limit?=5, semantic_floor?, derived?}` → `{derived, algorithm, stale, revision:{recorded_graph, current_graph}, plan, hits:[{community, score, text, paragraph, level?, parent?, concept_count?, members?:[{name, strength}], members_truncated?}]}` — global search over a community-summaries artifact built by `taguru communities` (an ordinary context, default `{name}::communities`; `derived` overrides). Ranked by the same two-lane passage search, so `plan`/`semantic_floor` behave exactly as in `sources/search`; `stale: true` = the source graph moved since derivation (summaries describe an older graph, served honestly). No artifact = a refusal naming the build command, never an empty result |
| POST | `/contexts/{name}/evidence` | `{origins, labels?, dice_floor?, semantic_floor?, resolve_limit?, activate_decay?, activate_limit?, text_fallback_query?, search_limit?=5, include_communities?=false, budget?:{max_items?=40, max_bytes?=65536, max_tokens?=4000}, rerank?}` → `{items, citations, budget, omitted, omitted_total, omitted_by_reason, plan}` opt-in evidence assembly (ADR 0006): runs the same resolve → query (only when `labels` pins the facets) → activate (always) → search_passages → cite_passage fan-out `retrieve` runs, plus an opt-in community-summary search, then normalizes every result into one ranked (reciprocal-rank fusion), deduplicated, budget-selected package — never the raw per-lane results `retrieve` hands back. `origins`/`labels` share `retrieve`'s own string-or-array contract; the passage/community lanes search `text_fallback_query` when given, otherwise `origins` joined with `"; "`. Each `items[]` entry carries `candidate_id`, `kind` (`association`/`passage`/`community`, an open string), `fused_rank`, `lane_ranks`, `citation_refs` (locators only — the text lives once in the top-level `citations`), `corroboration?` (every independent source an association fact traces to — never collapsed to a count), `contradicts?` (candidate_ids of every association this one disagrees with — a same-`(subject,label)` different-`object`, or opposite-signed, pair is admitted or omitted as one atomic group, never split), `bytes`/`estimated_tokens`, and exactly one of `association`/`passage`/`community` (the existing wire shape, embedded verbatim). `budget` is three independent hard ceilings — reaching any one stops admission; an over-budget candidate is skipped, not a call-ending refusal, so even `max_items: 0` answers 200 with an empty package and every candidate named under `omitted` (capped like `Issue` lists) or counted in the uncapped `omitted_total`/`omitted_by_reason`. `plan.lanes` is one `{ran, reason?, floor?}` per fan-out lane (`resolve`, `query`, `activate`, `passages`, `communities`, `citations` — the same shape `sources/search`'s own `plan.contexts[].lanes` uses); `include_communities: true` without a derived-communities artifact is a *degrade* here (`plan.lanes.communities.ran: false`), never the refusal `communities/search` itself gives — community evidence is one opt-in input among several, not the entire point of this call. `plan.selection` reports dedup/contradiction-group/diversity-tier accounting; `rerank?: {model?}` opts into reordering the already-deduplicated, near-duplicate-suppressed pool through an optional reranker (ADR 0006 §12, #307) — absent `rerank`, or no `TAGURU_RERANK_URL`/`TAGURU_RERANK_MODEL` configured on this server, keeps `plan.reranker = {configured, ran: false}` and selection fully deterministic, at no network or credential cost. When configured and requested, `plan.reranker = {configured: true, ran: true, model}` on success, or `{configured: true, ran: false, reason}` on any degrade — a model mismatch, an empty/singleton pool, an open circuit, a timeout, a provider error, or a non-permutation response — where `reason` is one of the machine-readable tokens `not_configured`/`model_mismatch`/`empty_pool`/`circuit_open`/`timeout`/`provider_error`/`invalid_permutation`. A reranker may only reorder the pool it is handed — it can never add, drop, or edit a candidate — and every degrade falls back to the same deterministic reciprocal-rank-fusion order, still answered 200, never a call-ending refusal. Candidate text reaches a configured reranker provider and nowhere else — never a log line, an error message, or a metric label. Never changes `retrieve`'s or any direct endpoint's own behavior |
| POST | `/contexts/{name}/vocabulary/audit` | `{dice_floor?=0.6, cosine_floor?=0.6}` → spelling/synonym fork candidates |
| POST | `/contexts/{name}/consolidation/audit` | `{checks:["merge"\|"contradiction"\|"staleness", ...], limit?=100, evidence_cap?=20, dice_floor?=0.6, cosine_floor?=0.6, floor_secs?=0}` → `{detector, merge?:{total, candidates:[{a, b, tier, name_score, types_a?, types_b?, overlap, shared/only_a/only_b (+exact totals), fingerprint}]}, contradiction?:{total, candidates:[{kind:"objects", subject, label, functional_tendency, objects:[{object, weight, count, sources, latest?}]} \| {kind:"contested", ..., supporting_sources, disputing_sources}]}, staleness?:{total, undatable, candidates:[{subject, label, object, latest, neighborhood_latest, gap, sources}]}}` — merge/contradiction/staleness CANDIDATES for review, never verdicts (ADR 0012): `checks` is required (each section is a full-graph pass; over 1,000 entries refuses `over_limit` like every list-shaped input), `objects[].sources` names only sources whose own accumulated sum is nonzero — one that cancelled its own assertion to exactly zero attests nothing, the same rule `contested` applies to its sides — every candidate carries a `fingerprint` over its own evidence — judge once per fingerprint, reuse the judgment until it moves — and application is ordinary writes (alias, retract, negative weight, re-import) |
| POST | `/contexts/{name}/drift/audit` | `{unsourced_floor?, limit?, after?, include_twins?=false, dice_floor?=0.6, cosine_floor?=0.6}` → `{total, unsourced:[{unsourced_weight, unsourced_count, association}], dead_concept_aliases, dead_label_aliases, twins?}` graph-vs-archive drift: edges carrying weight no named source explains (worst magnitude first, paged like `unreachable_from`), aliases whose canonical no longer lives, and — only when `include_twins` — the same fork candidates `vocabulary/audit` finds |
| GET/PUT | `/contexts/{name}/schema` | GET → the installed schema document `{schema:1, mode, closed_labels, types, relations}` / 404 `no_schema` (distinct from `no_context`) when none is installed; PUT installs (or replaces) it, answering the document as installed — refused (400) for a document whose `relations` declare the reserved `schema:type` label, or when an already-persisted label alias resolves to it (rename the alias first). Installing changes what `strict` refuses from this point on; dry-run with `schema/validate` first |
| POST | `/contexts/{name}/schema/audit` | `{limit?, after?}` (body optional) → `{total, violations:[{association, issues}], untyped_concepts:{total, names}, undeclared_types:{total, names}, unknown_labels:{total, names}, reserved_alias_conflicts:{total, aliases}}` — judges every LIVE association against the installed document, the pre-existing violations `strict` can never surface on its own; candidates for review, never auto-fixed; only `violations` pages (`total` constant across pages, same cursor as recall/query); 404 `no_schema` without an installed document |
| POST | `/contexts/{name}/schema/validate` | `{document, limit?, after?}` → the same audit shape over the PROPOSED document, validated and evaluated without ever being persisted — the pre-flight before a `strict` flip; works identically with or without an installed schema |
| GET | `/contexts/{name}/export` | the context as an import batch stream (JSON Lines body, not the JSON envelope) — one batch per source, create block first, aliases last; `POST /import` (or `taguru import`) restores it, per-source retract-then-apply, answering `{batches: [...]}` in stream order (`taguru_group` records ride the same stream, restore after every batch as whole-record replaces, and answer under `groups: [...]`) |
| POST | `/contexts/{name}/promote` | `{into, sources, audit?=true}`, `?dry_run=true` to preview → `{batches: [...], aliases_dropped, audit?, audit_skipped?}` graph-path memory promotion (ADR 0018, docs/promotion.html): move the named source ids from this (scratch) context into the established context `into` WITHOUT re-extraction — the export/import round trip in one call. Each source moves whole (passage, `date`, tags, only its own share of every edge's weight; aliases ride exactly when their canonical is live in the promoted slice, `aliases_dropped` counts the rest), source ids survive (promoted citations still name the session), and applying is per-source retract-then-apply — re-promoting is idempotent. `into` must exist (never created here; write grant checked like `/import`'s body contexts) and its own schema judges the incoming batches; a named source missing from the scratch refuses the WHOLE request path-addressed, `nothing_written`; over 1,000 `sources` refuses `over_limit` before any per-id validation, like every list-shaped input. After a real apply, `audit` carries `consolidation/audit`'s full default report on `into` (all three checks) — candidates to judge, never applied; `audit_skipped` names why it could not run (the batches are durable by then). The dry run previews the same `batches` shape, writes nothing, audits nothing. Retiring the promoted scratch stays an explicit `sources/retract` |
| POST | `/contexts/{name}/compact` | rebuild the image without dead records, and rewrite the passage log without retracted sources' text (admin; the context's requests wait out the rebuild) → `{bytes_before, bytes_after, dead_edges, aliases_dropped, passages_compacted, image_persisted}` — `image_persisted: false` means the rebuild itself succeeded (the numbers above are real) but the smaller image has not yet reached disk (e.g. a full disk); the next flush tick retries it |
| POST | `/flush` | force every context's unflushed state to disk now, ahead of the periodic flusher → the flushed context names; admin, server-wide (refused for a context-scoped key — the answer names every flushed context, grant or no grant) |
| POST | `/mcp` | the MCP Streamable HTTP transport, stateless profile: each POSTed JSON-RPC message answered as plain `application/json` (no SSE stream, no session id — the spec's stateless profile). Tool calls dispatch in process onto the routes above under the outer request's own auth, scope, deadline, and body cap — one client request, one budget, one log line; `initialize` hands out the same manual `GET /protocol` serves |
| POST | `/maintenance/compact` | `?min_dead_ratio=0.0` (default; any dead weight at all) → sweep every context whose live dead ratio strictly exceeds it, worst ratio first, each rebuilt like `/contexts/{name}/compact`; admin, server-wide (refused for a context-scoped key, like `/flush`) — closes the server to ordinary traffic for the sweep (`/health` answers `503 maintenance` meanwhile, distinct from an actual fault) and reopens when it ends or the deadline cuts it short → `{contexts:[{name, bytes_before, bytes_after, dead_edges, aliases_dropped, passages_compacted, image_persisted}], skipped:[{name, error}], deadline_exceeded}` — `skipped` names every candidate the sweep selected but could not compact at all (a load failure, a quota refusal); a sweep that fails on every candidate no longer reads as a clean, empty run |

Reading `POST /contexts/{name}/evidence`'s `plan`: `plan.selection.dedup_dropped`
counts exact-key duplicates `fuse` folded together (the same fact
surfacing from both `query` and `activate`, say) before selection ever
ran; `plan.selection.contradiction_groups` counts how many live
disagreements existed in the pool, admitted or omitted whole either
way (never split — an item's own `contradicts` names the rest of its
group); `plan.selection.diversity_tier_width` is the fixed
`max(1, max_items / 4)` window admission round-robins sources within,
informational only. `plan.reranker.configured`/`.ran` being `false`
means exactly what the deterministic pipeline already promises: no
provider ran, order came from reciprocal-rank fusion alone, and the
package is still complete — never a degraded or partial answer. A
non-`false` `.ran` never changes membership, budget accounting, or any
other invariant this section describes — reordering the input to an
already invariant-preserving admission process cannot itself violate
an invariant (ADR 0006 §12).

## Auth

- If the server sets `TAGURU_API_TOKEN`, every request except
  `GET /health`, `GET /live`, `GET /metrics`, and `GET /version` needs
  `Authorization: Bearer <token>`; missing or wrong → `401` in the
  error shape below.
- The MCP bridge (taguru-mcp) reads its own `TAGURU_API_TOKEN` and
  attaches it to every request — when the server turns auth on, set
  the same value on the bridge.
- Unset = auth disabled (dev mode; never expose beyond localhost).
- Setting `TAGURU_PUBLIC_URL` additionally mounts OAuth for remote
  MCP clients (dynamic client registration, authorization code with
  PKCE) beside the API: `GET /.well-known/oauth-protected-resource`
  (also probed path-inserted at
  `/.well-known/oauth-protected-resource/mcp`),
  `GET /.well-known/oauth-authorization-server`,
  `POST /oauth/register`, `GET/POST /oauth/authorize` (the consent
  page and its approval), and `POST /oauth/token` (code exchange and
  refresh). These six paths join the auth-exempt list (a client must
  reach them before it has a token), and a token minted here rides
  `Authorization: Bearer` but opens `POST /mcp` ONLY — every other
  route still requires `TAGURU_API_TOKEN` (or a keyring key); an
  OAuth token presented elsewhere is a plain `401`.
- Keys may carry a scope (`TAGURU_KEY_SCOPES`): a role — read (the
  retrieval loop) ⊂ write (+ the ingest loop, group create/update) ⊂
  admin (+ context and group deletion and renaming, `/import`,
  `/flush`, `/maintenance/compact`) — and
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
`no_paragraph` / `no_group` / `no_schema` (404: the context exists but
has no schema document installed — distinct from `no_context`) /
`unknown_path` / `method_not_allowed` / `timeout` /
`already_exists` / `conflict` / `payload_too_large` / `rate_limited` /
`internal` / `embeddings_unconfigured` / `embeddings_failed` /
`overloaded` (shed at the global in-flight ceiling or the shared
heavy-operation ceiling for vocabulary audits/context compactions;
wait `Retry-After`) /
`unhealthy` (the write path is degraded) / `maintenance` (a
`POST /maintenance/compact` sweep is running — wait `Retry-After` and
retry) / `storage_full` / `read_only_replica` (403: this server is a
read replica — do NOT retry here; send the write to the writer the
message names) / `shard_unreachable` (502 from a `taguru router`: a shard
this request needs did not answer — retry once the shard or its load
balancer does).

**Rejected `add_associations`, `store_passages`, and `import` calls carry
structured detail** (additive fields, present only where they apply —
absent entirely from every other error and, outside the one exception
below, from every success):
`issues` (up to 20, with the true count named in `error` when there are
more) is an array of `{"path", "kind", "expected", "actual"}` — `path`
locates the offending field exactly (`associations[1].weight`,
`passages['doc'].questions[2].question`, `batches[0].concepts['alias']`),
`kind` is one of `missing` / `type` / `empty` / `too_long` / `range` /
`over_limit` / `unknown_reference` / `conflict` / `domain` — a context
schema's `domain`/`range` constraint on a relation (ADR 0009 §8.1)
adds `domain` and reuses the wire string `"range"` for the object side,
so a `range` issue on `associations[1].subject`/`.object` means the
schema's `domain`/`range` constraint, never the numeric-weight `range`
above; `path` always disambiguates (`.weight` for the numeric case,
`.subject`/`.object` for the schema case). `expected`/`actual`
describe the mismatch in the same words a human reads in `error`.
`integrity` says what a rejection actually left behind:
`"nothing_written"` (the whole call, or the whole rejected
`import` batch, wrote nothing) or `"durable_prefix"` (a multi-batch
`import` stream where earlier batches already landed — `durable_batches`
names exactly how many; never implies any part of the REJECTED batch
itself landed, since every batch is whole-or-none). `retryable_after_correction: true` marks a
rejection a corrected, COMPLETE resend can resolve (a validation issue,
a batch over its cap, a predicted alias conflict) — absent for a
rejection resending the same content cannot fix (auth, quota,
timeout). Over MCP, this same JSON rides twice on a tool error: as
prose inside `content[0].text` (unchanged from before this existed) and
again, parsed, as `structuredContent` — read either, but branch on
`code`/`kind`, never on wording. The correction discipline for all
three tools: preserve every association/alias/question/passage from
the attempted write, correct only the listed paths, resend the
COMPLETE source write or batch (never a partial resubmission of just
the fixed items), add no fact that was not already there, and if
correction still fails, leave the source unmodified and report the
failure rather than retrying blindly. The MCP tool-call layer's own
argument-shape rejections (a missing required argument, a wrong-typed
JSON-RPC argument) are a distinct, EARLIER surface — plain prose, no
`issues` — since nothing reached the server to build them from; schema
validation on the client/transport side and this server-side contract
are complementary, not redundant. The server never retries the
extracting LLM itself here — MCP has no model call to retry — so
correction and resubmission are entirely the calling host's
responsibility.

**The one success-envelope exception:** `import` against a context
whose schema mode is `warn` (ADR 0009 §7.1) answers `200` with the
batch applied, but the same `issues` array rides the success envelope
alongside `result` — `domain`/`range` violations this write raised,
identical `Issue` values to what a `strict` context would have refused
with. `warn` never refuses a write; it reports. Each affected batch's
own `ImportOutcome.schema_violations` carries the true count, surviving
`issues`' own 20-item cap. `off` mode and a schema-free context both
omit `issues` from a success exactly as before — this exception fires
only under `warn`.

- `401` auth (above). `404` unknown context or group. `409` duplicate
  create / alias conflict / a `POST /maintenance/compact` overlapping
  one already running.
- `507` context full (`storage_full`) — the refused call was not
  applied. Two ways here: the context hit the library's own capacity
  cap (further knowledge goes to a new context), or it reached a
  per-context storage quota the operator declared
  (`TAGURU_CONTEXT_QUOTAS`) — the message says which. One scope note:
  a multi-batch `/import` that hits 507 partway is a resumable
  prefix, not a no-op — its message reports the batches before the
  stop as landed durably, and a batch refused mid-apply may already
  have retracted the source it was replacing; re-sending the stream
  is exact either way (each batch replaces its own source). At a
  quota, retractions, alias removals, `DELETE`, and compaction still
  work: shrink the context (or have the operator raise its quota)
  and retry; do not blindly re-send the refused write. "Shrink" means
  those explicit operations — a replacement that carries new content
  counts as growth even when it would net smaller, because its true
  size is only knowable after it applies. To slim a source at the
  ceiling: retract it first (over `/import`, a header-only batch —
  just `taguru_batch`/`context`/`source` — is exactly that
  retraction), then re-send the smaller version.
- `501` `/embeddings/refresh` without a provider configured
  (server-side TAGURU_EMBED_*). `502` embedding provider failure
  (refresh, or the semantic fallback inside resolve) — retry later.
- `400`: association batch over 10,000 per request (nothing applied —
  split and resend; alias batches and removals share the same cap) /
  list-shaped read input over 1,000 items (origins, query terms,
  `sources/lookup` sources, cross-search `contexts` and `groups`,
  `promote` sources, `consolidation/audit` checks —
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
- Behind a `taguru router` (sharded deployments), the
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
  (additive), absent optional fields are omitted rather than null,
  and every enum-like field must be treated as open — an unrecognized
  value must not crash decoding.
- **Within one contract version, nothing breaks** (ADR 0005 §7). A
  `server` minor bump may still add things, but a break — a response
  container shape changing, a field being removed or renamed, a
  pagination envelope's shape changing, and the rest of ADR 0005 §4's
  table — requires the matching `http_contract` (or `mcp_contract`)
  bump below, landing in the same PR as a CHANGELOG "Changed" entry
  and a migration note.
- `GET /version` (auth-exempt, always `200` even while `/health`
  reports degraded) answers which wire-shape versions this server
  speaks, discovery only — not negotiation, so there is nothing to
  request beyond what this server already serves:
  ```json
  {
    "server": "0.6.0",
    "http_contract": {"current": 1, "supported": [1]},
    "mcp_contract": {"current": 1, "supported": [1]},
    "mcp_protocol": {"supported": ["2024-11-05", "2025-03-26", "2025-06-18"]},
    "batch_formats": [1],
    "image_formats": [1, 2, 3, 4, 5, 6],
    "communities_formats": [1]
  }
  ```
  `http_contract` covers every enveloped and non-enveloped HTTP
  request/response/error shape; `mcp_contract` covers only what MCP
  itself owns (the tool name/`inputSchema` table, `retrieve`'s
  composed output, the `isError`/`structuredContent` convention, and
  the JSON-RPC error vocabulary) — a pure HTTP-shape change bumps
  `http_contract` alone, even though it is also visible over MCP. The
  same facts are folded into this document's own live trailer (the
  `## This server` section `GET /protocol` and every MCP
  `initialize`'s `instructions` carry), so an MCP client learns them
  without a second connection.
- Both official SDKs declare their own supported `http_contract`
  range and check it against `GET /version` before their first real
  request, raising a dedicated error with a concrete upgrade remedy on
  a genuine mismatch — never on a compatible patch/minor difference,
  and never on an absent or unreadable `/version` (a server predating
  this endpoint is treated as speaking `http_contract: 1`, not refused
  outright).
- The batch format (`taguru_batch: 1`) and the image format are
  versioned independently of the API: old batch files stay readable,
  and images migrate forward on load. Rolling a server BINARY back
  past an image-format bump needs the data rolled back with it — the
  release notes flag format bumps.
