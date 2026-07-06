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
   empty reads, writes, last-read/write unix seconds).
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
3. **Outline, then narrow**: `describe` a hub concept first (which
   labels, how many, per role), then `query` just the facets you need
   (`"label": ["住所","職歴"]`). Don't pull whole profiles.
4. **Expand and rank**: `activate` spreads from origins (strongest
   first, `path` shows the route; strength is an ordering within one
   call — never compare across calls). `explore` walks structure
   exhaustively with hop-distance annotations.
5. **Answer from the originals**: feed the attributions' source ids to
   `POST /contexts/{name}/sources/lookup` and ground your wording in
   the passages. Reflect negative weights as negation and attribution
   counts as strength of support.
6. **Switch to the text lane**: knowledge that never fit a triple
   (procedural detail, conditions, discourse) was never in the graph.
   When graph results can't compose the answer, run
   `POST /contexts/{name}/sources/search` (full-text over passages).
   Graph first, text as the safety net.

## Ingest loop

1. Decompose the document into (subject, label, object, weight).
   - **Check before mint**: `resolve` / `resolve_label` before coining
     any spelling; reuse what exists. `GET /contexts/{name}/labels`
     lists the relation vocabulary.
   - Don't re-assert paraphrases within one document (inflates
     weight). DO re-assert across documents (that's corroboration).
   - Negation: positive label, negative weight.
   - Make implicit membership explicit (whose 杜氏 is 高瀬? — add the
     edge).
2. `POST /contexts/{name}/associations` in batches — one document per
   request, a `source` on every element.
3. Register originals: `POST /contexts/{name}/sources` (source id →
   passage). Keep passages paragraph-to-rule sized — BM25 length
   normalization buries details inside section-sized passages.
4. Audit reachability: `POST /contexts/{name}/unreachable_from` with
   the document's main entities. Non-empty = membership edges are
   missing. If embeddings are configured, finish with
   `POST /contexts/{name}/embeddings/refresh` (diff-only, idempotent;
   unnecessary when the server runs `TAGURU_EMBED_AUTO`).
5. At milestones, `POST /contexts/{name}/vocabulary/audit` lists fork
   candidates (lexical twins = spelling drift, semantic twins =
   synonym forks). Candidates, not verdicts. Same referent → pick one
   canonical, point an alias at it, use the canonical from then on.
   (Forks that already accumulated facts cannot be merged — that's
   rebuild territory.)
6. When live wording misses, register alternate spellings:
   `POST /contexts/{name}/aliases`. Aliases are entry-only; results
   always return the canonical. An alias cannot join two existing
   concepts (that would be a merge — rebuild territory).
7. **Document updated? Sync the diff**:
   `POST /contexts/{name}/sources/retract` withdraws the old version's
   contributions (weights, attributions, passage), then ingest the new
   version normally. Concepts and edges remain; only weights come
   down.

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
| GET | `/contexts` | `?limit=1000&after=name` → `{total, contexts:[{name, description, pinned, loaded, dice_floor, semantic_floor, stats, usage}]}` (keyset paging by name) |
| GET | `/contexts/{name}` | one directory row / 404 |
| PUT | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → create |
| PATCH | `/contexts/{name}` | `{description?, pinned?, dice_floor?, semantic_floor?}` → update metadata |
| DELETE | `/contexts/{name}` | delete, files included |
| POST | `/contexts/{name}/associations` | `[{subject,label,object,weight,source?}]` → applied count |
| POST | `/contexts/{name}/recall` | `{cue, limit?}` → `{total, matches}` |
| POST | `/contexts/{name}/query` | `{subject?, label?, object?, limit?}` — each position a string or an array → `{total, matches}` |
| POST | `/contexts/{name}/describe` | `{concept}` → label outline (counts per role) / null |
| POST | `/contexts/{name}/explore` | `{origins, max_depth?, limit?}` → `{total, matches:[{distance, path, association}]}` (hop cap 10, applied when omitted; truncation keeps the nearest) |
| POST | `/contexts/{name}/activate` | `{origins, decay?=0.5, limit?=20}` → `[{strength, path, association}]` |
| POST | `/contexts/{name}/resolve` | `{cue, dice_floor?, semantic_floor?}` → `[{name, score, tier}]` concept candidates |
| POST | `/contexts/{name}/resolve_label` | `{cue, dice_floor?, semantic_floor?}` → `[{name, score, tier}]` relation candidates |
| POST | `/contexts/{name}/embeddings/refresh` | re-embed new/changed concept and label glosses (run after ingest) |
| GET | `/contexts/{name}/labels` | relation vocabulary (canonical only) |
| GET/POST | `/contexts/{name}/aliases` | export / `{concepts:{alias:canonical}, labels:{...}}` |
| GET/POST | `/contexts/{name}/sources` | registered source list / `{passages:{source:text}}` |
| POST | `/contexts/{name}/sources/lookup` | `{sources:[...]}` → `{passages, missing}` |
| POST | `/contexts/{name}/sources/search` | `{query, limit?=5}` → `[{source, score, text}]` full-text over passages |
| POST | `/contexts/{name}/sources/retract` | `{source}` → withdraw that source's contributions (diff sync) |
| POST | `/contexts/{name}/unreachable_from` | `{origins, limit?}` → `{total, matches}` unreachable associations |
| POST | `/contexts/{name}/vocabulary/audit` | `{dice_floor?=0.6, cosine_floor?=0.6}` → spelling/synonym fork candidates |

## Auth

- If the server sets `TAGURU_API_TOKEN`, every request except
  `/health` and `/metrics` needs `Authorization: Bearer <token>`;
  missing or wrong → `401` in the error shape below.
- The MCP bridge (taguru-mcp) reads its own `TAGURU_API_TOKEN` and
  attaches it to every request — when the server turns auth on, set
  the same value on the bridge.
- Unset = auth disabled (dev mode; never expose beyond localhost).

## Errors and limits

- `401` auth (above). `404` unknown context. `409` duplicate create /
  alias conflict.
- `507` context full (`ContextFull`) — the write was NOT applied;
  further knowledge goes to a new context.
- `501` `/embeddings/refresh` without a provider configured
  (server-side TAGURU_EMBED_*). `502` embedding provider failure
  (refresh, or the semantic fallback inside resolve) — retry later.
- `400`: association batch over 10,000 per request (nothing applied —
  split and resend) / weight not finite or |weight| > 1,000,000
  (whole batch refused) / name too long (subject, label, object,
  source, alias ≤ 1024 bytes — names are headings, not bodies:
  passages go to sources, long knowledge gets decomposed; context
  name ≤ 64, description ≤ 4096). `408` timeout (default 30 s —
  narrow the query and retry). `413` body over the cap (default
  8 MiB; this one answers in plain text, not the JSON error shape).
  `429` this key is over its request budget — wait the `Retry-After`
  seconds and continue; prefer batching writes over rapid-fire calls.
- Off-axis errors answer in the same shape: unknown path `404`, right
  path wrong verb `405`, broken JSON `400`, wrong Content-Type `415`,
  well-formed but mistyped JSON `422`.
- recall / query / explore / unreachable_from default `limit` to 100.
  `total` above the returned count = truncation
  (recall/query/unreachable_from keep the strongest |weight|, explore
  keeps the nearest hops). Narrow or raise `limit` — capped at 1000
  everywhere.
- A write that returned 200 is durable via the WAL (it survives a
  crash and replays on restart). Only when the server runs
  `TAGURU_WAL=0` can writes inside the flush interval (default 5 s)
  be lost.
