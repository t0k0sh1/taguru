# taguru-code evaluation protocol

Four axes decide whether the taguru-code approach graduates from
spike to product. Axis 1 and 3 are automated; axis 2 and 4 are a
recorded manual protocol. The comparison baseline is **graphify**
(the incumbent knowledge-graph skill) and **bare Grep/Glob**.

## Axis 1 — accuracy (automated, CI-able)

Ground truth is derivable because facts are deterministic AST output:

```bash
taguru-code evalset --out eval.jsonl --sample 200
taguru-code eval --eval eval.jsonl --thresholds thresholds.json  # exit 3 on regression
```

Case kinds: bare tail cues (`parse_batch`), qualified cues
(`Context::resolve`), path cues (`model.rs`). Metrics: `hit1_rate`,
`hit10_rate`, `line_drift` (a hit whose line range disagrees with a
fresh parse — must stay 0 right after a sync).

Suggested thresholds: `{"hit1_rate": 0.85, "hit10_rate": 0.95, "line_drift": 0}`.
Note the honest caveat: ambiguous tails (`new`, `tests`, `load`)
legitimately miss hit@10 — the qualified-cue cases exist to show the
disambiguation path works.

## Axis 2 — agent task evaluation (vs graphify, vs Grep)

20-30 location questions about THIS repository, asked of a coding
agent (Claude Code) three ways — same questions, fresh session each:

- A: taguru-code + the taguru-code-nav skill
- B: graphify (`graphify query`)
- C: bare Grep/Glob (no skill)

Question shapes (draw from real work, not from the eval.jsonl):
"where is X defined", "what file implements Y", "what does module Z
contain", "find the definition of the method A::B", plus 2-3 with
deliberately misspelled names.

Record per question, per method:

| metric | how |
|---|---|
| correct | did the agent land on the right file:line? |
| tool calls | count of tool invocations to the answer |
| tokens | session token usage delta |
| wall time | seconds to the answer |

**Graduation bar: A beats B on correctness AND on tokens; A's tool
calls ≤ C's.** If A loses either, the approach goes back for rework,
not release.

## Axis 3 — freshness (automated)

Integration test: full sync → edit+commit → incremental sync → find
reflects new lines; rename+delete+commit → sync → old names gone,
new names found. Lives in the taguru-code test suite.

## Axis 4 — ingestion cost

Record for this repository (release build):

- full `sync` wall time and resulting `.taguru/` size
- incremental sync after a one-file commit (target: seconds)
- graphify's full build time and any LLM cost on the same repo
  (taguru-code's LLM cost is zero by construction)

## Reporting

One results file per run of the protocol:
date, HEAD sha, binary version, the four axes' numbers, and the
verdict against the graduation bar.
