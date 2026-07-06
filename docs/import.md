# Offline batch import — the `taguru import` file contract

`taguru import` applies batch files straight to the data directory,
without (and never beside) a running server — the directory lock
refuses to run while one holds the directory, and vice versa (an
advisory flock, dependable on local disks; the README notes the
network-filesystem caveats). It is
the bulk path: initial loads, migrations between instances, replaying
the output of an extraction pipeline. Live, incremental writing stays
on the HTTP API / MCP tools.

```sh
taguru import batch/                  # every *.jsonl under batch/, name order
taguru import --dry-run batch/        # validate and report; touch nothing
taguru import --config prod.env a.jsonl b.jsonl
```

Exit codes: `0` everything applied · `1` something was refused or
failed (details on stderr) · `2` usage error.

## One file = one source's complete truth

Applying a file means: **retract the source, then apply the file**.
That single rule gives the operations their production properties:

- **Idempotent** — importing the same file twice ends in the same
  state; weights never double.
- **Revisable** — a corrected file for the same source replaces the
  old facts entirely; it is the same differential sync
  (`retract_source` → re-ingest) agents perform live.
- **Retryable** — a file that failed partway (capacity, disk) is
  fixed and re-imported; the retraction makes the retry exact.

This is also why association lines may **not** carry their own
`source` field (they are refused): a source the header does not name
would survive the retraction and double on every re-import. The
header's source is stamped on every association in the file.

A file with a header and no op lines is a **pure retraction**: "this
source now asserts nothing."

The retract-then-apply order has one sharp edge: when the apply stage
fails *persistently* — capacity is the realistic case — the source
sits empty (old facts retracted, new ones refused) until a re-import
succeeds. Retraction frees a source's attributions but never
un-interns vocabulary: a concept or label, once minted, holds its id
for the context's lifetime, so a pipeline that keeps re-importing
revised files under churning names grows the context monotonically.
Watch the counts with `taguru inspect`, size headroom with `taguru
estimate`, and treat a context that has crept near its caps to a
rebuild rather than another revision.

## File format

JSON Lines, UTF-8. Blank lines are ignored. The first non-blank line
is the header; every later line is one operation.

```jsonl
{"taguru_batch": 1, "context": "sake", "source": "docs/aomine.md", "create": {"description": "酒蔵の知識"}}
{"passage": "青嶺酒造は1907年創業。杜氏は高瀬。"}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0}
{"alias": "Aomine Brewery", "canonical": "青嶺酒造", "kind": "concept"}
{"alias": "設立年", "canonical": "創業年", "kind": "label"}
```

**Header** — `taguru_batch` must be `1` (the one version this build
reads). `context` and `source` are required. `create` is optional and
carries the same fields as `PUT /contexts/{name}` (`description`,
`pinned`, `dice_floor`, `semantic_floor`); it is used only when the
context does not exist yet — importing into an absent context without
a create block is refused.

**Association** — `subject` / `label` / `object` / `weight`, exactly
the fields of the associations endpoint minus `source` (see above).

**Alias** — `alias` / `canonical` / `kind`, where `kind` is
`"concept"` or `"label"`. Aliases are context-level, not
source-scoped: a re-import re-registers them (a no-op for an
unchanged pair), and retraction never removes them. Registering an
alias whose canonical is only introduced by this file's associations
works — associations apply first. Re-pointing an existing alias at a
*different* canonical is a conflict and fails the file — an import
must not silently re-wire a vocabulary. Be aware that an alias, once
registered, is permanent for the context's lifetime (nothing removes
one, the API included): the ways out of a conflict are keeping the
old canonical, choosing a different alias spelling in the revised
file, or rebuilding the context.

**Passage** — the source's original text, at most one per file,
stored behind the same source id (`sources/lookup` and
`sources/search` serve it).

Unknown fields and unrecognized line shapes are refused with the line
number.

## Caps (the API's, enforced per line)

| What | Cap |
|---|---|
| context name | 64 bytes |
| source id, subject/label/object, alias/canonical | 1 024 bytes |
| create.description | 4 096 bytes |
| weight | finite, \|w\| ≤ 1e6 |
| passage | 8 MiB (the HTTP default body cap) |
| any one line | 16 MiB |

Files themselves have no op-count cap; applies are chunked at the
API's batch size (10 000) internally, and long runs flush
periodically so no context's WAL approaches `TAGURU_WAL_MAX_BYTES`.

## Validation, then application

Every file must parse — and no two files may claim the same
(context, source) — before **anything** applies; a malformed line
refuses the whole run with its line number and nothing is written
(`--dry-run` stops there by design). Failures that only the apply
stage can discover (capacity, disk) are reported per file while the
remaining files continue: files are independent by construction (one
source each), and a partially applied file heals on re-import.

The writes go through the same registry as the server's — WAL-staged
and fsynced, budget-enforced, flushed at the end — so `taguru
inspect` and a subsequent boot see exactly what a live ingest would
have produced. Import counts as writes in each context's usage stats.

Validating everything first has a cost: every parsed file stays in
memory until the run applies, so a run's footprint tracks the
**total** size of its files, not the largest one. Feed a
million-document migration as several invocations (one directory
slice each) — slicing costs nothing, since imports are idempotent
and every file carries its own source.

## Embeddings

With `TAGURU_EMBED_URL`/`TAGURU_EMBED_MODEL` set (or given via
`--config`), every touched context re-embeds its changed glosses at
the end of the run — the same idempotent refresh the server's
`TAGURU_EMBED_AUTO` performs. `--no-embed` skips it. A refresh
failure does not undo anything: the graph is imported and durable,
and the run exits 1 naming the recovery
(`POST /contexts/{name}/embeddings/refresh` on the running server).

## Producing batch files

Taguru deliberately does not extract facts from prose — that is the
reading LLM's job (the server never holds model credentials). A
typical pipeline reads documents, has a model emit associations in
the `/protocol` discipline, and writes one batch file per document
with the document id as the source. The Converse-loop pattern in
[bedrock.md](bedrock.md) and the tool vocabulary in
[llm-protocol.md](llm-protocol.md) are the two halves of that
producer; the batch file is just its offline serialization.
