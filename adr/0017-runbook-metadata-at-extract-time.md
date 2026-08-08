# 0017. Promotion-runbook metadata at extract time

- **Status**: Accepted
- **Date**: 2026-08-09
- **Issue**: #466 (S1)
- **Related**: #465 (the runbook whose conventions these flags encode),
  ADR 0011 (why `date` is load-bearing), ADR 0005 (the batch contract
  the metadata rides), issue #167 (the passage-line metadata fields)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How `taguru extract` writes the promotion runbook's source conventions
— session source id, assertion date, topic tags — into the batches it
emits (#466 S1). Out of scope: bundling the promotion sequence itself
(a verb or MCP tool, #466's remaining splits), and the SDK producers.

## 2. Context

The 2026-08-08 runbook rehearsal (#466's gate record) found the single
most mechanical step of every promotion: extract knows only document
paths, so the operator rewrites each emitted batch's `source` to the
runbook's `session:{agent}:{id}`, and hand-adds the `date` and `tags`
the scratch conventions require — every time, purely mechanically,
with a text editor against a generated file. The import wire format
has carried all three fields since #167; extract simply never had a
way to be told them.

## 3. Decision

**`--source-id ID`, `--date WHEN`, and `--tag TAG` (repeatable) bake
the runbook's conventions into the written batch. All three are
manifest computation inputs and none is a checkpoint input.**

1. **`--source-id` replaces the header's source**: verbatim for a
   single document; with several, each document gets `ID/{file stem}`
   — the runbook's own `/{doc}` convention, made automatic because
   import's retract-then-apply is per source id, and one id covering
   two documents would silently fold them. Two documents whose stems
   collide fail the second with the reason, before any model call.
   The manifest stays keyed by the document PATH — the path names the
   input; this names the output.
2. **`--date` and `--tag` ride the passage line**, exactly where the
   wire format carries source metadata. Requiring the passage is
   therefore enforced as a usage error against `--no-passage`, not a
   silent drop — an associations-only source stores no metadata and is
   invisible to every windowed read (docs/promotion.html's own
   warning). `--date` accepts epoch seconds (the wire unit) or
   `YYYY-MM-DD` (what a session note records — that day's UTC
   midnight, round-tripped through the rendering direction so a
   non-existent date is refused rather than normalized).
3. **Manifest inputs, not checkpoint inputs**: all three are baked
   into the emitted file, so a change must rewrite the batch (the
   `context`/`description` precedent — a skip would leave the old id
   or date in place). But none of them reaches the prompt — the model
   is still shown the document path — so cached chunk answers stay
   reusable across a metadata change: the rewrite costs zero model
   calls for checkpointed units. The fingerprint records the EFFECTIVE
   written source (suffix included), so revising the suffix scheme
   re-extracts too; `""`/`0`/`[]` are the off values, keeping pre-S1
   manifest entries matching default runs.
4. **Flag-only, no env counterparts**: a session id and its date are
   per-invocation values, not deployment settings — the
   `--context`/`--description` precedent, so `KNOWN_KEYS` and the
   config dialect are untouched.

## 4. Consequences

- The runbook's step 2 loses its hand-editing: extract now emits an
  import-ready promotion batch directly, and the `#496` controls
  (`--vocabulary` for the resolve-first rule, `--coverage` for the
  review's mechanical floor) compose with it — the flags are
  orthogonal by construction.
- Batches, manifests, and checkpoints from before this change parse
  and match unchanged (`serde(default)` on the new manifest fields;
  the no-flags batch is byte-for-byte identical).
- Rust-only, like every extract control; the SDK producers inherit the
  whole set together in their own follow-up.
- The remaining #466 splits (an MCP promotion tool over the graph
  path; a CLI text-path preset) build on this without changes here.
