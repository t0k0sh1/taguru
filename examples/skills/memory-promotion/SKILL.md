---
name: memory-promotion
description: "Promote session notes (episodic memory) into a permanent Taguru context (semantic memory): scratch-context conventions, the one-call promote tool for already-structured scratch, the extract→import→audit→retract procedure for prose, and the rules that keep provenance and time intact. Use when ending a working session whose notes are worth keeping, or when asked to consolidate scratch knowledge."
---

# memory-promotion

Taguru already has the destination (the association graph) and the
vehicle (`taguru extract` / `POST /import`); what makes promotion work
is discipline, not new machinery. This skill fixes the conventions and
the procedure (issue #465; design: issues #423, ADR 0011, ADR 0012).

## Conventions — follow these from the FIRST scratch write

- **Scratch is an ordinary context**, named `scratch-{agent}` (one per
  agent, long-lived) or `scratch-{agent}-{topic}` when sessions must
  not mix. No TTL exists and none is coming: forgetting is an explicit
  operation in this store, always.
- **Source ids name the session, globally uniquely**:
  `session:{agent}:{id}` (the id a UUID or an equally unique token)
  for the session's running note, with `/{doc}` appended when one
  session produces several documents. Global uniqueness is
  load-bearing, not cosmetic: import is retract-then-apply **per
  source id**, so two agents sharing a bare `session:{id}` in one
  permanent context would silently replace each other's assertions.
  The id keeps meaning after promotion — a promoted fact's citation
  still points at the session that produced it.
- **Always declare `date`** (epoch seconds, the session's day is fine)
  when storing passages. `date ?? stored_at` is the assertion time
  every time-windowed read and the staleness audit runs on (ADR 0011);
  an associations-only source with no passage is invisible to every
  window, so store at least the session note as a passage.
- Tag scratch sources with the session's topic tags — passage search's
  `tags` filter is how a later session finds its own trail.

## During a session

Write notes as you go: facts as associations (subject, label, object,
weight, `source: "session:{id}"`), the running note as a passage under
the same source id. Re-asserting across sessions is corroboration;
re-asserting within one note inflates weight — don't.

## Promotion — end of session, or on request

1. **Review what the scratch holds**: `recall`/`query` the scratch
   context, or `taguru communities --context scratch-...` for a themed
   overview when the scratch has grown.
2. **Graph path — one call when the structure is already right**
   (ADR 0018): when the keepers are the scratch's own structured
   associations (you wrote them during the session; there is no prose
   left to extract), call the `promote` MCP tool on the scratch
   context with `{into: PERMANENT, sources: [the session ids]}`. Each
   source moves whole — passage, `date`, tags, only its own share of
   every edge — source ids survive (citations still name the
   session), re-promotion is idempotent, and the landing-zone audit
   comes back in the same response: jump straight to step 5's
   judgments, then step 6. `dry_run: true` previews with nothing
   written. Steps 3–4 are the TEXT path, for keepers that exist as
   prose.
3. **Extract the keepers**: `taguru extract` over the session passages
   into import batches targeting the PERMANENT context —
   `--source-id session:{agent}:{id}`, `--date`, and `--tag` bake the
   conventions into the batch (ADR 0017); `--vocabulary` (over a
   `taguru export` of the permanent context) steers spellings to the
   ones the graph already uses, and `--coverage` reports what the
   extraction left behind, sentence by sentence (ADR 0015/0016).
4. **Import**: `POST /import` / `taguru import` — retract-then-apply
   per source, so re-promoting the same session is idempotent, not
   duplicated.
5. **Audit the landing zone**: judge the consolidation audit on the
   permanent context (bundled in `promote`'s response on the graph
   path; standalone via `taguru consolidation --context NAME` or the
   `audit_consolidation` MCP tool) — promotion is exactly when merge
   twins and contradictions appear. Judgments are proposals; apply the
   accepted ones through ordinary writes (alias / retract / negative
   weight / re-import).
6. **Retire the promoted scratch**: `retract_source` the promoted
   session sources from the scratch context (or delete the whole
   scratch context when everything promoted). Unpromoted scratch stays
   until someone decides otherwise — that is the posture, not a gap.

## What NOT to do

- Don't promote by copying text into a new spelling universe — step
  3's vocabulary steering (and, on the graph path, the audit's merge
  candidates) is what keeps one referent one spelling.
- Don't invent an end date for a superseded fact: assert the new fact
  with its own date and let as-of queries and the audit sort the
  regimes out (ADR 0011 §6).
- Don't auto-delete scratch on a timer, and don't skip the audit on a
  large promotion — the audit run is cheap when nothing changed
  (fingerprint-reused judgments cost zero LLM calls).
