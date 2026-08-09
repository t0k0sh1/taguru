# 0018. Graph-path promotion as a server verb (MCP `promote`)

- **Status**: Accepted
- **Date**: 2026-08-09
- **Issue**: #466 (S2)
- **Related**: #465 (the runbook this bundles), ADR 0012 (the audit it
  runs and the explicit-forgetting posture it keeps), ADR 0017 (S1,
  the extract flags that feed this), ADR 0009 §13 (the credential
  boundary that decides CLI vs server), ADR 0005 (the batch contract
  the transfer rides)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

One server verb — `POST /contexts/{name}/promote`, advertised as the
MCP tool `promote` — that moves named scratch sources into a permanent
context over the **graph path**: the scratch's already-structured
associations, passages, dates, and tags, promoted without
re-extraction and therefore without an LLM or any model credential.
Out of scope: the text path (session notes → `taguru extract` → import
— needs model credentials the server never holds, so it stays a CLI
composition; a preset verb is #466 S3 if ever needed), applying audit
judgments, and retiring the promoted scratch.

## 2. Context

The #466 rehearsal measured one promotion at 8–10 manual operations;
S1 (ADR 0017) removed the hand-editing from the extract half. What
remains mechanical is the transfer itself: export the scratch, keep
only the keeper sources, re-head every batch at the permanent context,
import, audit the landing zone. Every step is an existing server
operation — which is exactly why the rehearsal's re-evaluation
concluded the graph path can be a server verb while the text path
cannot: no step needs a model credential, so bundling them behind
`/mcp` respects ADR 0009 §13 as-is. The judgment points the rehearsal
proved irreducible — WHICH sources to promote, and what to do with
audit candidates — stay with the calling agent on both paths.

## 3. Decision

**`promote` IS export → filter → re-head → import → audit, in one
request, built from the same machinery the manual procedure uses.**
Body: `{into, sources[], audit?}`; `?dry_run=true` previews.

1. **The transfer is the export/import round trip, not a third write
   path.** The scratch's [`ExportSnapshot`] is filtered to the named
   sources — each association keeps only their attributions, its
   count/weight recomputed from what is kept, edges left with nothing
   drop — then rendered as an ordinary import stream headed at `into`
   and applied batch by batch with `POST /import`'s own
   retract-then-apply. Re-promoting the same sources is therefore
   idempotent, aliases are carried exactly when their canonical is
   live in the promoted slice (the render's standing rule — the count
   of dropped ones rides the response), the unsourced residual cannot
   travel (filtering makes attributed count equal total by
   construction), and the scratch's schema never installs into the
   destination (the snapshot's schema is cleared; the destination's
   own schema judges the incoming batches instead, refusing in
   `strict` exactly as an import would).
2. **Provenance travels whole.** Source ids, `stored_at`, `date`, and
   tags ride the stream verbatim, so a promoted fact's citation still
   names the session that produced it and every windowed read keeps
   working — the runbook's provenance promise, now enforced by
   construction rather than by operator care.
3. **Promote never creates and never retires.** The destination must
   already exist (checked up front, and the stream's create block is
   stripped so a context deleted mid-request refuses instead of
   resurrecting under the scratch's meta — promotion lands in an
   established context, never silently mints one). The promoted
   scratch stays until the agent explicitly retracts it —
   `retract_source`/context deletion remain the runbook's step 5;
   forgetting stays an explicit operation (ADR 0012's posture).
4. **Missing sources refuse whole, before anything applies.** A
   mistyped session id under retract-then-apply would otherwise
   no-op silently; instead every requested id must exist in the
   scratch (as a passage or a live attribution) or the request refuses
   naming the absentees, `nothing_written`.
5. **The audit is bundled, its judgments are not.** After a real
   apply, the destination gets the same merge/contradiction/staleness
   computation `audit_consolidation` runs (all three checks, default
   ceilings), riding back as `audit` — candidates with fingerprints,
   never applications; `audit: false` opts out for a large destination
   and `dry_run` skips it (nothing landed to audit). Tuned re-runs
   stay one `audit_consolidation` call away, fingerprint reuse intact.
6. **Write role, both contexts checked.** Promote is an ingest-loop
   verb — `retract_source`'s classification, not `/import`'s Admin
   (it cannot create contexts and carries no group or schema records).
   The route check covers the scratch; the handler checks the
   key's grant on `into` before anything applies, `/import`'s
   body-context discipline. Through `taguru router` the request
   proxies whole to the shard owning the scratch, so a destination
   living on another shard refuses there (`no_context`) — promotion
   through the router requires the pair on one shard, a documented
   divergence in route.rs's own list.

## 4. Consequences

- The runbook's steps 3–4 become one call on the graph path, and step
  2 disappears from it entirely (no extract when the scratch's
  structure is already right); with S1, a session already written in
  structured form promotes with: review → `promote` → judge audit
  candidates → `retract_source`. The irreducible judgment points are
  now the ONLY manual steps.
- No new credential surface and no new write semantics: everything the
  verb does was already expressible with existing operations, so the
  security review surface is composition, not new capability.
- The response reuses `/import`'s per-batch outcome shape and the
  audit's section shapes — clients that parse either parse this.
- Building the dry run exposed two standing `/import?dry_run` defects
  the same change fixes at the root: a preview held no cross-batch
  state, so a restore whose aliases trail their canonicals (every
  export — aliases ride the last batch) and every post-first batch of
  a fresh-name restore (the create block rides only the first)
  refused spuriously where the real import applies cleanly. Previews
  now seed each batch's checks with what the batches before it would
  intern and create. Cross-batch alias CONFLICTS remain un-predicted —
  that gap only lets a preview pass what a real run would refuse, the
  advisory direction the capacity caps already occupy.
- Not in this split, deliberately: promoting BETWEEN servers (export's
  file form already covers migration), a `since`/`until` or tag filter
  choosing sources server-side (the agent already holds
  `list_sources`), and the text-path preset (#466 S3, if S1 plus this
  proves insufficient).
