# 0048. Grants and quotas are keyed by context id; a derived context carries its parent's id

- **Status**: Accepted
- **Date**: 2026-10-04
- **Issue**: #966 (child of #961, under #958)
- **Related**: ADR 0045 (§2.2 grants/quotas on the id; §2.7 no
  compatibility), ADR 0046/0047 (every column that points at a context
  carries its id), #959 (`TAGURU_KEY_GRANTS`), #961 decisions 3 and 4
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

After ADR 0046/0047 every stored reference to a context carried its id,
except two settings an operator writes by hand — `TAGURU_KEY_GRANTS`
(`contexts`) and `TAGURU_CONTEXT_QUOTAS` (the keys) — and the link
between a source context and its derived artifacts (`NAME::communities`,
`NAME::consolidation`), which was the name plus a suffix. A rename moved
a grant or a quota onto whatever else carried the old name, a second
context with the same name inherited both, and a rename of the parent
silently detached its artifacts.

## 2. Decision

1. **Grants and quotas are keyed by id.** `contexts` in a grant and the
   quota keys are canonical context ids (`GET /contexts`' `id`). A
   value that is not one refuses boot (and a hot reload is refused with
   the previous table kept): a name would match nothing — a key locked
   out, a context uncapped — so it must not run. Entries naming an id
   that does not exist yet are fine (an import header may choose an id).
2. **A rename moves no grant or quota, and needs no destination
   check.** The destination of `rename` is a display name; no grant
   mentions a name, so the former gate on `to` is removed.
3. **A grant refusal echoes the id as given, never the display name.**
   Resolving the name for a context outside the grant would tell a
   scoped key that the id is live and what it is called — the existence
   oracle the cross-search and group gates were already built to avoid.
   (The quota refusal does name the context, beside its id, because its
   audience is the operator.)
4. **The default derived context's id derives from the parent's id.**
   `derived_context_id("{parent_id}::communities")` (and
   `…::consolidation`) — the SHA-256 construction `taguru-code sync`
   already uses. The offline CLI and the server find the same artifact
   without asking each other and without a name lookup; a first run
   registers it through the ordinary create block under the default
   display name. Operators can also read the id from `GET /contexts` to
   grant or cap it.
5. **A manifest names its source by id.** `source_context_id`
   (communities) / `context_id` (consolidation) replace the name; a
   search compares it to the id the caller addressed, so an override
   (`derived_id`) pointing at another source's artifact is a 409. The
   manifest keeps FORMAT_VERSION `2026-10-01` — the revision is still
   unreleased — and an older manifest fails to parse with the rebuild
   message rather than being read as something else.
6. **Overrides are ids.** `--into` (both commands) and the wire's
   `derived_id` take an id (validated); the response names
   `derived_id` and `derived_name`. The analysis stream's header names
   the analyzed context `context_id`.
7. **No wire boundary resolves a name.** The last one (`derived`) is
   gone, so the ambiguous-name conflict has no caller and is removed;
   `taguru-code sync`'s offline name lookup stays.
8. **Out of scope**: the router's name-keyed route map (its probing of
   ids is ADR 0047 §6), MCP's single-`context` argument names (#967).

## 3. Consequences

- Operators write ids into grant/quota JSON — the cost of a setting that
  no longer drifts when a label does. A scoped key that must read a
  derived artifact needs the artifact's id granted too, readable once it
  is built.
- Artifacts built before this change are not found; rebuild (the old
  context is an ordinary context and can be deleted).
- Two contexts with the same display name each own their own artifact
  and their own grants, and a rename of either is a label change only.
