# 0047. Groups and cross-context search name their members by id

- **Status**: Accepted
- **Date**: 2026-10-01
- **Issue**: #965 (child of #961, under #958)
- **Related**: ADR 0046 (every column that points at a context carries
  its id — this finishes the list its §2.7 left for later steps),
  ADR 0045 (id apart from name; §2.7 no compatibility), ADR 0042
  (`type` / `version` / `id` columns), #961 decision 1 (a context's
  `name` is not unique)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0046 moved the import layer to ids and said that group records,
the group API's member lists and the cross-search request and response
came in a later #965 step. Until then a `group` listed member *names*
and a cross-search body named *names*, so a group could not tell two
same-named contexts apart, a rename had to rewrite every group that
mentioned the old name, and an export of a group and its members only
round-tripped while names stayed unique.

## 2. Decision

1. **A group's members are ids.** `GroupRecord.contexts` became
   `context_ids`; the group listing, the group export line, `PUT
   /groups/{name}` (`context_ids`) and `PATCH` (`add_context_ids`,
   `remove_context_ids`) all carry canonical-UUID ids. A value that is
   not one is a 400 `invalid_argument` (`is not a context id`, same
   message family as a path segment); an id no context carries is the
   existing `no_context`. The on-disk record is `deny_unknown_fields`,
   so a file still holding `contexts` fails to parse and is set aside
   `.corrupt` — no compatibility (ADR 0045 §2.7).
2. **A rename touches no group.** Membership names ids, which a rename
   never changes; the rewrite loop in `rename_context` is removed. The
   group `fingerprint` (a change token over member revisions) no
   longer moves on a rename, since the revision counters do not either.
   A child group is still named by group name (groups keep names as
   their key), so `groups/{name}/rename` is unchanged.
3. **Cross-search takes `context_ids`.** `POST /recall`, `/query` and
   `/sources/search` take `context_ids` (and `groups`); a name is a
   400. Responses tag each match/hit with `context_id` **and**
   `context_name` — the id because every follow-up call needs it, the
   name because a display column should not need a second lookup. The
   plan's per-context entries carry the same pair; a graph plan's list
   is `context_ids`; the cross cursor's tiebreak field is `context_id`.
4. **Ties break by id.** Cross-context ranking breaks an identical
   `(weight, subject, label, object)` tie by context id, because names
   repeat. Passage merging keeps its order (direct targets first, then
   group-resolved ones in id order).
5. **The retrieval cache key includes the display name.** Responses echo
   `context_name`, and a rename moves no revision counter, so a cached
   page would otherwise serve the old name after a rename.
6. **The router locates an id by probing.** The route map stays
   name-keyed (grants are, until #966); a cross-search or group id is
   located by `GET /contexts/{id}` on the shards (a single-shard map
   skips the probe). Groups exist on every shard with a per-shard member
   projection. An id every reachable shard denies is refused as
   unknown; an id that an *unreachable* shard might own is sent to the
   unreached shards so the answer degrades to a labeled partial, as
   every fan-out read already does.
7. **MCP follows the wire.** The tools that mirror a wire body take the
   wire's names (`context_ids`, `add_context_ids`, `remove_context_ids`,
   cursor `context_id`). Renaming the single-`context` `context`
   argument stays #967.
8. **Not touched**: grants and quotas keyed on names (#966), the
   evaluate/benchmark manifests' own `context` fields (they are tool
   configuration, not a stored column), and the replica/restore
   surface (already addressed by file stem, which is the id).

## 3. Consequences

- A group and its member contexts now round-trip across data
  directories exactly, because the ids travel with the export.
- Renaming a context is a metadata change only again.
- A client that built group member lists from names must map them to
  ids through `GET /contexts` first; a name is refused rather than
  guessed at, since several contexts may carry it.
- Through the router, a cross-search over an id costs one probe per
  shard on a miss; a single-shard deployment pays nothing.
