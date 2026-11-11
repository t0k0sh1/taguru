# 0050. Groups have ids: the file is stemmed by a minted id, nesting refers by id, and the name is a non-unique label

- **Status**: Accepted
- **Date**: 2026-11-11
- **Issue**: #993 (under #958); implemented by #994 (records and files),
  #995 (HTTP paths and create), #996 (reference columns), #997 (MCP,
  CLI, SDKs, docs)
- **Related**: ADR 0045 (id apart from name; §2.7 no compatibility),
  ADR 0046 / 0048 (context references by id), #961 (the same move for
  contexts)
- **Supersedes**: ADR 0047 §2.2, in part — its clause "a child group is
  still named by group name (groups keep names as their key)". The
  rest of ADR 0047 stands. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

After #961 a context has an id, and everything that points at a context
carries that id. A group still had none: its file was `{name}.group`
under a percent-encoded name, a parent named its children by name, and
a rename moved the file, rewrote every parent, and kept a
`.grouprenaming` marker so boot could finish a rename a crash had cut
short. The wire addressed groups by name everywhere — `/groups/{name}`,
the nested `groups` list, the cross-search `groups` list, the export
file name.

## 2. Decision

1. **A group's `name` is not unique** (#993 decision 1). Behind the
   router, a group write is broadcast to every shard one after another,
   so a uniqueness check on each shard can race; an import from another
   server can also bring in a name that is already taken. The id
   identifies the group, so the name does not have to. Finding a group
   by name is a listing operation, as it is for contexts (ADR 0045
   §2.4).
2. **The id is minted by the server: a canonical UUID v4, which is also
   the file stem** — `{id}.group` holds `{id, name, description,
   context_ids, group_ids}`. One function mints every group id. A
   record with no `id`, or a stem that is not an id, comes from a data
   directory written before this decision. Boot refuses it and points
   to the migration (export with the release that wrote it, then import
   here; ADR 0045 §2.7), and does not treat the file as corrupt and
   reset it. A record whose `id` is not its own stem is also refused.
3. **Nesting refers to child groups by id** (`group_ids` on file). A
   rename therefore rewrites the group's own record and nothing else:
   no file move, no parent rewrite, no rename marker, and nothing for
   boot to resume.
4. **Creating a group is `POST /groups {name, description?,
   context_ids?, group_ids?}`**, and the response is the row with its
   id (#993 decision 2). It mirrors `POST /contexts`, and MCP
   `create_group` takes the same shape. The router mints the id itself
   and sends the same id to every shard, because every shard holds
   every group. Creating a group with `PUT /groups/{name}` goes away.
5. **On the wire, nested groups and cross-search groups are lists of
   ids**: `group_ids`, `add_group_ids`, `remove_group_ids` (#993
   decision 3). A client looks a name up with `GET /groups` first. Where
   an offline tool must name a group before any server exists, an id
   derived from a seed is allowed, built the same way as
   `derived_context_id`.
6. **Export files are named by id**: `{out}/{id}.group.jsonl` (#993
   decision 4).
7. **The name bridge, while the wire still speaks names.** #994 changes
   the files and the registry and leaves the wire as it was. Every
   entrance still takes a group name and resolves it through a name
   index, and keeps the name unique within the API: create and rename
   refuse a name another group already has. A name that several groups
   share can only come from a data directory put together by hand.
   Every name entrance answers it with a 409 `conflict` and does not
   pick one of the groups. #995 removes the bridge.

## 3. Consequences

- A data directory from an earlier release does not boot on this
  build. That is the documented break (ADR 0045 §2.7), and boot says
  how to migrate.
- A group rename can no longer be left half done. The marker and its
  resume logic (`resume_rename_markers` and the rest) are deleted. They
  were the last users of that machinery, because a context rename
  stopped using markers in #963.
- Until #995 lands, the wire behaves exactly as before; this PR changes
  no fixture. While the bridge is in place, two groups that share a
  name collide on one export file name. The export file name moves to
  the id in #996.
