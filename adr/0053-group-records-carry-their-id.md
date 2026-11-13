# 0053. A group record carries the group's id; a restore lands under it, and the router stops aligning ids

- **Status**: Accepted
- **Date**: 2026-11-13
- **Issue**: #996 (under #993 / #958)
- **Related**: ADR 0050 (groups have ids; §2.5 nesting and cross-search
  lists are ids, §2.6 export files are named by id), ADR 0052 (a group
  create takes an optional id), ADR 0042 (`type` / `version` / `id`
  columns), ADR 0045 §2.7 (no compatibility)
- **Supersedes**: ADR 0052 §3, in part — its second bullet ("Import
  records still name groups by name until #996 …") and the three
  sub-bullets under it: the router no longer creates empty groups
  before an import, no longer removes them after a refusal, and no
  longer refuses a name the shards hold under different ids. The rest
  of ADR 0052 stands. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0050 §2.5 and §2.6 move the wire's nesting lists, the cross-search
targets and the export file names to group ids. The import stream's
`group` record was the last place that still named a group by name:
its `id` column held the name, and its `groups` column listed the
children by name. A restore looked the name up, replaced the group it
found, or minted a new id when it found none.

Behind a router, that lookup happened on every shard separately, so a
record new to the fleet would get a different id on each shard. ADR
0052 §3 worked around it: before broadcasting, the router created each
new group empty on every shard under one id, and undid that after a
refusal. It also refused a name the shards held under different ids,
since a name lookup could not tell those groups apart.

## 2. Decision

1. **A `group` record is `{type, version, id, name, description?,
   context_ids?, group_ids?}`.**
   - `id` is the group's id: the canonical UUID, the `id` column of
     `GET /groups`. A writer creating a new group picks a fresh one.
   - `name` is the display name and is required.
   - `group_ids` lists the children by id. The `groups` column is gone.
   - A value in `id` or `group_ids` that is not a group id refuses the
     stream with its line number, as a name in `context_ids` already
     does.
2. **A restore lands under the record's id.** The record replaces the
   group with that id whole, its name included, or creates the group
   under that id when none stands. Names play no part, so two records
   that share a name are two groups. A child id must name a standing
   group or another record of the same set, and a member id a
   registered context; otherwise the set is refused before anything
   is written, as before.
3. **The router broadcasts the records and nothing else.** Every shard
   restores each record under the same id. That is the same-id
   guarantee a `POST /groups` broadcast gives (ADR 0050 §2.4), so the
   empty groups ADR 0052 §3 created first, their removal after a
   refusal, and the refusal of a name split across shards are removed.
   The partial-write policy is unchanged: a shard that refuses stops
   the broadcast, the shards before it keep the restored records, and
   re-sending the stream converges.
4. **The import report names the id.** Each row under `groups` is
   `{id, name, outcome, contexts, groups}`, where `contexts` and
   `groups` stay member counts.
5. **The file format revision becomes `2026-11-13`.** The record's
   shape changed, so a stream an earlier build wrote is refused by its
   `version` instead of being half-read (ADR 0042 §3.2; ADR 0045 §2.7).

## 3. Consequences

- An exported group restores as the same group anywhere, and so do its
  children. A stream that carries a group and its children restores
  the nesting exactly, whatever the names are.
- The router's import path for groups is one broadcast, like every
  other group write. A shard missing a group (drift) gets it under the
  fleet's id the next time the record is imported.
- A hand-written record must now carry an id. To create a new group,
  take one from `uuidgen`; to change a standing group, take its id
  from `GET /groups`.
- Streams written before this revision must be re-exported with the
  build that wrote them and imported again (ADR 0045 §2.7).
