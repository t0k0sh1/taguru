# 0052. A group create takes an optional id; a create that stops partway names it

- **Status**: Accepted
- **Date**: 2026-11-12
- **Issue**: #995 (decisions recorded on the issue, 2026-11-12)
- **Related**: ADR 0050 (groups have ids), ADR 0045 (id apart from name)
- **Supersedes**: ADR 0050 §2.4, in part — its create body
  `{name, description?, context_ids?, group_ids?}` gains an optional
  `id`. The rest of ADR 0050 stands. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0050 §2.4 has the router mint a group's id and send the same id to
every shard, because every shard holds every group. It leaves open how
the id reaches a shard. The router calls a shard with the caller's own
credentials (`hop_headers` forwards them), so there is no router-only
path: anything a shard accepts from the router, a caller can send too.

A router writes a group to its shards one after another and stops at
the first refusal; the shards before it keep the write. For a delta
update, retrying the request converges. For a create it would not: a
retry gets a new id from the router, and the copies left on the earlier
shards become groups nobody asked for.

## 2. Decision

1. **`POST /groups` takes an optional `id`.** The body is
   `{name, id?, description?, context_ids?, groups?}`.
   - Without `id`, the server mints one.
   - With `id`, it must be the canonical UUID spelling, or the request
     is a 400.
   - A create under an `id` that already stands answers that group's
     row when the stored record is identical to the one asked for, and
     `already_exists` (409) when it differs.
   - The router puts the id it minted into the body it sends every
     shard. A caller may bring its own id the same way.
2. **A create that stops partway names the id.** The router keeps the
   current policy for group writes: stop at the first refusal and leave
   the earlier shards as they are. The refusal it answers keeps the
   shard's status and code. When at least one shard already holds the
   group, its message also names the id and those shards, so the caller
   can finish the create by sending it again with that `id`. When the
   router minted the id, the message also offers `DELETE /groups/{id}`
   to remove the group. A caller's own id may name a group that stood
   before this create, so for it the message offers no DELETE.

## 3. Consequences

- A caller can create a group under an id of its choosing. This gives
  no new reach: the id names nothing until the create succeeds, and a
  taken id is refused.
- Import records still name groups by name until #996. A shard that
  restores a name it does not hold creates the group under an id of its
  own, so through a router a record new to the fleet would land under a
  different id on each shard. Before broadcasting the records, the
  router creates each such group empty on every shard that lacks it,
  under the id the other shards already use for that name (or one it
  mints when none does), and each shard's restore then replaces it in
  place. The import report still labels the record `created`.
  - If a shard refuses the records, the router removes those empty
    groups from that shard and the shards after it, which never
    restored them. The shards before it keep the restored records. It
    removes a group only while it is still the empty group it made, so
    a group a restore filled or another request changed stays.
  - A name the shards hold under different ids between them is a name
    two groups share. The router refuses an import that names it, as a
    record or as a child, the way a single instance refuses an
    ambiguous name, before anything is applied.
  - If a shard's directory cannot be read, the router skips this step
    and the shards fall back to their own ids, with a warning in the
    log.
