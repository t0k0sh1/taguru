# 0046. Every column that points at a context carries the context's id

- **Status**: Accepted
- **Date**: 2026-10-01
- **Issue**: #965 (child of #961, under #958)
- **Related**: ADR 0045 (id apart from name; §2.7 no compatibility),
  ADR 0042 (`type` / `version` / `id` columns), #961 decision 1 (a
  context's `name` is not unique) and decision 2 (what `create` means
  once the header names an id)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0045 gave a context an id and demoted its name to a display label
that may repeat. Every URL already takes the id (#964). The records
that *point at* a context — the source header's `context`, the schema
record's `context`, the import response's `batches[i].context`,
promote's `into`, `taguru extract --context` — still carried the
name, so a stream could not say which of two same-named contexts it
meant, and an export taken from one data directory named a context by
a label that the importing side was free to give to something else.

## 2. Decision

1. **A column that points at a context carries its id** — a
   lowercase hyphenated UUID, the `id` column of `GET /contexts`. The
   source header and the schema record say `context_id`; the import
   response says `context_id` in each batch and each schema outcome;
   promote's `into` is an id. A value that is not a canonical UUID is
   refused with a message derived from the path-segment rejection:
   `'{value}' is not a context id: context_id takes a lowercase
   hyphenated UUID — the id column of GET /contexts, or a fresh one
   (e.g. from uuidgen) alongside create — not the context's name`.
2. **`create` carries the name.** `create` is `{name, description?,
   pinned?, dice_floor?, semantic_floor?}` with `name` required and
   non-empty. Its meaning is: *if no context carries `context_id`,
   create one under this id and this name*; when one does, `create` is
   ignored, exactly as before. A client that creates a context
   therefore chooses its id (a fresh UUID) and repeats it in every
   later file. A header naming an unknown id without a `create` block
   is refused the way an unknown context was.
3. **An export writes the id and `create.name`.** Restoring into
   another data directory reproduces the same id; restoring under a
   different id is an explicit rewrite by the caller.
4. **`taguru extract` and `taguru benchmark extract` take
   `--context ID`**, plus `--name NAME` (and `--description`, which
   needs `--name`) to add the create block. The name and description
   fold into the manifest's existing description fingerprint, so
   changing either re-extracts, as changing the destination always did.
5. **The router resolves an id, not a name.** For an existing id it
   probes the shards (`GET /contexts/{id}`); for an id no shard
   carries, the header goes to the shard `create.name` maps to; with no
   create block it goes to the first shard, which refuses in the same
   words a single instance would — so a sharded deployment answers
   exactly as a single one.
6. **No compatibility** (ADR 0045 §2.7). A file or response still
   carrying `context` fails with `unknown field`. `FORMAT_VERSION` is
   not bumped: the unreleased `http_contract` 2 already covers the
   break (ADR 0045 §8), and `unknown field` names the column.
7. **Out of scope here, in later #965 steps or other issues**: group
   records' `contexts` and the group API's member lists, cross-search
   request and response (`context_ids`, `context_id` + `context_name`),
   evidence items, replica and `restore` (#965), grants and quotas keyed
   on names (#966), MCP argument renames (#967).

## 3. Consequences

- A stream no longer depends on a label staying unique or stable:
  renaming a context, or creating a second one with the same name,
  cannot redirect an import.
- Hand-written files must carry a UUID; a quick-start has to say
  `uuidgen`. The cost is deliberate — the alternative is a name that
  can mean two contexts.
- Until grants are re-keyed (#966), the import path resolves a batch's
  display name from its id for the grant and quota checks.
- Group records still list names in this step, so an export-then-import
  of a group and its members round-trips only while names stay unique.
  The group step removes that.
