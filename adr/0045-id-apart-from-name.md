# 0045. A context, a group, and a source each have an id apart from their name

- **Status**: Accepted
- **Date**: 2026-09-19
- **Issue**: #958
- **Related**: #851 (a record's own key is `id`), ADR 0042 (`type` /
  `version` / `id` columns), #957 (context and group rows say `id`),
  #959 (`TAGURU_KEY_GRANTS`)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

A context has one identifier: its name. The name is any UTF-8 string
up to 1024 bytes — `酒蔵/2026` is legal — and it serves as the URL path
segment (`/contexts/{name}/…`, 46 routes), the data-directory file stem
(bytes outside `[A-Za-z0-9_-]` become `%XX`), the value every other
record uses to point at the context (source headers, schema records,
group member lists, cross-search matches, import results, the grant
and quota settings), and, since #957, the row's own `id`. A group is
the same.

A source is identified by the string in its header line's `id`
(`{"type": "source", "id": "docs/aomine.md", …}`). Whoever produces the
file writes it — `taguru extract` writes the document's path, a person
writes whatever they choose — and taguru stores it verbatim and hands
the same string back in every association, hit, and citation. A source
has no name apart from it.

Consequences:

- A name that is not URL-safe makes every URL unreadable and
  untypeable (`/contexts/%E9%85%92%E8%94%B5%2F2026/…`), and nothing in
  the docs or `GET /protocol` says what a name may contain or how to
  encode it.
- `POST /contexts/{name}/rename` changes the identifier itself: the
  context's whole file family is renamed, every group naming it is
  rewritten, both names are reserved for the duration, and a marker
  file makes the move resumable. Everything else that points at the
  context — an external index, a grant, a quota — has to follow, and
  what cannot follow breaks silently.
- Calling that value `id` (#957) does not make it one: a name is a
  label people choose; an id is a key the system keeps stable.

Restricting names to URL-safe characters was considered and rejected:
the name would still be the id, and moving the human-readable name
into `description` would leave the context without one.

## 2. Decision

1. **A context, a group, and a source each have two distinct
   properties: `id` and `name`.** `id` is a UUID the server assigns
   when the record is created and never changes. `name` is the
   human-readable label — any UTF-8 string within the existing length
   cap, changeable. For a source, `name` is the string its header line
   carried (`docs/aomine.md`, a URL). `description` stays what it is.
   Sort order of `id` carries no meaning and is not a requirement.
2. **`id` is what refers to a context, a group, or a source everywhere
   a machine reads it**: the URL path (`/contexts/{id}/…`,
   `/groups/{id}/…`), the data-directory file stem, the source header
   line and the schema record (`context_id`), a group's member lists
   (`context_ids`, `group_ids`), cross-search matches and import
   results (`context_id`), an association's attribution, a search hit,
   a citation, a retraction, and a change-feed event (`source_id`),
   the grant and quota settings (keyed by id), replication, MCP tool
   arguments, and SDK method arguments.
3. **A response that returns a context, a group, or a source carries
   both `id` and `name`.**
4. **Finding by name is a listing operation.** To address exactly one
   record, use its id. To find records by name (exact or partial),
   filter the listing; whether a listing offers a name filter, and its
   matching rule, follows from a concrete need, not from this ADR.
5. **Whether `name` must be unique is decided per entity** in its
   child issue, not here.
6. **Rename changes `name` and nothing else.** `POST
   /contexts/{id}/rename` rewrites the `name` value in the context's
   own metadata; no file is renamed, no group is rewritten, no name is
   reserved, and the resumable-move machinery goes.
7. **No compatibility.** A data directory whose contexts, groups, or
   sources carry no id is not read; the operator exports with the
   previous release and imports with this one, which assigns ids.
8. **`http_contract`** is bumped if the change is breaking under ADR
   0005 §4 and no unreleased bump already covers it; otherwise not.

## 3. Consequences

- URLs are readable and typeable: `/contexts/3f2a9c1e-…/recall`. A
  human still reaches a context by name through the listing.
- Every record and setting that named a context, group, or source by
  name changes shape or key; the child issues under #958 list them.
  The wire fixtures, both SDKs, the LangChain packages, `taguru-code`,
  docs, and examples follow.
- `GET /contexts` sorts by `name` and pages on `(name, id)`; `id` order
  carries no meaning.
- The source header line becomes `{"type": "source", "version": …,
  "id": <uuid>, "name": "docs/aomine.md", "context_id": <uuid>, …}`
  when taguru writes it; what a hand-written file must carry (whether
  `id` may be omitted and assigned at import) is the source child
  issue's to decide.
