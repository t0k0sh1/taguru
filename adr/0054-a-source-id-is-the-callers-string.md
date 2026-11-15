# 0054. A source's id is the string its caller chose; fields that name a source say `source_id`

- **Status**: Accepted
- **Date**: 2026-11-15
- **Issue**: #1038 (decisions recorded on the issue, 2026-11-15), #1039
  (under #958)
- **Related**: ADR 0045 (an id apart from the name), #851 (a column that
  points at a record says which record's id it holds), ADR 0042 (`type` /
  `version` / `id` columns)
- **Supersedes**: ADR 0045, in part — the source parts of §2.1 (a source
  gets a server-assigned UUID `id` and keeps its string as `name`), §2.3
  (a response returning a source carries both `id` and `name`), and §3's
  header line `{…, "id": <uuid>, "name": "docs/aomine.md", …}`. §2.2's
  list of places that refer to a source by `source_id` stands. ADR 0045
  stands as written for contexts and groups. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0045 gave contexts, groups and sources the same treatment: a
server-assigned UUID `id`, with the human-readable string as `name`. For
contexts and groups the reasons were concrete. Their names were URL path
segments, so a name outside the URL-safe set made every path
unreadable, and a rename changed the identifier itself, so everything
that pointed at the record had to follow.

A source has neither problem:

- It never appears in a URL path. Every `/contexts/{id}/sources/…` route
  takes the source in the request body.
- It cannot be renamed. Nothing in the API changes a source's string.

What a source's string does do is carry idempotency. Ingest replaces a
document per source (retract, then apply), so re-ingesting a document
under the same string replaces it instead of adding a second copy.
`taguru extract` uses the document's path, `taguru-code` the file path,
LangChain loaders `metadata["source"]`, and an LLM writing through MCP
names the document on every `add_associations` element.

Giving a source a server-assigned id would leave two ways to write:

- Create the source first, receive its id, and write with the id. Every
  ingest pays one more round trip, and the LLM's ingest loop must
  remember an id it did not choose.
- Write with the string and let the server resolve it to an id. For
  re-ingest to stay idempotent, the string must then be unique within
  the context — at which point the string is the key, and the id is a
  second name for it.

## 2. Decision

1. **A source's id is the string its caller chose.** The server stores
   it verbatim and never assigns one. A source has no separate `name`.
   The id is scoped to its context, as before.
2. **A field that names a source says `source_id`; a list of them says
   `source_ids`** (#851). A listing row's own key is `id`.
   - #1039 renames the write side: the `associations` element's `source`,
     the retract body's `source`, promote's `sources`, the import report
     row's `source`, and the `GET /contexts/{id}/sources` row (`name` →
     `id`; the duplicate bare `sources` list is removed).
   - The read and audit sides (attributions, hits, citations, lookup,
     search and explain, consolidation, evidence) follow in #1040 and
     #1041.
   - A count is not an id and keeps its name (`stats.sources`, a search
     plan's filter counts). The `/contexts/{id}/sources/…` routes name the
     collection and keep theirs.
3. **The import stream's source header keeps `id`, and it stays
   required.** Its value is the caller's string. Nothing assigns,
   converts or reinterprets it, so export → import reproduces every
   source under the same id.
4. **A request that still carries a renamed field is refused by name.**
   Where a body is walked leniently and would otherwise ignore an unknown
   key — the `associations` elements — the old key is a 400 issue on its
   own path, so an old caller cannot have its facts written unsourced
   without a word. Elsewhere the renamed field is simply required and its
   absence is refused as before.
5. **The data directory is unchanged.** The WAL and the passage store
   keep their internal `source` keys, which no client reads, and the file
   format revision does not move. `http_contract` stays on the unreleased
   2 (ADR 0045 §2.8).

## 3. Consequences

- Ingest stays one call per document, and an LLM keeps naming the
  document it is writing about.
- A caller that wants a stable source must choose a stable string, as it
  always had to. Two documents written under one string are one source.
- Every wire field that names a source changes name across three child
  issues. Until #1041 lands, the write side says `source_id` while the
  read side still says `source`.
- LangChain Documents keep `metadata["source"]`, the loaders' own
  convention, so documents from any loader ingest without remapping.
