# 0044. `GET /version` reports the record format once, as `record_formats`, and `import --url` / `export --url` check it before a byte moves

- **Status**: Accepted
- **Date**: 2026-09-18
- **Issue**: #937 (last step)
- **Related**: ADR 0042 (one `version` date for every record; its
  "leaves for later" list named this rename), ADR 0043 (the schema
  document joined that date)
- **Supersedes**: ADR 0005 §3 items 5 and 7 (`batch_formats`,
  `communities_formats`) and ADR 0009 §13's `schema_formats` preflight —
  the three dimensions and the schema-only preflight; the rest of both
  ADRs stands. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Context

ADR 0005 gave `GET /version` one dimension per taguru-written,
taguru-reread stamp: `batch_formats` for the source stream,
`communities_formats` for the communities artifact, and (ADR 0009 §13)
`schema_formats` for the schema document — each an integer list,
each checked by equality. ADR 0042 replaced those stamps with one
`version` date shared by every JSON / JSONL record, so after #937's
earlier steps the three dimensions reported the same single string
three times. Three names for one fact invite a reader to look for a
difference that does not exist, and the CLI's `import --url` preflight,
which read `schema_formats` only when the stream carried a `schema`
record, no longer matched what the check is about: every record in
the stream follows that one revision, not just the schema one.

## 2. Decision

1. `GET /version` reports **`record_formats`**: the list of record
   format dates this build reads and writes (today one element,
   `["2026-09-17"]`), checked by equality. `batch_formats`,
   `schema_formats`, and `communities_formats` are gone.
   `image_formats` stays as it was: the binary snapshot is not a JSON
   record and keeps its integer range.
2. **`import --url` and `export --url` both read `record_formats`
   before a byte moves, on every invocation.** The verb refuses, naming
   both sides, when the list does not contain this build's date — and
   when the server reports no `record_formats` at all, since such a
   server predates the format this build writes and reads. Absence is
   no longer "safe" for `export`: a server without the dimension writes
   records this build refuses line by line, and the preflight says so
   once, up front.
3. `http_contract` stays at 1 — the owner's decision for the whole
   #937 series (recorded on #937): the response-shape changes of ADR
   0042 through this one are taken as one pre-1.0 format cut, not as
   contract revisions.

```json
{
  "server": "0.9.8",
  "http_contract": {"current": 1, "supported": [1]},
  "mcp_contract": {"current": 1, "supported": [1]},
  "mcp_protocol": {"supported": ["2024-11-05", "2025-03-26", "2025-06-18"]},
  "record_formats": ["2026-09-17"],
  "image_formats": [1, 2, 3, 4, 5, 6]
}
```

## 3. Consequences

- A client reading `batch_formats` / `schema_formats` /
  `communities_formats` finds none; `record_formats` is the one
  dimension. Neither official SDK read the old three.
- `import --url` sends one `GET /version` it did not send before for a
  schema-free stream. A CLI of this release against a server of an
  earlier one refuses before sending, with an upgrade message, instead
  of the server's own "not a source file header" refusal.
- The protocol manual (`GET /protocol`) and the MCP `initialize`
  instructions carry the same block, so they change together.
