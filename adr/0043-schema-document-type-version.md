# 0043. The schema document states `type` and `version` too; an installed schema does not cross over by itself

- **Status**: Accepted
- **Date**: 2026-09-18
- **Issue**: #937
- **Related**: ADR 0042 (the three columns; its "leaves for later"
  list named this document and waited for the decision recorded here),
  ADR 0009 §5 (the schema document and its at-rest posture), ADR 0005
  (`GET /version`'s `schema_formats`)
- **Supersedes**: — / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The schema document — one struct serving three places: the body of
`PUT /contexts/{name}/schema` (and what `GET` serves back), the
arguments of MCP's `put_schema` / `validate_schema`, and
`{stem}.schema.json` in the data directory.

## 2. Context

ADR 0042 moved every record taguru reads to `type` / `version` / `id`
and left this one document on `"schema": 1` — the same defect it
removed everywhere else (a key that declares the kind by its presence
and carries a revision in a value that reads as an identifier). It
held back for one reason: the third place above is the server's own
persisted state, so changing the shape without a reading courtesy
stops the server at boot on any data directory that holds an installed
schema. That cost needed the owner's word. The owner gave it on #937:
change the document, no compatibility.

## 3. Decision

1. The document opens with **`"type": "schema"`**, required, and
   carries **`version`** under ADR 0042's rule unchanged: the shared
   date, optional on input, an absent one meaning the running build's
   own, a present one equal to the build's or refused by name, `null`
   not an omission. There is no `id`: a `context` has one schema.
2. **What installs states its version.** `install` fills an omitted
   `version` in, so the stored file, its digest, and every `GET` carry
   it, and spelling it out or leaving it off installs the identical
   document.
3. **`"schema": 1` is not read** — not on the wire, not over MCP, not
   at rest. It is an unknown field like any other.
4. `GET /version` reports `schema_formats` as the same date string the
   other format dimensions carry. `http_contract` stays at 1 (owner's
   decision, #937).

```json
{"type": "schema", "version": "2026-09-17", "mode": "strict", "closed_labels": false, "types": {…}, "relations": {…}}
```

## 4. Consequences

- A data directory in which any `context` has a schema installed by an
  earlier release does not boot under this one: `{stem}.schema.json`
  fails to parse (`unknown field "schema"`), the bytes are set aside as
  `{stem}.schema.corrupt`, and the server stops — the posture ADR 0009
  chose so that a schema never silently degrades to `mode: off`.
- Nothing converts it. The way across, checked against a real data
  directory: rewrite each `{stem}.schema.json` so it opens with
  `"type": "schema"` (and, optionally, `"version"`) in place of
  `"schema": 1`, then set `schema_digest` in `{stem}.meta.json` to the
  sha256 of the rewritten file's bytes. Or, before upgrading, save each
  document with `GET`, and after upgrading install it again on a data
  directory that has none.
- Clients that build the document — both SDKs' `SchemaDocument`, MCP
  callers — send `type` and may leave `version` off; an SDK never sends
  it as `null`.
