# 0042. A record says what it is, which revision, and which one — in three columns: `type`, `version`, `id`

- **Status**: Accepted
- **Date**: 2026-09-17
- **Issue**: #937 (implements a decision recorded in #851)
- **Related**: ADR 0002 (where `taguru_batch` was introduced), ADR 0003
  §10 (equality-versus-range acceptance of a version stamp — §5 below
  says how to read it now), ADR 0005 (`GET /version`'s format
  dimensions), ADR 0009 §13 (the schema record in the import stream)
- **Supersedes**: ADR 0041 (bare record keys with the old spelling
  still read) — its keys, and its reading courtesy, both go. /
  **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

How a JSON / JSONL record taguru reads or writes states three facts
about itself: what kind of record it is, which revision of the file
format it follows, and — when it has an identity — which one it is.
This ADR decides the rule for every such record; #937 lands it in
steps, and this first step covers the three stream-level records of a
source stream: the source file header, the `group` record, and the
`schema` record.

Not in scope: the binary snapshot image (`IMAGE_VERSION`) and
`PROMPT_VERSION`, which are not JSON columns; Prometheus metric names;
names inside the program (`Batch.source` stays what it was); and any
column that *refers* to another record (an association's `source`), which
#851 tracks separately.

## 2. Context

A source file opened with

```json
{"taguru_batch": 1, "context": "sake", "source": "docs/aomine.md"}
```

and the one key `taguru_batch` did three jobs. Its presence said "this
line opens a source file" and so also "the previous one ends here". Its
value said "file format revision 1". Neither job is what the key's name
says, and the value is not what the key names: a reader meeting
`"taguru_batch": 1` — or, after ADR 0041 took the prefix off its
siblings, `"group": 1` — reads an identifier, "group number one". The
record's actual identifier sat in a column called `source` (or `name`),
which does not say it is one.

So a kind was needed and was never declared as a kind; it was lodged
inside the version stamp, and the stamp was spelled so that it did not
look like a version. ADR 0041 kept that arrangement and made it easier
to misread.

## 3. Decision

1. **`type`** — a string column whose value is the record's kind:
   `"source"`, `"group"`, `"schema"`, and onward for the other files
   (#937). A reader decides what a line is by this value and by nothing
   else. Operation lines of a source file keep naming themselves by
   their own content key (`subject`, `alias`, `passage`, …) and carry
   no `type`.
2. **`version`** — a string column holding the file format's revision
   as a date, `"2026-09-17"`. One date covers every record type: a
   revision is a statement about the family of shapes, not about one.
   - **It may be omitted, and an omitted `version` means the revision
     of the build that is reading the file.** Not the oldest revision:
     that reading obliges a reader to keep every old interpretation
     alive forever, and turns a forgotten column into silently old
     behaviour.
   - Everything taguru writes carries `version`. Omission is for files
     a person writes.
   - A `version` that is present must equal the build's own; any other
     value is refused by name.
3. **`id`** — the one column that identifies the record. The source
   file header's `source` becomes `id`; the `group` record's `name`
   becomes `id`. A record never has two columns that look like its
   identity. A `schema` record has no `id`: a `context` has exactly one
   schema, so `context` already says which.
4. **`kind` is for a sub-classification inside a record** (an alias
   line's `concept` | `label`), never for what the record is. Records
   that use `kind` for their type move to `type` as #937 reaches them.
5. **No compatibility.** The old spellings are not read, and there is
   no transition alias. A source file, an exported stream, or a
   `*.group.jsonl` written by an earlier release is refused by this one
   and has to be produced again. The owner decided this outright
   (#851): the project is pre-1.0 and carrying two spellings of a file
   format costs more than re-exporting.

After this step:

```json
{"type": "source", "version": "2026-09-17", "id": "docs/aomine.md", "context": "sake", "create": {…}}
{"type": "group", "version": "2026-09-17", "id": "kura", "contexts": ["sake"]}
{"type": "schema", "version": "2026-09-17", "context": "sake", "mode": "warn", "closed_labels": false, "types": {…}, "relations": {…}}
```

## 4. What this step leaves for later

- **The schema document** — the body of `PUT /contexts/{name}/schema`,
  MCP's schema tools, and `{stem}.schema.json` in the data directory —
  still carries `"schema": 1`. It is one struct for all three, and the
  third is the server's own persisted state: a data directory holding
  the old shape is quarantined at boot and the server stops. Changing
  it under rule 5 strands every installed schema, which is a different
  cost from refusing an input file, so it waits for an explicit
  decision on how (or whether) an existing data directory crosses over.
- The evaluation, benchmark, consolidation, and communities files, and
  the records whose `kind` is really a `type` — the remaining steps of
  #937.
- `GET /version` reports the date under its existing `batch_formats`
  name for now; renaming that dimension belongs to #937's last step.

## 5. Reading the earlier ADRs

- ADR 0002, 0003, 0004, 0005, 0009, 0011, and 0041 spell the source
  file header `taguru_batch` and the records `taguru_group` /
  `taguru_schema` (or `group` / `schema` as version keys). Read those as
  the `type` values of §3, without editing the documents.
- ADR 0003 §10's rule — a file anyone may hand to taguru is accepted by
  equality, never by guessing — stands, applied to `version` when it is
  present. What changes is that absence is now a legal, defined state
  (§3.2) rather than a parse failure.

## 6. Consequences

- Every source file, export, and group file written before this
  release must be written again; `taguru export` from a running
  pre-upgrade server does not help, since it writes the old shape.
  Re-extract, or export with the new build from a data directory (the
  snapshot and WAL formats did not change).
- Writers spell the header one way: `src/format.rs` renders it, columns
  in reading order (`type`, `version`, `id`, `context`), for every
  producer but `taguru export`, which serializes its own struct in the
  same order.
- The LangChain SDKs write the same header and pin the same date; a
  drift between the Rust constant and theirs is a test failure on
  either side.
