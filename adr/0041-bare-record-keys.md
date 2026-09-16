# 0041. Record keys drop the `taguru_` prefix: `group`, `schema`, `eval`, `evaluation`, `evaluate_thresholds`, `consolidation`, `benchmark_*`

- **Status**: Accepted
- **Date**: 2026-09-16
- **Issue**: #933 (implements a decision recorded in #851)
- **Related**: ADR 0003 §10 (equality-versus-range acceptance — unchanged
  by this rename; only the key's spelling moves), ADR 0005 (the wire
  contract fixtures that carry an import stream), ADR 0009 §13 (the
  schema record in the import stream), ADR 0040 (the previous
  terminology wave)
- **Supersedes**: nothing. / **Superseded by**: —

Once Accepted, this document's Decision is immutable: a changed decision gets a
new `adr/000N-*.md` that names this one in *Supersedes*, never an edit here.

## 1. Scope

The JSON key that marks a record's kind and carries its format version,
in every NDJSON stream and JSON file taguru reads or writes, when that
key was spelled with a `taguru_` prefix. Twelve keys:

| Before | After | Where |
|---|---|---|
| `taguru_group` | `group` | import stream, `taguru export`'s `*.group.jsonl` |
| `taguru_schema` | `schema` | import stream (ADR 0009 §13) |
| `taguru_eval` | `eval` | eval-set files (`taguru evaluate`, `taguru benchmark search`) |
| `taguru_evaluation` | `evaluation` | `taguru evaluate`'s report |
| `taguru_evaluate_thresholds` | `evaluate_thresholds` | `taguru evaluate --thresholds` |
| `taguru_consolidation` | `consolidation` | the consolidation manifest stored as a passage |
| `taguru_benchmark_manifest` / `_runs` / `_models` / `_measurements` / `_differences` / `_retrieval` | `benchmark_manifest` / … | `taguru benchmark`'s artifacts |

Not in scope, deliberately:

- **`taguru_batch`**, the source file's header line. Its record already
  uses `source` for the origin it cites, so the bare noun is taken, and
  the right shape for that header is undecided (#851). It keeps its
  prefix until a separate decision.
- **`taguru_communities`**, the communities manifest. That record already
  carries a `communities` list, so the bare noun collides. Same
  deferral.
- **Prometheus metric names** (`taguru_requests_total`, …). The prefix
  there is the exporter namespace convention that keeps taguru's series
  apart from every other exporter's on a shared `/metrics`.

## 2. Context

Every record kind in taguru's streams is identified by one key whose
name is the kind and whose value is the format version: `{"passage":
…}`, `{"alias": …}`, and on disk the schema file's `{"schema": 1,
"mode": …}` (src/schema.rs). The stream-level records — group, schema —
and the standalone artifacts — eval sets, evaluation reports,
thresholds, benchmark files — followed the same rule but with a
`taguru_` prefix, a habit inherited from `taguru_batch` (ADR 0002)
rather than a decision anyone recorded. The prefix says nothing the
file's own location and shape do not already say, and #851's
terminology pass, which asks that every name carry exactly one meaning
and no ornament, flagged it.

The prefix is also the reason the import stream's schema record and
the on-disk schema file — the same content — spelled their marker
differently (`taguru_schema` vs `schema`).

## 3. Decision

1. **The twelve keys in §1 lose their prefix.** Writers emit the bare
   key only. The value, its meaning, and the acceptance rule (equality
   for hand-written and cross-program files, ADR 0003 §10's range for
   the benchmark manifest and the evaluation report) do not change.
2. **Readers accept the old spelling for a transition.** Every reader
   of a §1 record takes `taguru_<key>` as an alias of `<key>`: serde
   `alias` on the struct field, or a second key lookup where the
   reader walks a `serde_json::Value`. A file or stream written by any
   earlier release loads unchanged. The alias is a reading courtesy,
   not a second format: a writer that still emits the prefixed key is
   writing an old format, and nothing here promises the alias outlives
   the next major.
3. **Error wording follows the key.** "`schema 2 is not a version this
   taguru reads`", "`eval must be 1`", and the rest name the bare key,
   whichever spelling the offending line used.
4. **`taguru_batch` and `taguru_communities` stay as they are** (§1),
   and this ADR is not the place their eventual shape gets decided.

## 4. Consequences

- Import streams, exported `*.group.jsonl` files, eval sets, evaluation
  reports, thresholds files, and benchmark artifacts written by this
  release carry the bare keys. Tools outside taguru that pattern-match
  the old keys need the same alias.
- ADR 0005's wire fixtures that carry a schema record
  (`tests/fixtures/wire/http/import_with_schema.json`) change spelling;
  the request shape is otherwise identical.
- Every earlier ADR that spells one of the §1 keys with its prefix is
  describing the same record under its old name. Read `taguru_group` as
  `group`, and so on, without editing those documents.
- A regression test per reader pins the alias: an old-spelling
  record must parse, and its version check must still fire.
