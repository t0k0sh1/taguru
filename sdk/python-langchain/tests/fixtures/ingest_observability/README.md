# Connector observability golden fixtures

Pins of the JSONL event trail and run-summary JSON `sync_object_storage`
and `sync_references` emit (ADR 0007 §11, issue #353). Read/compared by
`sdk/python-langchain/tests/unit/test_connector_observability_fixtures.py`.

Deliberately **not** at the repo-root `tests/fixtures/` (`model_output/`,
`wire/`): those are cross-language pins read by Rust, Python, and
TypeScript, and `wire/` additionally gates breaking changes via
`sdk/spec/check_contract.py`. This shape has exactly one producer — this
Python module — with no Rust or TypeScript twin, so it lives beside the
tests that read it instead.

## Files

- `s3_sync.jsonl` / `s3_sync.summary.json` — `sync_object_storage` over a
  `FakeObjectStore`: imported, unchanged (pre-seeded listing fingerprint),
  failed (unsupported extension), skipped (an object that vanished between
  listing and fetch), a duplicate listing key, and `tags_dropped`. The
  duplicate-key event's `source` is IDENTICAL to the claimed object's own
  events (`s3://<bucket>/a.md` on both) — unlike `references_run.jsonl`
  below, S3's duplicate-key case has no separate input string to key the
  rejected event by. Identify it via `diagnostic.code ==
  "duplicate_source"`, not by assuming `source` alone partitions one
  contiguous per-source sub-sequence (`RunRecorder.duplicate`'s own
  docstring, `observability.py`).
- `references_dry_run.jsonl` / `references_dry_run.summary.json` —
  `sync_references(dry_run=True)`: the per-kind dry-run table (§11) —
  local file matching its file-probe checkpoint (`unchanged`), local file
  with no probe (`parsed`), unsupported extension (`skipped`), a missing
  file (`skipped`), and a URL (always `parsed`, no network access at all).
- `references_run.jsonl` / `references_run.summary.json` — a real
  `sync_references` pass: an imported local file, a duplicate reference of
  that same file, and a redirected URL (`retarget`).

## Regenerating

```sh
TAGURU_UPDATE_INGEST_OBSERVABILITY_FIXTURES=1 \
  pytest sdk/python-langchain/tests/unit/test_connector_observability_fixtures.py
```

The same escape-hatch shape `tests/http_api/contract.rs`'s
`TAGURU_UPDATE_WIRE_FIXTURES` uses. Review the diff before committing —
this flag does not itself validate anything.

## Normalization

Volatile fields are blanked to a fixed placeholder before a fixture is
written or compared, mirroring `tests/http_api/contract.rs::
normalize_volatile`'s posture (documented there, applied here by
`test_connector_observability_fixtures.py`'s own `_normalize*` helpers —
nothing reads this list back, keep both in sync by hand):

- `elapsed_ms` → `0.0`, `duration_ms` → `0.0` — different every run.
- `events_path` → `"<events>"` when present — an absolute path in whatever
  environment generated it.
- Any absolute `tmp_path` prefix or ephemeral test-server `host:port`
  appearing inside a `source` field **or inside a diagnostic `message`**
  (an OS error like `No such file or directory: '<path>'`, or this
  driver's own `duplicate()` wording) → `<tmp>` / `http://<server>`. Both
  places matter — a fixture that only normalizes `source` and misses the
  same path embedded in `message` text will fail on every machine except
  the one that generated it.
