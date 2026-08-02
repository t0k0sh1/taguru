"""Machine-readable summary fixture tests (issue #353's own explicit
acceptance criterion: "machine-readable summary の fixture テストがある").

Three golden scenarios pin the JSONL event trail and the run summary JSON
this SDK's two connector drivers (:func:`sync_object_storage`,
:func:`sync_references`) emit — one producer each, no Rust/TypeScript twin,
so the goldens live beside this SDK's own tests
(``sdk/python-langchain/tests/fixtures/ingest_observability/``), not at the
repo-root ``tests/fixtures/`` reserved for cross-language pins
(``model_output/``, ``wire/``).

Regenerate with ``TAGURU_UPDATE_INGEST_OBSERVABILITY_FIXTURES=1 pytest
sdk/python-langchain/tests/unit/test_connector_observability_fixtures.py``
— the same escape hatch shape ``tests/http_api/contract.rs``'s
``TAGURU_UPDATE_WIRE_FIXTURES`` uses. Volatile fields (``elapsed_ms``,
``duration_ms``, ``events_path``, and any absolute temp-directory/ephemeral
-port text embedded in a ``source``) are normalized to a fixed placeholder
before writing or comparing, mirroring
``tests/http_api/contract.rs::normalize_volatile``'s own posture — a slow
CI run, a different ``tmp_path``, or a different free port must never show
up as a golden diff.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruIngester
from taguru_langchain.ingest_connectors import (
    S3ObjectCheckpoint,
    sync_object_storage,
    sync_references,
)
from taguru_langchain.ingest_connectors.html import HtmlConnector
from taguru_langchain.ingest_connectors.objectstore import ObjectNotFoundError, object_fingerprint
from taguru_langchain.ingest_connectors.text import TextFileConnector

from .._httpd import Route, serve
from .._s3 import FakeObjectStore
from .test_checkpoints import RecordingCheckpointStore

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ingest_observability"
UPDATE_ENV_VAR = "TAGURU_UPDATE_INGEST_OBSERVABILITY_FIXTURES"

EMPTY_ANSWER = json.dumps({"associations": [], "aliases": [], "questions": []})


def _ingester(sync_client: Taguru, async_client: AsyncTaguru) -> TaguruIngester:
    return TaguruIngester(
        context="sake",
        llm=FakeListChatModel(responses=[EMPTY_ANSWER] * 20),
        client=sync_client,
        async_client=async_client,
    )


def _normalize(text: str, *, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _normalize_events_jsonl(jsonl: str, *, replacements: list[tuple[str, str]]) -> str:
    lines = []
    for line in jsonl.splitlines():
        event = json.loads(line)
        event["elapsed_ms"] = 0.0
        event["source"] = _normalize(event["source"], replacements=replacements)
        diagnostic = event.get("diagnostic")
        if diagnostic is not None:
            # `message` matters here too, not just `source` — an OS error
            # ("No such file or directory: '<tmp_path>/...'") or this
            # module's own `duplicate()` wording ("already claimed as
            # '<tmp_path>/...'") both embed the volatile path INSIDE the
            # message text, not only in a dedicated field.
            diagnostic["source"] = _normalize(diagnostic["source"], replacements=replacements)
            diagnostic["message"] = _normalize(diagnostic["message"], replacements=replacements)
        lines.append(json.dumps(event, sort_keys=True, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def _normalize_summary(summary: dict[str, object]) -> dict[str, object]:
    normalized = dict(summary)
    normalized["duration_ms"] = 0.0
    if normalized.get("events_path") is not None:
        normalized["events_path"] = "<events>"
    return normalized


def _compare_or_write(name: str, *, jsonl: str, summary: dict[str, object]) -> None:
    jsonl_path = FIXTURES_ROOT / f"{name}.jsonl"
    summary_path = FIXTURES_ROOT / f"{name}.summary.json"
    summary_text = json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    if os.environ.get(UPDATE_ENV_VAR):
        FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(jsonl, encoding="utf-8")
        summary_path.write_text(summary_text, encoding="utf-8")
        return

    assert jsonl_path.exists(), f"missing golden {jsonl_path} — regenerate with {UPDATE_ENV_VAR}=1"
    assert summary_path.exists(), (
        f"missing golden {summary_path} — regenerate with {UPDATE_ENV_VAR}=1"
    )
    assert jsonl == jsonl_path.read_text(encoding="utf-8")
    assert summary_text == summary_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The stable summary key set — a literal list, not a derived one, so an
# added/removed key is a deliberate two-place edit (here and the golden).
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = sorted(
    [
        "kind",
        "connector_run",
        "connector",
        "duration_ms",
        "interrupted",
        "discovered",
        "unchanged",
        "parsed",
        "extracted",
        "imported",
        "skipped",
        "failed",
        "locators_dropped",
        "sections_dropped",
        "tags_dropped",
        "deleted_detected",
        "retracted",
        "events",
        "events_path",
    ]
)


def test_summary_key_set_matches_every_golden() -> None:
    for path in sorted(FIXTURES_ROOT.glob("*.summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert sorted(summary) == _SUMMARY_KEYS, path


# ---------------------------------------------------------------------------
# Golden 1: sync_object_storage — imported, unchanged (pre-seeded
# fingerprint), failed (unsupported extension), skipped (vanished object),
# a duplicate listing key, and tags_dropped.
# ---------------------------------------------------------------------------


def test_s3_sync_golden(sync_client: Taguru, async_client: AsyncTaguru) -> None:
    store = FakeObjectStore("golden-bucket")
    same_when = datetime(2026, 1, 1, tzinfo=timezone.utc)

    store.put("a.md", b"alpha content")
    store.put("b.md", b"beta content", last_modified=same_when)  # pre-seeded -> unchanged
    store.put("c.zip", b"PK\x03\x04...")  # no installed connector -> failed
    store.put("vanished.md", b"gone by fetch time")
    store.put("many-tags.md", b"tagged content", tags={f"t{i:02d}": "v" for i in range(40)})

    checkpoints = RecordingCheckpointStore()
    meta_b = next(m for m in store.list("") if m.key == "b.md")
    fingerprint_b = object_fingerprint(meta_b)
    assert fingerprint_b is not None
    S3ObjectCheckpoint(checkpoints).save(f"{store.base_uri}/b.md", fingerprint_b)

    real_get = store.get

    def get_with_one_vanished(key: str, *, version_id: str | None = None) -> object:
        if key == "vanished.md":
            raise ObjectNotFoundError("gone by fetch time")
        return real_get(key, version_id=version_id)

    store.get = get_with_one_vanished  # type: ignore[method-assign]

    real_list = store.list

    def list_with_one_duplicate(prefix: str) -> object:
        metas = list(real_list(prefix))
        duplicate = next(m for m in metas if m.key == "a.md")
        return iter([duplicate, *metas])

    store.list = list_with_one_duplicate  # type: ignore[method-assign]

    report = sync_object_storage(
        store, "", ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )

    replacements = [(store.base_uri, "s3://<bucket>")]
    jsonl = _normalize_events_jsonl(report.events_jsonl(), replacements=replacements)
    summary = _normalize_summary(report.to_dict())
    _compare_or_write("s3_sync", jsonl=jsonl, summary=summary)


# ---------------------------------------------------------------------------
# Golden 2: sync_references(dry_run=True) — the per-kind dry-run table:
# local unchanged (matching probe), local parsed (no probe), unsupported
# extension (skipped), missing file (skipped), URL (always parsed, no
# network).
# ---------------------------------------------------------------------------


def test_references_dry_run_golden(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    probed = tmp_path / "probed.md"
    probed.write_text("paragraph one.", encoding="utf-8")
    unprobed = tmp_path / "unprobed.md"
    unprobed.write_text("paragraph two.", encoding="utf-8")
    unsupported = tmp_path / "unsupported.exe"
    unsupported.write_text("binary-ish", encoding="utf-8")
    missing = tmp_path / "missing.md"

    checkpoints = RecordingCheckpointStore()
    # A real run over `probed` alone, to populate its file-probe checkpoint
    # — `unprobed` deliberately never runs for real, so its dry-run stays
    # `parsed`.
    sync_references(
        [str(probed)], ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )

    report = sync_references(
        [
            str(probed),
            str(unprobed),
            str(unsupported),
            str(missing),
            "http://127.0.0.1:1/unreachable.html",
        ],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        dry_run=True,
    )

    replacements = [(str(tmp_path), "<tmp>")]
    jsonl = _normalize_events_jsonl(report.events_jsonl(), replacements=replacements)
    summary = _normalize_summary(report.to_dict())
    _compare_or_write("references_dry_run", jsonl=jsonl, summary=summary)


# ---------------------------------------------------------------------------
# Golden 3: sync_references — a real run: an imported local file, a
# duplicate reference of that same file, and a redirected URL (retarget).
# ---------------------------------------------------------------------------


def test_references_run_golden(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = tmp_path / "manual.md"
    reference.write_text("paragraph one.", encoding="utf-8")

    with serve(
        {
            "/old.html": Route(location="/new.html"),
            "/new.html": Route(body=b"<html><body><p>paragraph two.</p></body></html>"),
        }
    ) as server:
        report = sync_references(
            [str(reference), str(reference), f"{server.base_url}/old.html"],
            ingester=_ingester(sync_client, async_client),
            checkpoints=RecordingCheckpointStore(),
            connectors=(TextFileConnector(), HtmlConnector(allow_private_networks=True)),
        )
        replacements = [
            (str(tmp_path), "<tmp>"),
            (server.base_url, "http://<server>"),
        ]

    jsonl = _normalize_events_jsonl(report.events_jsonl(), replacements=replacements)
    summary = _normalize_summary(report.to_dict())
    _compare_or_write("references_run", jsonl=jsonl, summary=summary)
