"""sync_references (ADR 0007 §11, issue #353): the cross-connector local-
file/URL driver — dispatch that never lets a URL fall into a local-path
connector's ``open()``, the per-kind ``dry_run`` table, the
enumerate-before-fetch ordering, ``duplicate_source``, redirect
``retarget``, and interruption/failure reporting — exercised against real
:class:`~taguru_langchain.ingest_connectors.text.TextFileConnector` and
:class:`~taguru_langchain.ingest_connectors.html.HtmlConnector` instances,
a real ``tests/_httpd.py`` server for URL fetches, and the same in-memory
``FakeServer`` every other ingest test in this suite uses."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruIngester
from taguru_langchain.ingest import IngestOutcome
from taguru_langchain.ingest_connectors.document import (
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    FingerprintInputs,
)
from taguru_langchain.ingest_connectors.html import HtmlConnector
from taguru_langchain.ingest_connectors.protocol import Connector
from taguru_langchain.ingest_connectors.references import (
    default_connectors,
    plan_references,
    sync_references,
)
from taguru_langchain.ingest_connectors.text import TextFileConnector

from .._httpd import Route, serve
from .test_checkpoints import RecordingCheckpointStore

EMPTY_ANSWER = json.dumps({"associations": [], "aliases": [], "questions": []})


def _ingester(
    sync_client: Taguru | None, async_client: AsyncTaguru | None, **overrides: object
) -> TaguruIngester:
    return TaguruIngester(
        context="sake",
        llm=FakeListChatModel(responses=[EMPTY_ANSWER] * 50),
        client=sync_client,
        async_client=async_client,
        **overrides,  # type: ignore[arg-type]
    )


def _write(tmp_path: Path, name: str, data: str) -> str:
    path = tmp_path / name
    path.write_text(data, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# plan_references — classification and dispatch, before any fetch
# ---------------------------------------------------------------------------


def test_local_extension_dispatches_to_the_text_connector(tmp_path: Path) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    plans = plan_references([reference], connectors=default_connectors())
    assert len(plans) == 1
    assert isinstance(plans[0].connector, TextFileConnector)
    assert plans[0].kind == "path"
    assert plans[0].diagnostic is None


def test_a_markdown_shaped_url_is_never_claimed_by_the_text_connector() -> None:
    """The bug `references.py` exists to prevent: TextFileConnector.supports
    only checks the suffix, so a naive `supports()`-first dispatch would
    hand this URL to `open()` as though it were a local path."""
    plans = plan_references(["https://example.com/readme.md"], connectors=default_connectors())
    assert len(plans) == 1
    assert plans[0].kind == "url"
    assert isinstance(plans[0].connector, HtmlConnector)
    assert not isinstance(plans[0].connector, TextFileConnector)


def test_object_storage_scheme_is_reported_unsupported_naming_the_other_driver() -> None:
    plans = plan_references(
        ["s3://bucket/key.pdf", "file:///tmp/x.pdf"], connectors=default_connectors()
    )
    assert len(plans) == 2
    for plan in plans:
        assert plan.connector is None
        assert plan.diagnostic is not None
        assert plan.diagnostic.code == "unsupported_format"
        assert "sync_object_storage" in plan.diagnostic.message


def test_unrecognized_local_extension_is_unsupported_format(tmp_path: Path) -> None:
    reference = _write(tmp_path, "a.exe", "binary-ish")
    plans = plan_references([reference], connectors=default_connectors())
    assert plans[0].connector is None
    assert plans[0].diagnostic is not None
    assert plans[0].diagnostic.code == "unsupported_format"


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    plans = plan_references([str(tmp_path / "never-existed.md")], connectors=default_connectors())
    assert plans[0].connector is None
    assert plans[0].diagnostic is not None
    assert plans[0].diagnostic.code == "unreadable"


def test_a_directory_reference_is_unreadable(tmp_path: Path) -> None:
    plans = plan_references([str(tmp_path)], connectors=default_connectors())
    assert plans[0].connector is None
    assert plans[0].diagnostic is not None
    assert plans[0].diagnostic.code == "unreadable"


def test_duplicate_reference_is_flagged_and_the_first_keeps_its_connector(
    tmp_path: Path,
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    plans = plan_references([reference, reference], connectors=default_connectors())
    assert len(plans) == 2
    assert plans[0].connector is not None
    assert plans[0].diagnostic is None
    assert plans[1].connector is None
    assert plans[1].diagnostic is not None
    assert plans[1].diagnostic.code == "duplicate_source"


def test_two_urls_differing_only_by_a_denylisted_query_param_are_duplicates() -> None:
    plans = plan_references(
        [
            "https://example.com/report.html?token=abc",
            "https://example.com/report.html?token=xyz",
        ],
        connectors=default_connectors(),
    )
    assert plans[1].diagnostic is not None
    assert plans[1].diagnostic.code == "duplicate_source"


def test_source_bytes_is_the_files_own_size(tmp_path: Path) -> None:
    reference = _write(tmp_path, "a.md", "twelve bytes")
    plans = plan_references([reference], connectors=default_connectors())
    assert plans[0].source_bytes == len(b"twelve bytes")


def test_source_bytes_is_zero_for_a_url() -> None:
    plans = plan_references(["https://example.com/a.html"], connectors=default_connectors())
    assert plans[0].source_bytes == 0


# ---------------------------------------------------------------------------
# dry_run — per-kind rules (ADR 0007 §11)
# ---------------------------------------------------------------------------


class _SpyConnector:
    """Wraps a real connector, recording every `read()` call — used to
    assert dry-run never calls `read()` at all."""

    def __init__(self, delegate: Connector) -> None:
        self._delegate = delegate
        self.read_calls: list[str] = []

    @property
    def parser(self) -> str:
        return self._delegate.parser

    @property
    def parser_version(self) -> str:
        return self._delegate.parser_version

    @property
    def parse_options_digest(self) -> str:
        return self._delegate.parse_options_digest

    def supports(self, reference: str) -> bool:
        return self._delegate.supports(reference)

    def read(self, reference: str) -> ConnectorDocument:
        self.read_calls.append(reference)
        return self._delegate.read(reference)


def _spy_connectors() -> tuple[Sequence[Connector], _SpyConnector]:
    spy = _SpyConnector(TextFileConnector())
    return (spy, HtmlConnector()), spy


def test_dry_run_local_file_with_no_probe_reports_parsed_and_never_reads(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    connectors, spy = _spy_connectors()
    checkpoints = RecordingCheckpointStore()
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=connectors,
        dry_run=True,
    )
    assert report.parsed == 1
    assert report.unchanged == 0
    assert spy.read_calls == []
    # A dry-run probe LOOKUP is expected (that's how "unchanged" gets
    # decided at all) — the "touches nothing" contract is about never
    # WRITING to the checkpoint store under dry-run.
    assert ("save", f"file-probe:{reference}") not in checkpoints.log
    assert ("save", f"connector:{reference}") not in checkpoints.log


def test_dry_run_local_file_matching_probe_reports_unchanged(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    connectors, spy = _spy_connectors()
    checkpoints = RecordingCheckpointStore()

    # A real (non-dry) run first, to populate the file-probe checkpoint.
    sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=connectors,
    )
    spy.read_calls.clear()

    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=connectors,
        dry_run=True,
    )
    assert report.unchanged == 1
    assert report.parsed == 0
    assert spy.read_calls == []


def test_dry_run_local_file_with_stale_mtime_reports_parsed(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    connectors, spy = _spy_connectors()
    checkpoints = RecordingCheckpointStore()
    sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=connectors,
    )
    spy.read_calls.clear()

    # Touch the file with new content — a different size, so the probe
    # can never match by accident.
    Path(reference).write_text("paragraph one, edited.", encoding="utf-8")

    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=connectors,
        dry_run=True,
    )
    assert report.parsed == 1
    assert report.unchanged == 0
    assert spy.read_calls == []


def test_dry_run_local_file_with_changed_options_digest_reports_parsed(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    checkpoints = RecordingCheckpointStore()
    sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=(TextFileConnector(extract_headings=True), HtmlConnector()),
    )
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        connectors=(TextFileConnector(extract_headings=False), HtmlConnector()),
        dry_run=True,
    )
    assert report.parsed == 1
    assert report.unchanged == 0


def test_dry_run_file_vanishing_between_plan_and_probe_reports_parsed(
    tmp_path: Path,
    sync_client: Taguru,
    async_client: AsyncTaguru,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The narrow TOCTOU `_dry_run_plan`'s own `OSError` branch guards:
    the file existed when `plan_references` stat'd it, but is gone by the
    time the dry-run probe stats it again a moment later."""
    reference = _write(tmp_path, "a.md", "paragraph one.")
    real_stat = os.stat
    calls = {"n": 0}

    def flaky_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        calls["n"] += 1
        if calls["n"] > 1:
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", flaky_stat)
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
        dry_run=True,
    )
    assert report.parsed == 1


def test_dry_run_url_reports_parsed_and_makes_no_network_request(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    # Port 1 refuses connections immediately on any sane host — if
    # `sync_references` ever attempted a fetch under dry_run, this would
    # raise instead of quietly reporting `parsed`.
    report = sync_references(
        ["http://127.0.0.1:1/unreachable.html"],
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
        dry_run=True,
    )
    assert report.parsed == 1
    assert report.unchanged == 0
    assert report.failed == 0


def test_dry_run_vanished_file_is_skipped_unreadable_not_a_crash(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """`plan_references` itself stats every path reference (dry-run or
    not, ADR 0007 §11: cheap metadata reads are fine even under dry-run)
    — a file that vanished before the plan is built is caught there,
    reported `skipped`/`unreadable`, never reaching `_dry_run_plan`'s own
    (much narrower) vanished-between-plan-and-probe race at all."""
    reference = _write(tmp_path, "a.md", "paragraph one.")
    checkpoints = RecordingCheckpointStore()
    sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
    )
    Path(reference).unlink()
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        dry_run=True,
    )
    assert report.skipped == 1
    skipped = [e for e in report.events if e.phase == "skipped"][0]
    assert skipped.diagnostic is not None
    assert skipped.diagnostic.code == "unreadable"


# ---------------------------------------------------------------------------
# Enumeration before the first fetch (ADR 0007 §11)
# ---------------------------------------------------------------------------


def test_every_discovered_event_lands_before_the_first_read(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    references = [_write(tmp_path, f"{i}.md", f"paragraph {i}.") for i in range(3)]
    connectors, spy = _spy_connectors()
    discovered_before_any_read: list[bool] = []

    class _OrderTrackingSpy(_SpyConnector):
        def read(self, reference: str) -> ConnectorDocument:
            discovered_before_any_read.append(len(self.read_calls) == 0)
            return super().read(reference)

    ordered = _OrderTrackingSpy(TextFileConnector())
    report = sync_references(
        references,
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
        connectors=(ordered, HtmlConnector()),
    )
    assert report.imported == 3
    discovered_events = [e for e in report.events if e.phase == "discovered"]
    assert len(discovered_events) == 3
    # All three `discovered` events precede all three `read()` calls — true
    # by construction (plan_references + the discover loop run to
    # completion before the process loop starts), asserted here via the
    # event ordering: every `discovered` line for EVERY source appears
    # before the first `parsed`/`imported` line for ANY of them.
    first_non_discovered_index = next(
        i for i, e in enumerate(report.events) if e.phase != "discovered"
    )
    assert first_non_discovered_index == 3


# ---------------------------------------------------------------------------
# Real run, checkpoint reuse, duplicate handling, redirects
# ---------------------------------------------------------------------------


def test_real_run_then_rerun_is_unchanged_then_dry_run_is_also_unchanged(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    checkpoints = RecordingCheckpointStore()

    first = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
    )
    assert first.imported == 1

    second = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
    )
    assert second.unchanged == 1
    assert second.imported == 0

    third = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        dry_run=True,
    )
    assert third.unchanged == 1


def test_duplicate_reference_is_read_exactly_once(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    connectors, spy = _spy_connectors()
    report = sync_references(
        [reference, reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
        connectors=connectors,
    )
    assert spy.read_calls == [reference]
    # The winning occurrence's own tally is undisturbed by the duplicate.
    assert report.imported == 1
    assert any(
        e.diagnostic is not None and e.diagnostic.code == "duplicate_source" for e in report.events
    )


def test_redirected_url_retargets_without_double_counting(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    with serve(
        {
            "/old.html": Route(location="/new.html"),
            "/new.html": Route(body=b"<html><body><p>paragraph one.</p></body></html>"),
        }
    ) as server:
        report = sync_references(
            [f"{server.base_url}/old.html"],
            ingester=_ingester(sync_client, async_client),
            checkpoints=RecordingCheckpointStore(),
            # The test server is on 127.0.0.1 — HtmlConnector's default
            # SSRF guard blocks private/internal destinations, so this
            # test explicitly opts back in, same as
            # test_html_connector_fetch.py.
            connectors=(TextFileConnector(), HtmlConnector(allow_private_networks=True)),
        )
    assert report.imported == 1
    assert report.discovered == 0
    assert report.failed == 0
    # `discovered` legitimately kept the PRE-redirect URL (that's the
    # honest history retarget() preserves in the JSONL trail) — but every
    # later phase, and the tally itself, moved to the POST-redirect one.
    by_phase = {e.phase: e.source for e in report.events}
    assert by_phase["discovered"] == f"{server.base_url}/old.html"
    assert by_phase["imported"] == f"{server.base_url}/new.html"


def test_two_urls_redirecting_to_the_same_final_url_do_not_corrupt_the_tally(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The post-fetch twin of `test_duplicate_reference_is_read_exactly_once`
    — two DIFFERENT pre-fetch references (so `plan_references`'s own
    pre-fetch dedup cannot catch this) both redirect to the SAME final
    URL. Naively overwriting the winning source's state when the second
    one's `retarget` runs would silently turn its real `imported` outcome
    back into `unchanged`/`discovered` in the final tally."""
    with serve(
        {
            "/old-a.html": Route(location="/new.html"),
            "/old-b.html": Route(location="/new.html"),
            "/new.html": Route(body=b"<html><body><p>paragraph one.</p></body></html>"),
        }
    ) as server:
        report = sync_references(
            [f"{server.base_url}/old-a.html", f"{server.base_url}/old-b.html"],
            ingester=_ingester(sync_client, async_client),
            checkpoints=RecordingCheckpointStore(),
            connectors=(TextFileConnector(), HtmlConnector(allow_private_networks=True)),
        )
        new_url = f"{server.base_url}/new.html"

    # The real outcome survives: exactly one import, under the shared
    # final URL — never silently downgraded to unchanged/discovered by
    # the second reference's own retarget.
    assert report.imported == 1
    assert report.discovered == 0
    assert report.unchanged == 0
    assert report.failed == 0

    events_by_source: dict[str, list[str]] = {}
    for event in report.events:
        events_by_source.setdefault(event.source, []).append(event.phase)
    assert events_by_source[new_url][-1] == "imported"
    # The second (losing) reference is visible as a duplicate of the
    # winning final URL, not as a second competing history under it.
    duplicate_events = [
        e
        for e in report.events
        if e.diagnostic is not None and e.diagnostic.code == "duplicate_source"
    ]
    assert len(duplicate_events) == 1
    assert duplicate_events[0].diagnostic is not None
    assert new_url in duplicate_events[0].diagnostic.message


# ---------------------------------------------------------------------------
# should_stop, ingest failure, connector-raised exceptions
# ---------------------------------------------------------------------------


def test_should_stop_interrupts_and_leaves_the_tail_discovered(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    """The stop signal must fire strictly AFTER the first document's own
    ingest completes — `ingest_text` checks the same `should_stop` at its
    own chunk boundaries, so setting it any earlier (e.g. mid-`read()`)
    would interrupt the FIRST document instead of merely stopping the
    driver before the second one starts."""
    references = [_write(tmp_path, f"{i}.md", f"paragraph {i}.") for i in range(3)]
    stop_event = threading.Event()

    def on_event(event: object) -> None:
        if getattr(event, "kind", None) == "import_completed":
            stop_event.set()

    ingester = _ingester(sync_client, async_client, on_event=on_event)
    report = sync_references(
        references,
        ingester=ingester,
        checkpoints=RecordingCheckpointStore(),
        should_stop=stop_event,
    )
    assert report.interrupted is True
    assert report.imported == 1
    assert report.discovered == 2


class _StubIngester:
    """A minimal stand-in for `TaguruIngester` — `sync_references` only
    ever calls `.on_event` (via `RunRecorder.attached`, get/set) and
    `.ingest_text` (via `ingest_connector_document`) on it, so a full
    ingester (LLM, client, checkpoint store) is unnecessary overhead for a
    test whose only interest is what `sync_references` does with a
    controlled `IngestOutcome`."""

    def __init__(self, outcome: IngestOutcome) -> None:
        self.on_event: Callable[[object], None] | None = None
        self._outcome = outcome
        self.calls = 0

    def ingest_text(self, text: str, **kwargs: object) -> IngestOutcome:
        self.calls += 1
        return self._outcome


def test_ingest_failure_is_reported_failed_and_the_run_continues(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.md", "paragraph one.")
    b = _write(tmp_path, "b.md", "paragraph two.")
    stub = _StubIngester(IngestOutcome(source="doesn't matter", ok=False, error="model exploded"))
    report = sync_references(
        [a, b],
        ingester=stub,  # type: ignore[arg-type]
        checkpoints=RecordingCheckpointStore(),
    )
    assert report.failed == 2
    assert report.imported == 0
    assert stub.calls == 2
    failed_events = [e for e in report.events if e.phase == "failed"]
    assert all(
        e.diagnostic is not None and "model exploded" in e.diagnostic.message for e in failed_events
    )


class _RaisingIngester:
    """An ingester whose `ingest_text` raises — unlike a connector's own
    `read()`, `sync_references` does not wrap the bridge call in a
    try/except, so this exercises `RunRecorder` being managed by a `with`
    block: `events_out`'s file handle must still close."""

    def __init__(self) -> None:
        self.on_event: Callable[[object], None] | None = None

    def ingest_text(self, text: str, **kwargs: object) -> IngestOutcome:
        raise RuntimeError("ingester exploded")


def test_events_out_is_closed_even_when_ingest_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import taguru_langchain.ingest_connectors.references as references_module

    reference = _write(tmp_path, "a.md", "paragraph one.")
    events_path = tmp_path / "events.jsonl"

    closed = []
    original_close = references_module.RunRecorder.close

    def recording_close(self: object) -> None:
        closed.append(True)
        original_close(self)  # type: ignore[arg-type]

    monkeypatch.setattr(references_module.RunRecorder, "close", recording_close)

    with pytest.raises(RuntimeError, match="ingester exploded"):
        sync_references(
            [reference],
            ingester=_RaisingIngester(),  # type: ignore[arg-type]
            checkpoints=RecordingCheckpointStore(),
            events_out=events_path,
        )
    assert closed == [True]


class _RaisingConnector:
    parser = "raising-connector"
    parser_version = "1.0.0"
    parse_options_digest = "x"

    def supports(self, reference: str) -> bool:
        return reference.endswith(".boom")

    def read(self, reference: str) -> ConnectorDocument:
        raise RuntimeError("connector bug")


def test_a_connector_that_raises_is_reported_failed_not_propagated(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = str(tmp_path / "a.boom")
    Path(reference).write_text("irrelevant", encoding="utf-8")
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
        connectors=(_RaisingConnector(),),
    )
    assert report.failed == 1
    failed_events = [e for e in report.events if e.phase == "failed"]
    assert failed_events[0].diagnostic is not None
    assert "connector bug" in failed_events[0].diagnostic.message


def test_diagnostics_only_document_is_reported_failed(tmp_path: Path) -> None:
    """A connector that returns an empty-text, diagnostics-only document
    (e.g. `ocr_required`) never reaches the network — `bridge.py`'s own
    posture — and this driver reports it `failed`, not `skipped`, since
    the object WAS read; it simply had nothing usable in it."""

    class _EmptyTextConnector:
        parser = "empty"
        parser_version = "1.0.0"
        parse_options_digest = "x"

        def supports(self, reference: str) -> bool:
            return True

        def read(self, reference: str) -> ConnectorDocument:
            return ConnectorDocument(
                source=reference,
                text="",
                metadata=ConnectorMetadata(origin_uri=reference, display_name=reference),
                fingerprint_inputs=FingerprintInputs(
                    raw_content_sha256="x",
                    parser=self.parser,
                    parser_version=self.parser_version,
                    parse_options_digest=self.parse_options_digest,
                ),
                diagnostics=(Diagnostic(code="ocr_required", message="no text", source=reference),),
            )

    reference = str(tmp_path / "scan.pdf")
    Path(reference).write_text("irrelevant", encoding="utf-8")
    report = sync_references(
        [reference],
        ingester=_StubIngester(IngestOutcome(source=reference, ok=True)),  # type: ignore[arg-type]
        checkpoints=RecordingCheckpointStore(),
        connectors=(_EmptyTextConnector(),),
    )
    assert report.failed == 1
    assert report.imported == 0


# ---------------------------------------------------------------------------
# `extracted` — exactly once per imported source
# ---------------------------------------------------------------------------


def test_extracted_is_recorded_once_per_imported_source(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
    )
    assert report.imported == 1
    extracted_events = [e for e in report.events if e.phase == "extracted"]
    assert len(extracted_events) == 1
    assert extracted_events[0].source == reference


def test_a_preexisting_on_event_callback_still_fires_during_sync(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    seen: list[str] = []
    ingester = _ingester(sync_client, async_client, on_event=lambda event: seen.append(event.kind))
    sync_references([reference], ingester=ingester, checkpoints=RecordingCheckpointStore())
    assert "import_started" in seen
    # The caller's own callback is restored afterward, not left swapped.
    assert ingester.on_event is not None


# ---------------------------------------------------------------------------
# events_out — the JSONL sidecar end to end
# ---------------------------------------------------------------------------


def test_events_out_writes_every_phase_transition(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    reference = _write(tmp_path, "a.md", "paragraph one.")
    events_path = tmp_path / "events.jsonl"
    report = sync_references(
        [reference],
        ingester=_ingester(sync_client, async_client),
        checkpoints=RecordingCheckpointStore(),
        events_out=events_path,
    )
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    phases = [line["phase"] for line in lines if line["source"] == reference]
    assert phases == ["discovered", "parsed", "extracted", "imported"]
    assert report.events_path == str(events_path)
