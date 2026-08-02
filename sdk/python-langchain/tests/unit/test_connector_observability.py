"""Cross-connector observability (ADR 0007 §11, issue #353):
``SourceEvent``/``RunReport`` serialization, ``RunRecorder``'s last-phase-
only tally and ``retarget``, ``SourceEventSink``'s append-only JSONL
posture (mirroring ``taguru extract``'s ``DiagnosticsSink``,
src/extract.rs:2372), and the ``on_ingest_event``/``attached`` bridge that
lets a driver report the ``extracted`` phase from
``TaguruIngester``'s own ``ImportStarted`` event."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from taguru_langchain.events import (
    ChunkStarted,
    ImportCompleted,
    ImportStarted,
)
from taguru_langchain.ingest_connectors.document import Diagnostic
from taguru_langchain.ingest_connectors.observability import (
    PHASES,
    RUN_SUMMARY_KIND,
    RUN_SUMMARY_VERSION,
    SOURCE_EVENT_KIND,
    RunRecorder,
    RunReport,
    S3SyncReport,
    SourceEvent,
    SourceEventSink,
)

# ---------------------------------------------------------------------------
# SourceEvent / RunReport — serialization shape
# ---------------------------------------------------------------------------


def test_source_event_to_dict_carries_every_field_null_when_absent() -> None:
    event = SourceEvent(source="docs/manual.md", phase="discovered", elapsed_ms=0.0)
    assert event.to_dict() == {
        "kind": SOURCE_EVENT_KIND,
        "source": "docs/manual.md",
        "phase": "discovered",
        "elapsed_ms": 0.0,
        "bytes": 0,
        "parser": None,
        "diagnostic": None,
    }


def test_source_event_to_dict_nests_a_diagnostic_as_exactly_three_fields() -> None:
    diagnostic = Diagnostic(code="ocr_required", message="no usable text layer", source="a.pdf")
    event = SourceEvent(
        source="a.pdf",
        phase="skipped",
        elapsed_ms=31.4,
        bytes=2048,
        diagnostic=diagnostic,
    )
    assert event.to_dict()["diagnostic"] == {
        "code": "ocr_required",
        "message": "no usable text layer",
        "source": "a.pdf",
    }


def test_run_report_to_dict_key_set_is_pinned() -> None:
    """A literal key list in the test source, not a derived one — an added
    or removed key must edit both this list and the ADR/CHANGELOG in the
    same PR, never drift unnoticed (ADR 0007 §11's stable summary shape)."""
    report = RunReport()
    assert sorted(report.to_dict()) == sorted(
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
    assert report.to_dict()["kind"] == RUN_SUMMARY_KIND
    assert report.to_dict()["connector_run"] == RUN_SUMMARY_VERSION


def test_run_report_events_field_is_a_count_not_the_events_themselves() -> None:
    events = (
        SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0),
        SourceEvent(source="a.md", phase="parsed", elapsed_ms=5.0),
    )
    report = RunReport(events=events)
    assert report.to_dict()["events"] == 2


def test_run_report_events_jsonl_renders_one_line_per_event() -> None:
    events = (
        SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0),
        SourceEvent(source="a.md", phase="parsed", elapsed_ms=5.0),
    )
    report = RunReport(events=events)
    rendered = report.events_jsonl()
    lines = rendered.splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["phase"] == "discovered"
    assert json.loads(lines[1])["phase"] == "parsed"
    # Trailing-newline-terminated, matching what SourceEventSink.write()
    # produces incrementally on disk for the same run.
    assert rendered.endswith("\n")


def test_run_report_events_jsonl_is_empty_string_with_no_events() -> None:
    assert RunReport(events=()).events_jsonl() == ""


def test_s3_sync_report_is_the_run_report_alias() -> None:
    assert S3SyncReport is RunReport


def test_phases_matches_the_literal_type() -> None:
    assert PHASES == (
        "discovered",
        "unchanged",
        "parsed",
        "extracted",
        "imported",
        "skipped",
        "failed",
    )


def test_source_event_shares_the_rust_diagnostics_key_set() -> None:
    """Anchors ADR 0007 §8's promise that a connector diagnostic uses "the
    same three fields `taguru extract`'s own diagnostics sidecar
    (`DiagnosticsSink`/`AttemptRecord`, src/extract.rs:2372/:2566) already
    uses" — the Python-side parity check for that Rust doc comment, the
    same posture `test_events.py::
    test_attempt_failed_shares_the_rust_diagnostics_key_set` takes for
    `AttemptFailed`."""
    diagnostic = Diagnostic(code="corrupt", message="truncated", source="a.pdf")
    assert set(diagnostic.to_dict()) == {"code", "message", "source"}


# ---------------------------------------------------------------------------
# RunRecorder — last-phase-only tally, retarget, dropped/deletion counters
# ---------------------------------------------------------------------------


def test_last_phase_only_tally_counts_a_source_once_under_its_final_phase() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.record("a.md", "parsed")
    recorder.record("a.md", "extracted")
    recorder.record("a.md", "imported")
    report = recorder.finish()
    assert report.discovered == 0
    assert report.parsed == 0
    assert report.extracted == 0
    assert report.imported == 1
    assert len(report.events) == 4


def test_two_sources_tally_independently() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.discovered("b.md")
    recorder.record("a.md", "imported")
    recorder.record("b.md", "failed")
    report = recorder.finish()
    assert report.imported == 1
    assert report.failed == 1


def test_retarget_keeps_one_reference_counted_once() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("https://example.com/a")
    recorder.retarget("https://example.com/a", "https://example.com/a/")
    recorder.record("https://example.com/a/", "imported")
    report = recorder.finish()
    assert report.imported == 1
    assert report.discovered == 0
    assert report.failed == 0


def test_retarget_is_a_noop_for_an_unknown_old_source() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.retarget("never-discovered", "also-never")
    recorder.record("a.md", "imported")
    report = recorder.finish()
    assert report.imported == 1


def test_retarget_is_a_noop_when_old_equals_new() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.retarget("a.md", "a.md")
    recorder.record("a.md", "imported")
    report = recorder.finish()
    assert report.imported == 1


def test_retarget_returns_true_on_a_normal_rename() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("old")
    assert recorder.retarget("old", "new") is True


def test_retarget_onto_an_existing_different_source_does_not_clobber_it() -> None:
    """The bug this return-value contract exists to prevent: two distinct
    references (`a` and `b`) both retargeting onto the same already-
    imported `new` must never let the second one reset `new`'s tally."""
    recorder = RunRecorder(connector="test")
    recorder.discovered("a")
    assert recorder.retarget("a", "new") is True
    recorder.record("new", "imported")

    recorder.discovered("b")
    assert recorder.retarget("b", "new") is False

    report = recorder.finish()
    assert report.imported == 1
    assert report.discovered == 0
    assert report.unchanged == 0
    duplicate_events = [
        e
        for e in report.events
        if e.diagnostic is not None and e.diagnostic.code == "duplicate_source"
    ]
    assert len(duplicate_events) == 1
    assert duplicate_events[0].source == "b"


def test_duplicate_does_not_disturb_the_claimed_sources_tally() -> None:
    """The bug this method exists to prevent: a duplicate input sharing an
    already-imported source's id must never turn that source's tally back
    into `skipped`."""
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.record("a.md", "imported")
    recorder.duplicate("a.md", of="a.md")
    report = recorder.finish()
    assert report.imported == 1
    assert report.skipped == 0


def test_duplicate_is_visible_in_the_events_log_but_not_the_tally() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("https://example.com/a")
    recorder.record("https://example.com/a", "imported")
    recorder.duplicate("https://example.com/a?utm_source=x", of="https://example.com/a")
    report = recorder.finish()
    assert report.imported == 1
    assert report.discovered == 0
    duplicate_events = [
        e for e in report.events if e.source == "https://example.com/a?utm_source=x"
    ]
    assert len(duplicate_events) == 1
    assert duplicate_events[0].phase == "skipped"
    assert duplicate_events[0].diagnostic is not None
    assert duplicate_events[0].diagnostic.code == "duplicate_source"


def test_add_dropped_accumulates_across_calls() -> None:
    recorder = RunRecorder(connector="test")
    recorder.add_dropped(locators=1, sections=2)
    recorder.add_dropped(locators=3, tags=4)
    report = recorder.finish()
    assert report.locators_dropped == 4
    assert report.sections_dropped == 2
    assert report.tags_dropped == 4


def test_add_deletions_accumulates_across_calls() -> None:
    recorder = RunRecorder(connector="test")
    recorder.add_deletions(detected=2, retracted=1)
    recorder.add_deletions(detected=1, retracted=0)
    report = recorder.finish()
    assert report.deleted_detected == 3
    assert report.retracted == 1


def test_finish_interrupted_is_reported_verbatim() -> None:
    recorder = RunRecorder(connector="test")
    report = recorder.finish(interrupted=True)
    assert report.interrupted is True


def test_keep_events_false_drops_events_but_keeps_the_tally() -> None:
    recorder = RunRecorder(connector="test", keep_events=False)
    recorder.discovered("a.md")
    recorder.record("a.md", "imported")
    report = recorder.finish()
    assert report.events == ()
    assert report.imported == 1


def test_connector_name_is_reported_on_the_summary() -> None:
    recorder = RunRecorder(connector="taguru-pdf-connector")
    report = recorder.finish()
    assert report.connector == "taguru-pdf-connector"


# ---------------------------------------------------------------------------
# on_ingest_event / attached — the `extracted` phase
# ---------------------------------------------------------------------------


class _FakeIngester:
    """The only thing `attached` touches on a
    `~taguru_langchain.ingest.TaguruIngester` is its public `on_event`
    attribute — a bare stand-in avoids constructing a real ingester (LLM,
    client, checkpoint store) for a test that never ingests anything."""

    def __init__(self) -> None:
        self.on_event: object = None


def test_on_ingest_event_maps_import_started_to_extracted() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.on_ingest_event(ImportStarted(source="a.md"))
    report = recorder.finish()
    assert report.extracted == 1


def test_on_ingest_event_ignores_every_other_event_kind() -> None:
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    recorder.on_ingest_event(ChunkStarted(source="a.md", index=0, total=1))
    recorder.on_ingest_event(ImportCompleted(source="a.md", elapsed_seconds=1.0))
    report = recorder.finish()
    assert report.discovered == 1
    assert report.extracted == 0


def test_attached_records_extracted_for_the_duration_of_the_block() -> None:
    ingester = _FakeIngester()
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    with recorder.attached(ingester):  # type: ignore[arg-type]
        ingester.on_event(ImportStarted(source="a.md"))  # type: ignore[misc]
    report = recorder.finish()
    assert report.extracted == 1


def test_attached_chains_a_pre_existing_callback_never_replacing_it() -> None:
    seen: list[object] = []
    ingester = _FakeIngester()
    ingester.on_event = seen.append
    recorder = RunRecorder(connector="test")
    recorder.discovered("a.md")
    event = ImportStarted(source="a.md")
    with recorder.attached(ingester):  # type: ignore[arg-type]
        ingester.on_event(event)  # type: ignore[misc]
    assert seen == [event]
    assert recorder.finish().extracted == 1


def test_attached_restores_the_previous_callback_after_the_block() -> None:
    ingester = _FakeIngester()

    def original(event: object) -> None:
        pass

    ingester.on_event = original
    recorder = RunRecorder(connector="test")
    with recorder.attached(ingester):  # type: ignore[arg-type]
        assert ingester.on_event is not original
    assert ingester.on_event is original


def test_attached_restores_the_previous_callback_even_on_exception() -> None:
    ingester = _FakeIngester()

    def original(event: object) -> None:
        pass

    ingester.on_event = original
    recorder = RunRecorder(connector="test")
    with pytest.raises(ValueError, match="boom"):
        with recorder.attached(ingester):  # type: ignore[arg-type]
            raise ValueError("boom")
    assert ingester.on_event is original


# ---------------------------------------------------------------------------
# SourceEventSink — append-only JSONL, DiagnosticsSink's degrade posture
# ---------------------------------------------------------------------------


def test_sink_writes_one_flushed_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = SourceEventSink(path)
    sink.write(SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0))
    # Flushed, not just buffered — readable before `close()`.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["phase"] == "discovered"
    sink.write(SourceEvent(source="a.md", phase="parsed", elapsed_ms=5.0))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    sink.close()


def test_sink_truncates_an_existing_file_on_open(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("stale content from a prior run\n", encoding="utf-8")
    sink = SourceEventSink(path)
    sink.write(SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0))
    sink.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "stale content" not in lines[0]


def test_sink_open_failure_warns_once_and_the_run_continues(tmp_path: Path) -> None:
    directory_as_path = tmp_path / "not-a-file"
    directory_as_path.mkdir()
    with pytest.warns(RuntimeWarning, match="opening"):
        sink = SourceEventSink(directory_as_path)
    # No further warnings — write() silently no-ops rather than raising.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sink.write(SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0))
    sink.close()


def test_caller_supplied_text_io_is_never_closed_by_the_sink(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    stream = path.open("w", encoding="utf-8")
    sink = SourceEventSink(stream)
    sink.write(SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0))
    sink.close()
    assert not stream.closed
    stream.close()


def test_write_to_a_caller_closed_stream_degrades_instead_of_raising(tmp_path: Path) -> None:
    """A caller-supplied `TextIO` is never owned by the sink (the test
    above) — if the caller closes it first, a later `write` must degrade
    like any other write failure (one warning, then silent), not raise
    `ValueError: I/O operation on closed file` mid-sync."""
    path = tmp_path / "events.jsonl"
    stream = path.open("w", encoding="utf-8")
    sink = SourceEventSink(stream)
    stream.close()
    with pytest.warns(RuntimeWarning, match="writing to"):
        sink.write(SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0))
    # No further warnings — later writes silently no-op rather than raising.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sink.write(SourceEvent(source="b.md", phase="discovered", elapsed_ms=0.0))


def test_sink_path_property_reports_a_caller_stream_name(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        sink = SourceEventSink(stream)
        assert sink.path == str(path)


def test_sink_as_context_manager_closes_on_exit(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with SourceEventSink(path) as sink:
        sink.write(SourceEvent(source="a.md", phase="discovered", elapsed_ms=0.0))
    # Closed: reading the file after the block sees the flushed content.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


# ---------------------------------------------------------------------------
# RunRecorder + events_out end to end
# ---------------------------------------------------------------------------


def test_recorder_events_out_writes_the_same_shape_as_finish(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = RunRecorder(connector="test", events_out=path)
    recorder.discovered("a.md")
    recorder.record("a.md", "imported", parser="taguru-text-connector")
    report = recorder.finish()
    recorder.close()

    on_disk = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    from_report = [event.to_dict() for event in report.events]
    assert on_disk == from_report
    assert report.events_path == str(path)
    # Byte-for-byte, not just parsed-equal: `events_jsonl()`'s own trailing
    # newline must match what the sink wrote incrementally to `path`.
    assert report.events_jsonl() == path.read_text(encoding="utf-8")
