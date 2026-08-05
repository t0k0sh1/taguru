"""`watch_directory` (issue #414): the polling directory watcher — each
pass is a verbatim ``sync_object_storage`` run over a ``file://`` tree,
so these tests cover only what the watcher itself adds: the pass/wait
loop, cooperative stop at every stage, and the checkpoint idempotence
that makes a quiet pass cheap."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruIngester, watch_directory

from .conftest import FakeServer
from .test_checkpoints import RecordingCheckpointStore
from .test_s3_connector import EMPTY_ANSWER


def _ingester(sync_client: Taguru | None, async_client: AsyncTaguru | None) -> TaguruIngester:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return TaguruIngester(
        context="sake",
        llm=FakeListChatModel(responses=[EMPTY_ANSWER]),
        client=sync_client,
        async_client=async_client,
    )


def _watch(directory: Path, ingester: TaguruIngester, **overrides: object):  # type: ignore[no-untyped-def]
    kwargs: dict[str, object] = {
        "ingester": ingester,
        "checkpoints": RecordingCheckpointStore(),
        "interval_secs": 0.0,
    }
    kwargs.update(overrides)
    return watch_directory(directory, **kwargs)  # type: ignore[arg-type]


def test_a_missing_directory_fails_at_the_call_site_not_the_first_next(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    with pytest.raises(FileNotFoundError):
        _watch(tmp_path / "does-not-exist", _ingester(sync_client, async_client))


def test_a_negative_interval_is_refused(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    with pytest.raises(ValueError, match="interval_secs"):
        _watch(tmp_path, _ingester(sync_client, async_client), interval_secs=-1.0)


def test_each_pass_yields_a_report_and_a_quiet_pass_is_all_unchanged(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "a.md").write_bytes(b"alpha paragraph.")
    watcher = _watch(tmp_path, _ingester(sync_client, async_client))

    first = next(watcher)
    assert first.imported == 1
    assert len(fake_server.imported) == 1

    # Nothing changed: the second pass skips on the (size, mtime) listing
    # fingerprint — no re-import, and the same generator keeps going.
    second = next(watcher)
    assert second.unchanged == 1
    assert second.imported == 0
    assert len(fake_server.imported) == 1


def test_a_changed_file_is_picked_up_on_the_next_pass(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    target = tmp_path / "a.md"
    target.write_bytes(b"alpha paragraph.")
    watcher = _watch(tmp_path, _ingester(sync_client, async_client))
    next(watcher)

    target.write_bytes(b"alpha paragraph, revised for the second pass.")
    # Belt and braces against coarse filesystem mtime granularity: the
    # size already differs, but pin a distinct mtime too.
    os.utime(target, (1_700_000_000, 1_700_000_000))

    report = next(watcher)
    assert report.imported == 1
    assert len(fake_server.imported) == 2


def test_a_deleted_file_is_detected_but_never_retracted_by_default(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "a.md").write_bytes(b"alpha.")
    (tmp_path / "b.md").write_bytes(b"beta.")
    watcher = _watch(tmp_path, _ingester(sync_client, async_client))
    next(watcher)

    (tmp_path / "b.md").unlink()
    report = next(watcher)
    assert report.deleted_detected == 1
    assert report.retracted == 0
    assert fake_server.retracted == []


def test_sources_carry_the_file_uri_identity(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_bytes(b"alpha.")
    watcher = _watch(tmp_path, _ingester(sync_client, async_client))
    report = next(watcher)

    assert report.imported == 1
    [event] = [e for e in report.events if e.phase == "imported"]
    assert event.source.startswith("file://")
    assert event.source.endswith("/docs/a.md")


def test_a_stop_already_set_yields_no_pass_at_all(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "a.md").write_bytes(b"alpha.")
    stop = threading.Event()
    stop.set()
    watcher = _watch(tmp_path, _ingester(sync_client, async_client), should_stop=stop)

    assert list(watcher) == []
    assert fake_server.imported == []


def test_an_event_set_during_the_wait_ends_the_watch_promptly(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "a.md").write_bytes(b"alpha.")
    stop = threading.Event()
    # A wait long enough that only the Event ending it lets this test
    # finish — Event.wait wakes the moment it is set, no polling.
    watcher = _watch(
        tmp_path, _ingester(sync_client, async_client), interval_secs=600.0, should_stop=stop
    )

    first = next(watcher)
    assert first.imported == 1
    stop.set()
    with pytest.raises(StopIteration):
        next(watcher)


def test_a_callable_stop_is_honored_between_passes(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "a.md").write_bytes(b"alpha.")
    passes = 0

    def stop_after_one() -> bool:
        return passes >= 1

    watcher = _watch(tmp_path, _ingester(sync_client, async_client), should_stop=stop_after_one)
    for _report in watcher:
        passes += 1
    assert passes == 1


def test_dry_run_passes_touch_nothing(
    tmp_path: Path, sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    (tmp_path / "a.md").write_bytes(b"alpha.")
    watcher = _watch(tmp_path, _ingester(sync_client, async_client), dry_run=True)

    report = next(watcher)
    assert report.parsed == 1
    assert fake_server.imported == []
