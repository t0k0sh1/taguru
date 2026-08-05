"""The local directory watcher (ADR 0007 §13's deferred third connector,
issue #414): repeated, checkpoint-idempotent passes of the existing
object-storage sync over a ``file://`` directory tree.

Deliberately a POLLING loop over :func:`~taguru_langchain.
ingest_connectors.s3.sync_object_storage` with a
:class:`~taguru_langchain.ingest_connectors.objectstore.FileObjectStore`,
not a native filesystem-event subscription: inotify/FSEvents/
ReadDirectoryChangesW each behave differently (and all of them miss
events across network mounts and container bind mounts), a dependency
like ``watchdog`` would be this package's first non-parser runtime
dependency, and the sync it would trigger is already idempotent — an
event's only value here would be shaving latency off a poll interval.
The two-layer checkpoint makes the poll cheap in the steady state: an
unchanged file is skipped on its ``(size, mtime)`` listing fingerprint
without ever being opened, so a pass over a quiet tree costs one
directory walk. Each pass IS a ``sync_object_storage`` run, verbatim —
same source ids (``file:///abs/path/…``), same events and
:class:`~taguru_langchain.ingest_connectors.observability.RunReport`,
same deletion policy (default ``"report"``: a deleted file is counted,
never retracted, #217's no-destructive-sync-by-default requirement),
same ``dry_run``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TextIO

from ..checkpoints import CheckpointStore
from ..ingest import TaguruIngester, _normalize_stop
from .objectstore import FileObjectStore
from .observability import RunReport
from .s3 import DeletionPolicy, S3Connector, sync_object_storage

# How often a callable `should_stop` is re-checked while waiting out the
# interval between passes. A `threading.Event` needs no polling at all —
# `Event.wait` wakes the moment it is set — so this granularity only
# bounds the stop latency of the plain-callable form.
_STOP_POLL_SECS = 0.2


def watch_directory(
    directory: str | os.PathLike[str],
    *,
    ingester: TaguruIngester,
    checkpoints: CheckpointStore,
    interval_secs: float = 30.0,
    connector: S3Connector | None = None,
    deletion_policy: DeletionPolicy = "report",
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
    events_out: str | Path | TextIO | None = None,
) -> Iterator[RunReport]:
    """Watches ``directory`` by running one
    :func:`~taguru_langchain.ingest_connectors.s3.sync_object_storage`
    pass, yielding its :class:`RunReport`, waiting ``interval_secs``, and
    repeating — until ``should_stop`` says stop or the caller simply
    stops consuming the iterator.

    A generator on purpose: the caller owns the loop. ``for report in
    watch_directory(...)`` observes every pass as it completes,
    ``break`` ends the watch, and nothing accumulates — a watcher that
    returned ``list[RunReport]`` would grow without bound over a
    long-running watch. ``should_stop`` exists for the cooperative case
    (another thread deciding to stop a consumer blocked in ``next()``):
    it is checked before each pass, DURING each pass (``sync_object_
    storage`` checks it between objects), and continuously through the
    inter-pass wait — a :class:`threading.Event` ends the wait the
    moment it is set.

    The directory must exist when this is called (the same
    no-silent-``mkdir`` contract :class:`FileObjectStore` holds) and for
    the watch's lifetime — a tree that vanishes mid-watch raises out of
    the generator rather than being quietly treated as "everything was
    deleted."

    ``events_out`` is handed to every pass verbatim; note a *path* is
    opened fresh (truncated) per pass, so over a multi-pass watch only
    the last pass's sidecar survives — pass an open handle instead to
    accumulate every pass's events in one stream.

    Example::

        stop = threading.Event()
        for report in watch_directory(
            "manuals/", ingester=ingester, checkpoints=checkpoints,
            interval_secs=30, should_stop=stop,
        ):
            log.info("pass done: imported=%d unchanged=%d", report.imported, report.unchanged)
    """
    if interval_secs < 0:
        raise ValueError("interval_secs must be >= 0")
    # Eager, not inside the generator body: a missing directory should
    # fail at the call site, not on the first `next()`.
    store = FileObjectStore(directory)
    stop = _normalize_stop(should_stop)
    wait = _stop_aware_wait(should_stop, stop)

    def _passes() -> Iterator[RunReport]:
        while not stop():
            yield sync_object_storage(
                store,
                "",
                ingester=ingester,
                checkpoints=checkpoints,
                connector=connector,
                deletion_policy=deletion_policy,
                dry_run=dry_run,
                should_stop=should_stop,
                events_out=events_out,
            )
            if not wait(interval_secs):
                return

    return _passes()


def _stop_aware_wait(
    should_stop: Callable[[], bool] | threading.Event | None,
    stop: Callable[[], bool],
) -> Callable[[float], bool]:
    """Returns ``wait(seconds) -> keep_going``: sleeps out the interval
    but returns ``False`` the moment the stop condition fires. An
    ``Event`` waits natively; a plain callable is polled every
    ``_STOP_POLL_SECS``."""
    if isinstance(should_stop, threading.Event):
        return lambda seconds: not should_stop.wait(seconds)

    def wait(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            if stop():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(_STOP_POLL_SECS, remaining))

    return wait
