"""ConnectorCheckpoint (ADR 0007 §6.3, issue #347): "did I already fetch
and parse this object," composing with — never replacing —
:class:`~taguru_langchain.checkpoints.CheckpointStore`.

Also :class:`~taguru_langchain.ingest_connectors.checkpoint.FileProbeCheckpoint`
(ADR 0007 §11, issue #353): the local-file twin of
:class:`~taguru_langchain.ingest_connectors.s3.S3ObjectCheckpoint` — "does
this file's cheap metadata match what we saw last time," answerable with no
read at all, consulted only under ``--dry-run``."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from taguru import Locator

from taguru_langchain.checkpoints import FilesystemCheckpointStore
from taguru_langchain.ingest_connectors import (
    ConnectorCheckpoint,
    ConnectorDocument,
    ConnectorMetadata,
    FingerprintInputs,
    LocatorEntry,
)
from taguru_langchain.ingest_connectors.checkpoint import FileProbe, FileProbeCheckpoint

from .test_checkpoints import (
    FailingDeleteStore,
    FailingLoadStore,
    FailingSaveStore,
    RecordingCheckpointStore,
)


def _fingerprint(**overrides: str) -> FingerprintInputs:
    defaults = {
        "raw_content_sha256": "deadbeef",
        "parser": "taguru-text-connector",
        "parser_version": "1.0.0",
        "parse_options_digest": "cafef00d",
    }
    defaults.update(overrides)
    return FingerprintInputs(**defaults)


def _document(
    *, source: str = "doc.md", fingerprint: FingerprintInputs | None = None
) -> ConnectorDocument:
    return ConnectorDocument(
        source=source,
        text="paragraph one.\n\nparagraph two.",
        locators=(LocatorEntry(paragraph=1, locator=Locator(kind="page", value="1")),),
        metadata=ConnectorMetadata(origin_uri=source, display_name=source),
        fingerprint_inputs=fingerprint if fingerprint is not None else _fingerprint(),
    )


def test_round_trips_a_document_when_the_fingerprint_matches() -> None:
    store = RecordingCheckpointStore()
    checkpoint = ConnectorCheckpoint(store)
    document = _document()
    checkpoint.save(document)
    loaded = checkpoint.load(document.source, document.fingerprint_inputs)
    assert loaded == document


def test_load_is_none_on_a_fingerprint_mismatch() -> None:
    store = RecordingCheckpointStore()
    checkpoint = ConnectorCheckpoint(store)
    document = _document()
    checkpoint.save(document)
    mismatched = _fingerprint(raw_content_sha256="different")
    assert checkpoint.load(document.source, mismatched) is None


def test_load_is_none_when_the_stored_documents_source_does_not_match() -> None:
    """A CheckpointStore is only obligated to key by the string it was
    given (``CheckpointStore``'s own contract) — it is not guaranteed to
    never return another key's bytes under a different key (a buggy or
    lossy custom implementation). Two documents whose fetched bytes are
    byte-identical share a fingerprint, so the fingerprint check alone
    cannot catch this; the explicit ``document.source`` check does."""
    document_a = _document(source="a.md")
    payload = json.dumps(document_a.to_dict()).encode("utf-8")
    # Seed the store so a lookup under "b.md"'s own key returns "a.md"'s
    # checkpoint bytes verbatim — simulating a store bug/collision rather
    # than reproducing one through ConnectorCheckpoint's own key derivation.
    store = RecordingCheckpointStore(seed={"connector:b.md": payload})
    checkpoint = ConnectorCheckpoint(store)
    assert checkpoint.load("b.md", document_a.fingerprint_inputs) is None


def test_load_is_none_when_nothing_was_ever_saved() -> None:
    checkpoint = ConnectorCheckpoint(RecordingCheckpointStore())
    assert checkpoint.load("never-seen.md", _fingerprint()) is None


def test_load_is_none_on_corrupt_bytes() -> None:
    store = RecordingCheckpointStore(seed={"connector:doc.md": b"not json {"})
    checkpoint = ConnectorCheckpoint(store)
    assert checkpoint.load("doc.md", _fingerprint()) is None


def test_delete_removes_a_saved_document() -> None:
    store = RecordingCheckpointStore()
    checkpoint = ConnectorCheckpoint(store)
    document = _document()
    checkpoint.save(document)
    checkpoint.delete(document.source)
    assert checkpoint.load(document.source, document.fingerprint_inputs) is None


def test_namespace_prevents_collision_with_taguru_ingesters_own_checkpoint(
    tmp_path: Path,
) -> None:
    """A ConnectorCheckpoint and TaguruIngester's own chunk checkpoint can
    safely share one FilesystemCheckpointStore directory: the namespace
    prefix changes the full key string _checkpoint_file_name hashes, so
    the two never collide on the same file even for the identical bare
    source id."""
    store = FilesystemCheckpointStore(tmp_path)
    connector_checkpoint = ConnectorCheckpoint(store)
    document = _document(source="shared-source.md")
    connector_checkpoint.save(document)

    # TaguruIngester's own checkpoint machinery would call store.save
    # directly with the bare source id — simulate that here without
    # constructing a full ingester.
    store.save("shared-source.md", b'{"fingerprint": {}, "units": {}}')

    # The connector's own entry is untouched by the ingester writing under
    # the bare (unprefixed) key.
    loaded = connector_checkpoint.load("shared-source.md", document.fingerprint_inputs)
    assert loaded == document


def test_load_failure_warns_and_returns_none() -> None:
    with pytest.warns(RuntimeWarning, match="load"):
        checkpoint = ConnectorCheckpoint(FailingLoadStore())
        assert checkpoint.load("doc.md", _fingerprint()) is None


def test_save_failure_warns_but_does_not_raise() -> None:
    with pytest.warns(RuntimeWarning, match="save"):
        checkpoint = ConnectorCheckpoint(FailingSaveStore())
        checkpoint.save(_document())


def test_delete_failure_is_silently_ignored() -> None:
    checkpoint = ConnectorCheckpoint(FailingDeleteStore())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checkpoint.delete("doc.md")  # must not raise or warn


# ---------------------------------------------------------------------------
# FileProbeCheckpoint — the cheap "skip the dry-run reparse" layer
# ---------------------------------------------------------------------------


def _probe(**overrides: object) -> FileProbe:
    defaults: dict[str, object] = {
        "size": 4096,
        "mtime_ns": 1_700_000_000_000_000_000,
        "parser": "taguru-text-connector",
        "parser_version": "1.0.0",
        "parse_options_digest": "cafef00d",
    }
    defaults.update(overrides)
    return FileProbe(**defaults)  # type: ignore[arg-type]


def test_file_probe_checkpoint_round_trips() -> None:
    checkpoint = FileProbeCheckpoint(RecordingCheckpointStore())
    checkpoint.save("docs/manual.md", _probe())
    assert checkpoint.load("docs/manual.md") == _probe()


def test_file_probe_checkpoint_is_none_when_never_saved() -> None:
    checkpoint = FileProbeCheckpoint(RecordingCheckpointStore())
    assert checkpoint.load("never-seen.md") is None


def test_file_probe_checkpoint_is_none_on_corrupt_bytes() -> None:
    store = RecordingCheckpointStore(seed={"file-probe:docs/manual.md": b"not json {"})
    checkpoint = FileProbeCheckpoint(store)
    assert checkpoint.load("docs/manual.md") is None


def test_file_probe_checkpoint_is_none_when_source_does_not_match() -> None:
    """Same store-contract caveat `ConnectorCheckpoint` guards against: a
    lookup under a different source than the one the payload was recorded
    for must never return that payload."""
    payload = json.dumps(
        {
            "version": 1,
            "source": "a.md",
            "size": 1,
            "mtime_ns": 1,
            "parser": "taguru-text-connector",
            "parser_version": "1.0.0",
            "parse_options_digest": "x",
        }
    ).encode("utf-8")
    store = RecordingCheckpointStore(seed={"file-probe:b.md": payload})
    checkpoint = FileProbeCheckpoint(store)
    assert checkpoint.load("b.md") is None


def test_file_probe_checkpoint_is_none_on_a_stale_version() -> None:
    """A future version bump must not honor an old-format entry — the same
    degrade-to-uncached posture every version check in this SDK takes,
    never a partial/best-effort read of a mismatched version."""
    payload = json.dumps(
        {
            "version": 0,
            "source": "docs/manual.md",
            "size": 1,
            "mtime_ns": 1,
            "parser": "taguru-text-connector",
            "parser_version": "1.0.0",
            "parse_options_digest": "x",
        }
    ).encode("utf-8")
    store = RecordingCheckpointStore(seed={"file-probe:docs/manual.md": payload})
    checkpoint = FileProbeCheckpoint(store)
    assert checkpoint.load("docs/manual.md") is None


def test_file_probe_checkpoint_delete_removes_the_entry() -> None:
    checkpoint = FileProbeCheckpoint(RecordingCheckpointStore())
    checkpoint.save("docs/manual.md", _probe())
    checkpoint.delete("docs/manual.md")
    assert checkpoint.load("docs/manual.md") is None


def test_file_probe_checkpoint_load_failure_is_silently_uncached() -> None:
    checkpoint = FileProbeCheckpoint(FailingLoadStore())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert checkpoint.load("docs/manual.md") is None


def test_file_probe_checkpoint_save_failure_does_not_raise() -> None:
    checkpoint = FileProbeCheckpoint(FailingSaveStore())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checkpoint.save("docs/manual.md", _probe())


def test_file_probe_checkpoint_delete_failure_is_silently_ignored() -> None:
    checkpoint = FileProbeCheckpoint(FailingDeleteStore())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checkpoint.delete("docs/manual.md")  # must not raise or warn


def test_file_probe_checkpoint_namespace_prevents_collision_with_connector_checkpoint(
    tmp_path: Path,
) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    file_probe_checkpoint = FileProbeCheckpoint(store)
    connector_checkpoint = ConnectorCheckpoint(store)
    file_probe_checkpoint.save("shared-source.md", _probe())
    connector_checkpoint.save(_document(source="shared-source.md"))

    assert file_probe_checkpoint.load("shared-source.md") == _probe()
    assert connector_checkpoint.load(
        "shared-source.md", _document(source="shared-source.md").fingerprint_inputs
    ) == _document(source="shared-source.md")
