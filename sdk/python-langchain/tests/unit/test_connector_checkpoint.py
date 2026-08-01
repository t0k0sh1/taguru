"""ConnectorCheckpoint (ADR 0007 §6.3, issue #347): "did I already fetch
and parse this object," composing with — never replacing —
:class:`~taguru_langchain.checkpoints.CheckpointStore`."""

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
