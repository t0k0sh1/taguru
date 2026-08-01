"""Test-only :class:`~taguru_langchain.ingest_connectors.objectstore.ObjectStore`
support for the S3 connector's own tests (issue #351):
:class:`FakeObjectStore`, an in-memory store synthesized at test time (the
same "no binary/external fixture" convention `tests/_pdfs.py`/`tests/_docx.py`
already established), and :func:`write_bucket`, a small helper that lays out
a plain directory tree for :class:`~taguru_langchain.ingest_connectors.
objectstore.FileObjectStore` — the ``file://`` backend issue #351's own
acceptance criteria names explicitly ("minio/localstack を使わず file:// バ
ケットでテスト").

:class:`FakeObjectStore` exists ALONGSIDE ``FileObjectStore`` rather than
instead of it: the two together let a test choose the right level —
``FakeObjectStore`` for anything about listing/fingerprint/dispatch/
checkpoint logic (no filesystem, deterministic, can inject
``version_id``/``checksum``/tags/failures ``FileObjectStore`` can never
produce), ``FileObjectStore``-over-``write_bucket`` for anything proving the
real stdlib-only backend and a real end-to-end sync work.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from taguru_langchain.ingest_connectors.objectstore import (
    FetchedObject,
    ObjectMeta,
    ObjectNotFoundError,
)


@dataclass(slots=True)
class _StoredObject:
    body: bytes
    content_type: str | None = None
    etag: str | None = None
    version_id: str | None = None
    checksum: str | None = None
    last_modified: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)


class FakeObjectStore:
    """An in-memory :class:`~taguru_langchain.ingest_connectors.objectstore.
    ObjectStore` — seeded via :meth:`put`, never touches a filesystem or
    network. Tracks ``get_calls``/``tags_calls`` (key → count) so a test can
    assert a checkpoint hit skipped the fetch entirely, and accepts an
    injected ``fail_list_with``/``fail_get_with``/``fail_tags_with``
    exception instance to prove transient/permanent-error handling without
    a real botocore error."""

    def __init__(self, bucket: str = "test-bucket") -> None:
        self._bucket = bucket
        self._objects: dict[str, _StoredObject] = {}
        self.get_calls: dict[str, int] = {}
        self.tags_calls: dict[str, int] = {}
        self.fail_list_with: Exception | None = None
        self.fail_get_with: Exception | None = None
        self.fail_tags_with: Exception | None = None

    @property
    def base_uri(self) -> str:
        return f"s3://{self._bucket}"

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str | None = None,
        etag: str | None = None,
        version_id: str | None = None,
        checksum: str | None = None,
        last_modified: datetime | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Seeds (or replaces) one object. A later :meth:`put` of the same
        ``key`` is how a test simulates "the object changed" between two
        sync passes."""
        self._objects[key] = _StoredObject(
            body=body,
            content_type=content_type,
            etag=etag,
            version_id=version_id,
            checksum=checksum,
            last_modified=last_modified,
            tags=dict(tags) if tags else {},
        )

    def delete(self, key: str) -> None:
        """Removes ``key`` — how a test simulates "the object was deleted
        between two sync passes."""
        self._objects.pop(key, None)

    def list(self, prefix: str) -> Iterator[ObjectMeta]:
        if self.fail_list_with is not None:
            raise self.fail_list_with
        for key in sorted(self._objects):
            if not key.startswith(prefix):
                continue
            stored = self._objects[key]
            yield ObjectMeta(
                key=key,
                size=len(stored.body),
                last_modified=stored.last_modified,
                etag=stored.etag,
                version_id=stored.version_id,
                checksum=stored.checksum,
            )

    def get(self, key: str, *, version_id: str | None = None) -> FetchedObject:
        self.get_calls[key] = self.get_calls.get(key, 0) + 1
        if self.fail_get_with is not None:
            raise self.fail_get_with
        stored = self._objects.get(key)
        if stored is None or (version_id is not None and stored.version_id != version_id):
            raise ObjectNotFoundError(f"{key!r} does not exist in this fake bucket")
        return FetchedObject(body=stored.body, content_type=stored.content_type)

    def object_tags(self, key: str) -> Mapping[str, str]:
        self.tags_calls[key] = self.tags_calls.get(key, 0) + 1
        if self.fail_tags_with is not None:
            raise self.fail_tags_with
        stored = self._objects.get(key)
        return dict(stored.tags) if stored is not None else {}


def write_bucket(root: Path, objects: Mapping[str, bytes]) -> None:
    """Lays ``objects`` (key → bytes) out under ``root`` as a plain
    directory tree — what :class:`~taguru_langchain.ingest_connectors.
    objectstore.FileObjectStore` reads. ``root`` itself must already exist
    (mirrors ``FileObjectStore``'s/``src/ship.rs``'s own "the bucket must
    exist" contract; a test passes ``tmp_path`` directly, which pytest
    already creates)."""
    for key, body in objects.items():
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
