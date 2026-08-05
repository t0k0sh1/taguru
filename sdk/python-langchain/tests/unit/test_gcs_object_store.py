"""`GCSObjectStore` (issue #414): the `gs://` backend of the object-storage
boundary, exercised through the real ``google-cloud-storage`` exception
types raised by a network-free fake client injected through the class's
own documented ``client=`` escape hatch — the same "real library, no live
bucket" posture ``test_object_store.py`` takes with `botocore.stub.Stubber`
one layer down (google-cloud-storage ships no stubber of its own, so the
fake sits at the client surface instead of the HTTP layer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest
from google.api_core.exceptions import (
    Forbidden,
    NotFound,
    ServiceUnavailable,
    TooManyRequests,
)
from google.auth.exceptions import DefaultCredentialsError

from taguru_langchain.ingest_connectors.objectstore import (
    ObjectNotFoundError,
    PermanentStoreError,
    TransientStoreError,
    object_fingerprint,
    open_object_store,
)
from taguru_langchain.ingest_connectors.objectstore_gcs import GCSObjectStore

_UPDATED = datetime(2026, 3, 1, tzinfo=timezone.utc)


@dataclass
class FakeBlob:
    name: str = "docs/a.txt"
    size: int = 5
    updated: datetime | None = _UPDATED
    etag: str | None = "etag-1"
    generation: int | None = 7
    crc32c: str | None = "crc-1"
    md5_hash: str | None = "md5-1"
    content_type: str | None = "text/plain"
    metadata: dict[str, str] | None = None
    body: bytes = b"hello"

    def download_as_bytes(self) -> bytes:
        return self.body


@dataclass
class FakeBucket:
    blobs: dict[str, FakeBlob] = field(default_factory=dict)
    error: Exception | None = None
    seen_generation: list[int | None] = field(default_factory=list)

    def get_blob(self, key: str, generation: int | None = None) -> FakeBlob | None:
        self.seen_generation.append(generation)
        if self.error is not None:
            raise self.error
        return self.blobs.get(key)


@dataclass
class FakeClient:
    listed: list[FakeBlob] = field(default_factory=list)
    list_error: Exception | None = None
    fake_bucket: FakeBucket = field(default_factory=FakeBucket)
    seen_prefix: list[str] = field(default_factory=list)

    def list_blobs(self, bucket: str, prefix: str = "") -> Any:
        self.seen_prefix.append(prefix)
        if self.list_error is not None:
            raise self.list_error
        return iter(self.listed)

    def bucket(self, name: str) -> FakeBucket:
        return self.fake_bucket


def _store(client: FakeClient) -> GCSObjectStore:
    return GCSObjectStore("my-bucket", client=client)  # type: ignore[arg-type]


def test_client_is_mutually_exclusive_with_project() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        GCSObjectStore("b", project="p", client=FakeClient())  # type: ignore[arg-type]


def test_base_uri_names_the_bucket() -> None:
    assert _store(FakeClient()).base_uri == "gs://my-bucket"


# ---------------------------------------------------------------------------
# list — field mapping onto ObjectMeta and its fingerprint tiers
# ---------------------------------------------------------------------------


def test_list_maps_generation_to_version_id_and_prefers_crc32c() -> None:
    client = FakeClient(listed=[FakeBlob()])
    [meta] = _store(client).list("docs/")
    assert client.seen_prefix == ["docs/"]
    assert meta.key == "docs/a.txt"
    assert meta.size == 5
    assert meta.last_modified == _UPDATED
    assert meta.etag == "etag-1"
    # Generation is stamped on every overwrite whether or not the bucket
    # has versioning on — the top fingerprint tier, always available.
    assert meta.version_id == "7"
    assert meta.checksum == "crc32c:crc-1"
    assert object_fingerprint(meta) == ("version_id", "7")


def test_list_checksum_falls_back_to_md5_then_none() -> None:
    client = FakeClient(listed=[FakeBlob(crc32c=None), FakeBlob(crc32c=None, md5_hash=None)])
    first, second = _store(client).list("")
    assert first.checksum == "md5:md5-1"
    assert second.checksum is None


def test_list_without_generation_degrades_version_id_to_none() -> None:
    client = FakeClient(listed=[FakeBlob(generation=None)])
    [meta] = _store(client).list("")
    assert meta.version_id is None


def test_list_on_a_missing_bucket_is_permanent_not_object_not_found() -> None:
    store = _store(FakeClient(list_error=NotFound("no such bucket")))
    with pytest.raises(PermanentStoreError):
        list(store.list(""))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (Forbidden("denied"), PermanentStoreError),
        (DefaultCredentialsError("no ADC"), PermanentStoreError),
        (ServiceUnavailable("503"), TransientStoreError),
        (TooManyRequests("429"), TransientStoreError),
        (ConnectionResetError("reset"), TransientStoreError),
    ],
)
def test_list_error_classification(error: Exception, expected: type[Exception]) -> None:
    store = _store(FakeClient(list_error=error))
    with pytest.raises(expected):
        list(store.list(""))


# ---------------------------------------------------------------------------
# get — bytes + content type, generation pinning, not-found semantics
# ---------------------------------------------------------------------------


def test_get_returns_bytes_content_type_and_pins_the_listed_generation() -> None:
    bucket = FakeBucket(blobs={"docs/a.txt": FakeBlob()})
    store = _store(FakeClient(fake_bucket=bucket))
    fetched = store.get("docs/a.txt", version_id="7")
    assert fetched.body == b"hello"
    assert fetched.content_type == "text/plain"
    assert bucket.seen_generation == [7]


def test_get_without_version_id_passes_no_generation() -> None:
    bucket = FakeBucket(blobs={"a": FakeBlob()})
    _store(FakeClient(fake_bucket=bucket)).get("a")
    assert bucket.seen_generation == [None]


def test_get_missing_object_is_object_not_found() -> None:
    store = _store(FakeClient(fake_bucket=FakeBucket()))
    with pytest.raises(ObjectNotFoundError):
        store.get("gone.txt")


def test_get_not_found_from_the_service_is_object_not_found() -> None:
    # A vanished generation (overwritten object, versioning off) answers
    # 404 — "skip this object this pass," never a store failure.
    store = _store(FakeClient(fake_bucket=FakeBucket(error=NotFound("stale generation"))))
    with pytest.raises(ObjectNotFoundError):
        store.get("a", version_id="6")


def test_get_permission_denial_is_permanent() -> None:
    store = _store(FakeClient(fake_bucket=FakeBucket(error=Forbidden("denied"))))
    with pytest.raises(PermanentStoreError):
        store.get("a")


# ---------------------------------------------------------------------------
# object_tags — custom metadata, with the tags-only degrade
# ---------------------------------------------------------------------------


def test_object_tags_map_the_blobs_custom_metadata() -> None:
    blob = FakeBlob(metadata={"team": "ingest", "quarter": "q1"})
    store = _store(FakeClient(fake_bucket=FakeBucket(blobs={"a": blob})))
    assert store.object_tags("a") == {"team": "ingest", "quarter": "q1"}


def test_object_tags_degrade_to_empty_for_missing_blob_or_metadata() -> None:
    store = _store(FakeClient(fake_bucket=FakeBucket(blobs={"a": FakeBlob(metadata=None)})))
    assert store.object_tags("a") == {}
    assert store.object_tags("gone") == {}


def test_object_tags_degrade_on_permission_denial_but_raise_on_transient() -> None:
    store = _store(FakeClient(fake_bucket=FakeBucket(error=Forbidden("denied"))))
    assert store.object_tags("a") == {}
    store = _store(FakeClient(fake_bucket=FakeBucket(error=ServiceUnavailable("503"))))
    with pytest.raises(TransientStoreError):
        store.object_tags("a")


# ---------------------------------------------------------------------------
# open_object_store — the gs:// scheme
# ---------------------------------------------------------------------------


def test_open_object_store_dispatches_gs_urls() -> None:
    store, prefix = open_object_store("gs://my-bucket/reports/2026")
    assert isinstance(store, GCSObjectStore)
    assert store.base_uri == "gs://my-bucket"
    assert prefix == "reports/2026"


def test_open_object_store_requires_a_gs_bucket() -> None:
    with pytest.raises(ValueError, match="must name a bucket"):
        open_object_store("gs:///prefix-only")


def test_open_object_store_rejects_s3_only_knobs_for_gs() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        open_object_store("gs://bucket", region_name="asia-northeast1")
