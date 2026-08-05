"""`AzureBlobObjectStore` (issue #414): the `az://` backend of the
object-storage boundary, exercised through the real ``azure-core``
exception types raised by a network-free fake container client injected
through the class's own documented ``client=`` escape hatch — the same
"real library, no live container" posture the S3/GCS backends' tests
take."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)

from taguru_langchain.ingest_connectors.objectstore import (
    ObjectNotFoundError,
    PermanentStoreError,
    TransientStoreError,
    object_fingerprint,
    open_object_store,
)
from taguru_langchain.ingest_connectors.objectstore_azure import AzureBlobObjectStore

_MODIFIED = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _properties(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "name": "docs/a.txt",
        "size": 5,
        "last_modified": _MODIFIED,
        "etag": '"0x8D-etag"',
        "version_id": None,
        "content_settings": SimpleNamespace(content_md5=bytearray(b"\x01\xab"), content_type=None),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@dataclass
class FakeDownloader:
    body: bytes = b"hello"
    content_type: str | None = "text/plain"

    @property
    def properties(self) -> SimpleNamespace:
        return SimpleNamespace(content_settings=SimpleNamespace(content_type=self.content_type))

    def readall(self) -> bytes:
        return self.body


@dataclass
class FakeBlobClient:
    tags: dict[str, str] | None = None
    error: Exception | None = None

    def get_blob_tags(self) -> dict[str, str] | None:
        if self.error is not None:
            raise self.error
        return self.tags


@dataclass
class FakeContainerClient:
    listed: list[SimpleNamespace] = field(default_factory=list)
    list_error: Exception | None = None
    download_error: Exception | None = None
    downloader: FakeDownloader = field(default_factory=FakeDownloader)
    blob_client: FakeBlobClient = field(default_factory=FakeBlobClient)
    seen_prefix: list[str | None] = field(default_factory=list)
    seen_version: list[str | None] = field(default_factory=list)

    def list_blobs(self, name_starts_with: str | None = None) -> Any:
        self.seen_prefix.append(name_starts_with)
        if self.list_error is not None:
            raise self.list_error
        return iter(self.listed)

    def download_blob(self, key: str, version_id: str | None = None) -> FakeDownloader:
        self.seen_version.append(version_id)
        if self.download_error is not None:
            raise self.download_error
        return self.downloader

    def get_blob_client(self, key: str) -> FakeBlobClient:
        return self.blob_client


def _store(client: FakeContainerClient) -> AzureBlobObjectStore:
    return AzureBlobObjectStore("my-container", client=client)  # type: ignore[arg-type]


def _coded(error: HttpResponseError, code: str) -> HttpResponseError:
    error.error_code = code  # type: ignore[assignment]
    return error


def test_client_is_mutually_exclusive_with_account_and_endpoint() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AzureBlobObjectStore(
            "c",
            account="acct",
            client=FakeContainerClient(),  # type: ignore[arg-type]
        )


def test_base_uri_names_the_container() -> None:
    assert _store(FakeContainerClient()).base_uri == "az://my-container"


def test_construction_requires_an_account_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)
    with pytest.raises(ValueError, match="AZURE_STORAGE_ACCOUNT_NAME"):
        AzureBlobObjectStore("c")


def test_endpoint_url_must_not_embed_credentials() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        AzureBlobObjectStore("c", endpoint_url="https://key:secret@example.com")


# ---------------------------------------------------------------------------
# list — field mapping onto ObjectMeta and its fingerprint tiers
# ---------------------------------------------------------------------------


def test_list_maps_properties_and_strips_the_quoted_etag() -> None:
    client = FakeContainerClient(listed=[_properties()])
    [meta] = _store(client).list("docs/")
    assert client.seen_prefix == ["docs/"]
    assert meta.key == "docs/a.txt"
    assert meta.size == 5
    assert meta.last_modified == _MODIFIED
    assert meta.etag == "0x8D-etag"
    assert meta.version_id is None
    assert meta.checksum == "md5:01ab"
    assert object_fingerprint(meta) == ("checksum", "md5:01ab")


def test_list_version_id_rides_when_the_container_stamps_one() -> None:
    client = FakeContainerClient(listed=[_properties(version_id="2026-03-01T00:00:00Z")])
    [meta] = _store(client).list("")
    assert object_fingerprint(meta) == ("version_id", "2026-03-01T00:00:00Z")


def test_list_without_content_md5_degrades_checksum_to_none() -> None:
    props = _properties(content_settings=SimpleNamespace(content_md5=None, content_type=None))
    [meta] = _store(FakeContainerClient(listed=[props])).list("")
    assert meta.checksum is None


def test_list_on_a_missing_container_is_permanent_not_object_not_found() -> None:
    gone = ResourceNotFoundError(message="no container")
    gone.error_code = "ContainerNotFound"  # type: ignore[assignment]
    store = _store(FakeContainerClient(list_error=gone))
    with pytest.raises(PermanentStoreError):
        list(store.list(""))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ClientAuthenticationError(message="bad credential"), PermanentStoreError),
        (
            _coded(HttpResponseError(message="denied"), "AuthorizationPermissionMismatch"),
            PermanentStoreError,
        ),
        (ServiceRequestError(message="dns"), TransientStoreError),
        (_coded(HttpResponseError(message="busy"), "ServerBusy"), TransientStoreError),
    ],
)
def test_list_error_classification(error: Exception, expected: type[Exception]) -> None:
    store = _store(FakeContainerClient(list_error=error))
    with pytest.raises(expected):
        list(store.list(""))


# ---------------------------------------------------------------------------
# get — bytes + content type, version pinning, not-found semantics
# ---------------------------------------------------------------------------


def test_get_returns_bytes_content_type_and_pins_the_listed_version() -> None:
    client = FakeContainerClient()
    fetched = _store(client).get("docs/a.txt", version_id="v1")
    assert fetched.body == b"hello"
    assert fetched.content_type == "text/plain"
    assert client.seen_version == ["v1"]


def test_get_missing_blob_is_object_not_found() -> None:
    gone = ResourceNotFoundError(message="no blob")
    gone.error_code = "BlobNotFound"  # type: ignore[assignment]
    store = _store(FakeContainerClient(download_error=gone))
    with pytest.raises(ObjectNotFoundError):
        store.get("gone.txt")


def test_get_on_a_missing_container_stays_permanent() -> None:
    # Azure names the missing thing itself, so even the fetch path can
    # tell "the container is a typo" (permanent) apart from "this one
    # blob vanished" (skip this pass).
    gone = ResourceNotFoundError(message="no container")
    gone.error_code = "ContainerNotFound"  # type: ignore[assignment]
    store = _store(FakeContainerClient(download_error=gone))
    with pytest.raises(PermanentStoreError):
        store.get("a")


# ---------------------------------------------------------------------------
# object_tags — blob index tags, with the tags-only degrade
# ---------------------------------------------------------------------------


def test_object_tags_read_blob_index_tags() -> None:
    client = FakeContainerClient(blob_client=FakeBlobClient(tags={"team": "ingest"}))
    assert _store(client).object_tags("a") == {"team": "ingest"}


def test_object_tags_degrade_to_empty_on_none_denial_or_missing() -> None:
    assert _store(FakeContainerClient(blob_client=FakeBlobClient(tags=None))).object_tags("a") == {}
    denied = _coded(HttpResponseError(message="tags denied"), "AuthorizationPermissionMismatch")
    client = FakeContainerClient(blob_client=FakeBlobClient(error=denied))
    assert _store(client).object_tags("a") == {}
    gone = ResourceNotFoundError(message="no blob")
    client = FakeContainerClient(blob_client=FakeBlobClient(error=gone))
    assert _store(client).object_tags("a") == {}


def test_object_tags_raise_on_transient_failures() -> None:
    failing = FakeBlobClient(error=ServiceRequestError(message="dns"))
    with pytest.raises(TransientStoreError):
        _store(FakeContainerClient(blob_client=failing)).object_tags("a")


# ---------------------------------------------------------------------------
# open_object_store — the az:// scheme
# ---------------------------------------------------------------------------


def test_open_object_store_dispatches_az_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "myaccount")
    store, prefix = open_object_store("az://my-container/reports/2026")
    assert isinstance(store, AzureBlobObjectStore)
    assert store.base_uri == "az://my-container"
    assert prefix == "reports/2026"


def test_open_object_store_requires_an_az_container() -> None:
    with pytest.raises(ValueError, match="must name a container"):
        open_object_store("az:///prefix-only")


def test_open_object_store_rejects_s3_only_knobs_for_az() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        open_object_store("az://container", profile_name="prod")
