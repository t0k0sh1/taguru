"""The object-storage boundary (ADR 0007 §9, issue #351):
`object_fingerprint`'s checkpoint-fingerprint priority, `FileObjectStore`
(stdlib-only — needs no optional dependency at all), `open_object_store`'s
scheme dispatch, and `S3ObjectStore` (boto3, exercised via the real
library stubbed at the HTTP layer with `botocore.stub.Stubber` — never a
live bucket, never `moto`, matching this issue's own "no
minio/localstack" testing posture one layer up)."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pytest

import taguru_langchain.ingest_connectors.objectstore as objectstore_module
from taguru_langchain.ingest_connectors.objectstore import (
    FileObjectStore,
    ObjectMeta,
    ObjectNotFoundError,
    PermanentStoreError,
    S3ObjectStore,
    TransientStoreError,
    object_fingerprint,
    open_object_store,
)

from .._s3 import write_bucket

# ---------------------------------------------------------------------------
# object_fingerprint — ADR 0007 §9's priority order
# ---------------------------------------------------------------------------


def _meta(**overrides: object) -> ObjectMeta:
    defaults: dict[str, object] = {"key": "a.txt", "size": 5, "last_modified": None}
    defaults.update(overrides)
    return ObjectMeta(**defaults)  # type: ignore[arg-type]


def test_version_id_wins_over_everything_else() -> None:
    meta = _meta(
        version_id="v1",
        checksum="c1",
        last_modified=datetime(2026, 1, 1, tzinfo=timezone.utc),
        etag="e1",
    )
    assert object_fingerprint(meta) == ("version_id", "v1")


def test_checksum_wins_over_size_last_modified_and_etag() -> None:
    meta = _meta(checksum="c1", last_modified=datetime(2026, 1, 1, tzinfo=timezone.utc), etag="e1")
    assert object_fingerprint(meta) == ("checksum", "c1")


def test_size_and_last_modified_wins_over_etag_alone() -> None:
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    meta = _meta(size=42, last_modified=when, etag="e1")
    assert object_fingerprint(meta) == ("size_last_modified", f"42:{when.isoformat()}")


def test_etag_is_the_last_resort() -> None:
    meta = _meta(etag="e1")
    assert object_fingerprint(meta) == ("etag", "e1")


def test_no_usable_fingerprint_is_none() -> None:
    assert object_fingerprint(_meta()) is None


# ---------------------------------------------------------------------------
# FileObjectStore — stdlib only, the file:// test/air-gapped backend
# ---------------------------------------------------------------------------


def test_file_object_store_refuses_a_missing_bucket_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileObjectStore(tmp_path / "does-not-exist")


def test_file_object_store_lists_by_string_prefix_not_directory_boundary(
    tmp_path: Path,
) -> None:
    write_bucket(
        tmp_path,
        {
            "2026/q1-report.pdf": b"pdf bytes",
            "2026/q1-notes.txt": b"notes",
            "2026/q2-report.pdf": b"other",
        },
    )
    store = FileObjectStore(tmp_path)
    keys = sorted(meta.key for meta in store.list("2026/q1"))
    assert keys == ["2026/q1-notes.txt", "2026/q1-report.pdf"]


def test_file_object_store_list_reports_size_and_mtime(tmp_path: Path) -> None:
    write_bucket(tmp_path, {"a.txt": b"hello"})
    metas = list(FileObjectStore(tmp_path).list(""))
    assert len(metas) == 1
    assert metas[0].key == "a.txt"
    assert metas[0].size == 5
    assert metas[0].last_modified is not None
    assert metas[0].etag is None
    assert metas[0].version_id is None
    assert metas[0].checksum is None


def test_file_object_store_get_returns_bytes_and_guessed_content_type(tmp_path: Path) -> None:
    write_bucket(tmp_path, {"doc.pdf": b"%PDF-1.4 ..."})
    fetched = FileObjectStore(tmp_path).get("doc.pdf")
    assert fetched.body == b"%PDF-1.4 ..."
    assert fetched.content_type == "application/pdf"


def test_file_object_store_get_raises_not_found_for_a_missing_key(tmp_path: Path) -> None:
    store = FileObjectStore(tmp_path)
    with pytest.raises(ObjectNotFoundError):
        store.get("missing.txt")


def test_file_object_store_never_carries_tags(tmp_path: Path) -> None:
    write_bucket(tmp_path, {"a.txt": b"x"})
    assert FileObjectStore(tmp_path).object_tags("a.txt") == {}


def test_file_object_store_refuses_a_key_that_escapes_the_bucket(tmp_path: Path) -> None:
    write_bucket(tmp_path, {"a.txt": b"x"})
    store = FileObjectStore(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        store.get("../outside.txt")


def test_file_object_store_base_uri(tmp_path: Path) -> None:
    store = FileObjectStore(tmp_path)
    assert store.base_uri == f"file://{tmp_path.resolve().as_posix()}"


# ---------------------------------------------------------------------------
# open_object_store — scheme dispatch, mirroring src/ship.rs's open_store
# ---------------------------------------------------------------------------


def test_open_object_store_dispatches_file_scheme(tmp_path: Path) -> None:
    write_bucket(tmp_path, {"a.txt": b"x"})
    store, prefix = open_object_store(f"file://{tmp_path}")
    assert isinstance(store, FileObjectStore)
    assert prefix == ""


def test_open_object_store_rejects_s3_kwargs_for_file_scheme(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not accept"):
        open_object_store(f"file://{tmp_path}", region_name="us-east-1")


def test_open_object_store_rejects_an_s3_url_with_no_bucket() -> None:
    with pytest.raises(ValueError, match="bucket"):
        open_object_store("s3:///prefix")


def test_open_object_store_rejects_an_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="s3://|file://"):
        open_object_store("gs://bucket/prefix")


def test_open_object_store_returns_the_urls_own_path_as_prefix() -> None:
    pytest.importorskip("boto3")
    # An explicit region avoids depending on this machine's own ambient AWS
    # config (~/.aws/config) — S3ObjectStore's construction never resolves
    # a region on its own, so a CI runner with no AWS config at all must
    # behave identically to a dev machine that happens to have one.
    store, prefix = open_object_store("s3://my-bucket/reports/2026/", region_name="us-east-1")
    assert store.base_uri == "s3://my-bucket"
    assert prefix == "reports/2026/"


# ---------------------------------------------------------------------------
# S3ObjectStore — real boto3, stubbed at the HTTP layer (botocore.stub.Stubber)
# ---------------------------------------------------------------------------


def _stubbed_store(bucket: str = "test-bucket") -> tuple[S3ObjectStore, object]:
    import boto3
    from botocore.stub import Stubber

    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    stubber = Stubber(client)
    return S3ObjectStore(bucket, client=client), stubber


def _streaming_body(data: bytes) -> object:
    from botocore.response import StreamingBody

    return StreamingBody(io.BytesIO(data), len(data))


def test_endpoint_url_with_embedded_credentials_is_rejected() -> None:
    pytest.importorskip("boto3")
    with pytest.raises(ValueError, match="credentials"):
        S3ObjectStore("bucket", endpoint_url="https://user:pass@example.com")


def test_client_is_mutually_exclusive_with_connection_kwargs() -> None:
    pytest.importorskip("boto3")
    import boto3 as real_boto3

    client = real_boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="t", aws_secret_access_key="t"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        S3ObjectStore("bucket", client=client, region_name="us-east-1")


def test_missing_boto3_raises_a_clear_error_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(objectstore_module, "boto3", None)
    with pytest.raises(ImportError, match="boto3"):
        S3ObjectStore("bucket")


def test_base_uri_names_the_bucket() -> None:
    pytest.importorskip("boto3")
    store, _stubber = _stubbed_store("reports")
    assert store.base_uri == "s3://reports"


def test_list_returns_objects_from_an_unversioned_bucket() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_response("get_bucket_versioning", {}, {"Bucket": "test-bucket"})
    stubber.add_response(
        "list_objects_v2",
        {
            "Contents": [
                {"Key": "a.pdf", "Size": 5, "ETag": '"abc123"'},
                {"Key": "b.pdf", "Size": 9, "ETag": '"def456"'},
            ],
            "IsTruncated": False,
        },
        {"Bucket": "test-bucket", "Prefix": ""},
    )
    with stubber:
        metas = list(store.list(""))
    assert [meta.key for meta in metas] == ["a.pdf", "b.pdf"]
    # ETag's literal double quotes are stripped — S3 always wraps it in them.
    assert metas[0].etag == "abc123"
    assert metas[0].version_id is None


def test_list_uses_list_object_versions_when_versioning_is_enabled() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_response("get_bucket_versioning", {"Status": "Enabled"}, {"Bucket": "test-bucket"})
    stubber.add_response(
        "list_object_versions",
        {
            "Versions": [
                {"Key": "a.pdf", "Size": 5, "ETag": '"abc"', "VersionId": "v2", "IsLatest": True},
                {
                    "Key": "a.pdf",
                    "Size": 4,
                    "ETag": '"old"',
                    "VersionId": "v1",
                    "IsLatest": False,
                },
            ],
            "IsTruncated": False,
        },
        {"Bucket": "test-bucket", "Prefix": ""},
    )
    with stubber:
        metas = list(store.list(""))
    # Only the IsLatest=True entry survives — a prior version never
    # reappears as if it were a second, currently-present object.
    assert len(metas) == 1
    assert metas[0].version_id == "v2"


def test_list_raises_permanent_not_not_found_for_a_missing_bucket() -> None:
    """A missing bucket is a configuration error (a typo, a bucket never
    created), not one object vanishing mid-pass — `NoSuchBucket` must
    classify as `PermanentStoreError`, never `ObjectNotFoundError`, which
    `ObjectStore.list`'s own contract promises it never raises."""
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    # `_versioning_enabled()` degrades ANY `get_bucket_versioning` failure
    # to "assume not versioned" (a narrower, unrelated permission) — the
    # bucket-missing error a real deployment would hit surfaces from the
    # actual listing call itself, `list_objects_v2`.
    stubber.add_response("get_bucket_versioning", {}, {"Bucket": "test-bucket"})
    stubber.add_client_error(
        "list_objects_v2", service_error_code="NoSuchBucket", http_status_code=404
    )
    with stubber, pytest.raises(PermanentStoreError):
        list(store.list(""))


def test_get_bucket_versioning_failure_degrades_to_the_plain_listing() -> None:
    """ANY failure checking versioning status (a narrower permission than
    listing itself needs) must not fail the whole listing — it degrades to
    "assume not versioned" (ADR 0007 §9's own posture: a listing this class
    can otherwise complete must not fail over an optional check)."""
    pytest.importorskip("boto3")
    from botocore.exceptions import ClientError

    store, stubber = _stubbed_store()
    stubber.add_client_error(
        "get_bucket_versioning", service_error_code="AccessDenied", http_status_code=403
    )
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "a.pdf", "Size": 5, "ETag": '"abc"'}], "IsTruncated": False},
        {"Bucket": "test-bucket", "Prefix": ""},
    )
    with stubber:
        metas = list(store.list(""))
    assert [meta.key for meta in metas] == ["a.pdf"]
    del ClientError  # imported only to document the exception this degrades from


def test_get_returns_body_and_content_type() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_response(
        "get_object",
        {"Body": _streaming_body(b"hello world"), "ContentType": "application/pdf"},
        {"Bucket": "test-bucket", "Key": "a.pdf"},
    )
    with stubber:
        fetched = store.get("a.pdf")
    assert fetched.body == b"hello world"
    assert fetched.content_type == "application/pdf"


def test_get_with_a_version_id_requests_that_version() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_response(
        "get_object",
        {"Body": _streaming_body(b"old content")},
        {"Bucket": "test-bucket", "Key": "a.pdf", "VersionId": "v1"},
    )
    with stubber:
        fetched = store.get("a.pdf", version_id="v1")
    assert fetched.body == b"old content"


def test_get_raises_object_not_found_for_no_such_key() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
    with stubber, pytest.raises(ObjectNotFoundError):
        store.get("gone.pdf")


def test_get_raises_permanent_for_access_denied() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)
    with stubber, pytest.raises(PermanentStoreError):
        store.get("secret.pdf")


def test_get_raises_transient_for_slow_down() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_client_error("get_object", service_error_code="SlowDown", http_status_code=503)
    with stubber, pytest.raises(TransientStoreError):
        store.get("busy.pdf")


def test_get_raises_transient_for_an_unrecognized_error_code() -> None:
    """ADR 0007 §9's permanent bucket is a narrow, explicitly-named list;
    anything not on it defaults to transient — the same "assumed
    transient" posture `src/ship.rs`'s own `ShipError::Io` already takes,
    narrowed only by the explicit permanent carve-out."""
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_client_error(
        "get_object", service_error_code="SomeBrandNewErrorCode", http_status_code=500
    )
    with stubber, pytest.raises(TransientStoreError):
        store.get("a.pdf")


def test_object_tags_returns_a_mapping() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_response(
        "get_object_tagging",
        {"TagSet": [{"Key": "project", "Value": "quarterly"}]},
        {"Bucket": "test-bucket", "Key": "a.pdf"},
    )
    with stubber:
        tags = store.object_tags("a.pdf")
    assert dict(tags) == {"project": "quarterly"}


def test_object_tags_degrades_to_empty_on_access_denied() -> None:
    """s3:GetObjectTagging denied while the object itself is readable —
    degrades to "no tags," never a failure over a tags-only permission gap
    (ADR 0007 §9)."""
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_client_error(
        "get_object_tagging", service_error_code="AccessDenied", http_status_code=403
    )
    with stubber:
        tags = store.object_tags("a.pdf")
    assert dict(tags) == {}


def test_object_tags_degrades_to_empty_when_the_object_is_gone() -> None:
    pytest.importorskip("boto3")
    store, stubber = _stubbed_store()
    stubber.add_client_error(
        "get_object_tagging", service_error_code="NoSuchKey", http_status_code=404
    )
    with stubber:
        tags = store.object_tags("gone.pdf")
    assert dict(tags) == {}
