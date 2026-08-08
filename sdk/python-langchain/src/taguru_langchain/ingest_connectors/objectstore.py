"""The object-storage boundary (ADR 0007 §9, issue #351): a thin store
abstraction two backends implement — :class:`S3ObjectStore` (the real
thing, optional ``boto3`` dependency via the ``s3`` extra) and
:class:`FileObjectStore` (stdlib only, a ``file://`` directory tree).

Deliberately NOT a reimplementation of ``src/ship.rs``'s Rust
``object_store``-backed replication path: this package's own packaging
boundary (ADR 0007 §3/§4, Option C) keeps every connector client-side and
`src/` untouched, so an object-storage connector cannot call the same Rust
crate `src/ship.rs:176`'s ``open_store`` does without either adding a new
Rust binary or a subprocess/FFI boundary neither this ADR's traceability
table nor issue #351's own "``file://`` buckets, no minio/localstack"
testing requirement can satisfy without one existing already. What IS
reused verbatim is the *shape* of `open_store`: parse a URL, dispatch on
scheme, return a store scoped to one bucket/directory plus the prefix the
URL itself named — and the same refusal posture (`file://` never silently
``mkdir``s a missing "bucket"; an unsupported scheme names the ones that
are supported).

Two exception kinds only (ADR 0007 §9's explicit two-way split, extending
`src/ship.rs`'s own `ShipError::Io` "retry next pass" assumption with the
one case that must NOT retry forever): :class:`TransientStoreError`
(network/timeout/5xx/throttling — safe to retry on the next enumeration
pass) and :class:`PermanentStoreError` (invalid/expired/revoked credential,
or object-level permission denial — terminal, reported once, never
auto-retried). :class:`ObjectNotFoundError` is a third, narrower case (the
object vanished between listing and fetch) that is neither: the caller
treats it as "skip this object this pass," not a failure at all.
"""

from __future__ import annotations

import errno
import mimetypes
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING, Final, Literal, Protocol
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

try:
    import boto3
    from botocore.exceptions import (
        ClientError,
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        NoCredentialsError,
        ReadTimeoutError,
    )
except ImportError as _boto3_import_error:  # pragma: no cover - exercised only without boto3
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment, misc]
    ConnectionClosedError = Exception  # type: ignore[assignment, misc]
    ConnectTimeoutError = Exception  # type: ignore[assignment, misc]
    EndpointConnectionError = Exception  # type: ignore[assignment, misc]
    NoCredentialsError = Exception  # type: ignore[assignment, misc]
    ReadTimeoutError = Exception  # type: ignore[assignment, misc]
    _BOTO3_IMPORT_ERROR: ImportError | None = _boto3_import_error
else:
    _BOTO3_IMPORT_ERROR = None


class TransientStoreError(Exception):
    """A network/timeout/5xx/throttling failure — the caller's next
    enumeration pass retries from where this one left off, the same
    ``ShipError::Io`` assumption `src/ship.rs` already makes for
    replication."""


class PermanentStoreError(Exception):
    """Invalid/expired/revoked credentials, or object-level permission
    denial (403/``AccessDenied``-shaped). Terminal immediately: retrying an
    auth failure on every future enumeration pass would silently mask a
    real, fixable problem behind an endless quiet retry loop (ADR 0007
    §9)."""


# The same "not really a filesystem error, treat this path as not-a-file"
# errno set `pathlib.Path.is_file()` itself already ignores (a path that
# vanished between listing and stat, a path component that turned out not
# to be a directory, a stale/looping symlink) — `FileObjectStore.list()`'s
# own `os.scandir`-based scan below matches that posture exactly when it
# stats an individual entry.
_IGNORED_STAT_ERRNOS: Final = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})

# When it comes to OPENING a nested directory to recurse into, the scan is
# additionally tolerant of permission denials: an unreadable subtree
# (`EACCES`/`EPERM`) is skipped, never fatal, so one locked subdirectory
# does not abort enumeration of everything else. This matches
# `Path.rglob("*")`'s own tolerance — the enumeration `list()` used before
# this scan replaced it (issue #362) — for which partial coverage of a
# bulk ingest tree is the right posture. A permission denial opening the
# store's ROOT still surfaces (see `list`): that is a misconfigured store,
# not one unreadable corner of an otherwise-fine tree.
_IGNORED_DESCEND_ERRNOS: Final = _IGNORED_STAT_ERRNOS | {errno.EACCES, errno.EPERM}


class ObjectNotFoundError(Exception):
    """The object named no longer exists — it vanished between this run's
    listing and its fetch. Neither transient nor permanent: the caller
    treats this as "skip this object this pass," not a failure."""


@dataclass(slots=True, frozen=True, kw_only=True)
class ObjectMeta:
    """One listed object's identity and the fingerprint fields ADR 0007 §9
    ranks: ``version_id`` (only when the bucket has versioning enabled),
    ``checksum`` (only when a store cheaply returns one AT LISTING TIME —
    a real S3 ``ListObjectsV2``/``ListObjectVersions`` response does not,
    so :class:`S3ObjectStore` never populates this; it exists for a store
    that can), ``size``/``last_modified``, and ``etag`` (last resort:
    many S3-compatible stores compute a multipart upload's ``ETag`` in a
    way that is not a reliable content hash)."""

    key: str
    size: int
    last_modified: datetime | None
    etag: str | None = None
    version_id: str | None = None
    checksum: str | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class FetchedObject:
    """One object's full raw bytes plus whatever content-type the store
    itself reports for it (only available once the object is actually
    fetched — a listing alone never carries it). Used as dispatch's
    fallback signal (ADR 0007 §9's "extension first, MIME as fallback")
    when a key's own extension does not already resolve to a supported
    format."""

    body: bytes
    content_type: str | None = None


FingerprintTier = Literal["version_id", "checksum", "size_last_modified", "etag"]


def object_fingerprint(meta: ObjectMeta) -> tuple[FingerprintTier, str] | None:
    """ADR 0007 §9's checkpoint-fingerprint priority, applied to one listed
    object: ``version_id`` > ``checksum`` > ``(size, last_modified)`` >
    ``etag`` alone > ``None`` (no usable fingerprint at all — the object is
    always re-fetched). Never ``etag`` as the FIRST choice, per #217's
    explicit requirement."""
    if meta.version_id:
        return ("version_id", meta.version_id)
    if meta.checksum:
        return ("checksum", meta.checksum)
    if meta.last_modified is not None:
        return ("size_last_modified", f"{meta.size}:{meta.last_modified.isoformat()}")
    if meta.etag:
        return ("etag", meta.etag)
    return None


class ObjectStore(Protocol):
    """The structural interface :class:`S3ObjectStore`/:class:`FileObjectStore`
    both implement — a ``Protocol``, not an ABC, the same extension-point
    convention :class:`~taguru_langchain.ingest_connectors.protocol.Connector`
    and :class:`~taguru_langchain.checkpoints.CheckpointStore` already use."""

    @property
    def base_uri(self) -> str:
        """The store's own root, e.g. ``s3://my-bucket`` or
        ``file:///abs/dir`` — the prefix every source id this store's
        objects mint is built from (ADR 0007 §6.1: ``s3://bucket/key``)."""
        ...

    def list(self, prefix: str) -> Iterator[ObjectMeta]:
        """Every object whose key starts with ``prefix``, oldest-listing-API
        pagination handled internally. Raises :class:`TransientStoreError`/
        :class:`PermanentStoreError` on failure; never raises
        :class:`ObjectNotFoundError` (a prefix that matches nothing is an
        empty iterator, not a missing-object error)."""
        ...

    def get(self, key: str, *, version_id: str | None = None) -> FetchedObject:
        """The full raw bytes of one object, plus its content-type if the
        store knows one. Raises :class:`ObjectNotFoundError` if ``key`` (or
        ``version_id``) no longer exists, :class:`TransientStoreError`/
        :class:`PermanentStoreError` on any other failure."""
        ...

    def object_tags(self, key: str) -> Mapping[str, str]:
        """``key``'s object tags. Degrades to ``{}`` when tags specifically
        are unreadable (e.g. ``s3:GetObjectTagging`` denied while the
        object itself is readable) — never fails the object over a
        tags-only permission gap (ADR 0007 §9)."""
        ...


# ---------------------------------------------------------------------------
# file:// — stdlib only. The test/air-gapped backend (issue #351's own
# "minio/localstack を使わず file:// バケットでテスト" requirement), the same
# role `src/ship.rs`'s own `file://` scheme plays for replication.
# ---------------------------------------------------------------------------


class FileObjectStore:
    """A directory tree addressed the way an S3 bucket is: ``list(prefix)``
    matches on the relative POSIX path as a plain string prefix (not a
    directory boundary — ``prefix="2026/q1"`` matches ``2026/q1-report.pdf``
    too, mirroring S3's own prefix semantics), never mtime/size beyond what
    ``stat`` gives for free. ``etag``/``version_id``/``checksum`` are always
    ``None`` — a local directory has none of those — so
    :func:`object_fingerprint` degrades to the ``(size, last_modified)``
    tier here, exactly the tier real (non-versioned) S3 buckets use too."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        # The bucket must exist before anything reads from it — hold
        # file:// to the same contract `src/ship.rs:196-204` already does
        # for replication, instead of silently mkdir-ing a typo into a
        # fresh empty "bucket".
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory}: object-storage bucket directory does not exist")
        self._directory = directory.resolve()

    @property
    def base_uri(self) -> str:
        return f"file://{self._directory.as_posix()}"

    def list(self, prefix: str) -> Iterator[ObjectMeta]:
        entries = sorted(self._scan(self._directory), key=lambda item: item[0])
        for path, info in entries:
            key = path.relative_to(self._directory).as_posix()
            if not key.startswith(prefix):
                continue
            yield ObjectMeta(
                key=key,
                size=info.st_size,
                last_modified=datetime.fromtimestamp(info.st_mtime, tz=timezone.utc),
            )

    def _scan(self, directory: Path) -> Iterator[tuple[Path, os.stat_result]]:
        """Recurses ``directory`` via ``os.scandir``, yielding each regular
        file's path together with its own ``os.stat_result`` — one stat
        syscall per entry (``DirEntry.stat()`` is cached on the entry
        itself, reused here for both the file/directory type test AND the
        metadata :meth:`list` needs) in place of the previous
        ``sorted(directory.rglob("*"))`` plus ``Path.is_file()``/
        ``Path.stat()``'s own two separate stat calls per path.

        A symlinked directory is never recursed into —
        ``entry.is_dir(follow_symlinks=False)`` is true only for a REAL
        directory entry, matching ``Path.rglob("*")``'s own refusal to
        descend into one. A symlink to a regular file is still followed
        for its own stat (``entry.stat(follow_symlinks=True)``), matching
        ``Path.is_file()``/``Path.stat()``'s own default of following
        symlinks — so it is still yielded, under its own symlink path,
        exactly as the previous implementation yielded it. An unreadable
        subdirectory is skipped rather than aborting the whole scan (see
        ``_IGNORED_DESCEND_ERRNOS``)."""
        with os.scandir(directory) as scanned:
            for entry in scanned:
                path = Path(entry.path)
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                    if not is_directory:
                        info = entry.stat(follow_symlinks=True)
                except OSError as error:
                    if error.errno in _IGNORED_STAT_ERRNOS:
                        continue
                    raise
                if is_directory:
                    # Opening the subdirectory (its own `os.scandir`) is what
                    # can raise EACCES, and it does so lazily as this
                    # `yield from` first drives the sub-generator — hence the
                    # guard is here, around the descent, not the stat above.
                    try:
                        yield from self._scan(path)
                    except OSError as error:
                        if error.errno in _IGNORED_DESCEND_ERRNOS:
                            continue
                        raise
                    continue
                if S_ISREG(info.st_mode):
                    yield path, info

    def get(self, key: str, *, version_id: str | None = None) -> FetchedObject:
        del version_id  # file:// has no versions; a caller never passes one here
        path = self._resolve(key)
        try:
            body = path.read_bytes()
        except FileNotFoundError as error:
            raise ObjectNotFoundError(str(error)) from error
        except OSError as error:
            raise TransientStoreError(str(error)) from error
        return FetchedObject(body=body, content_type=mimetypes.guess_type(path.name)[0])

    def object_tags(self, key: str) -> Mapping[str, str]:
        del key  # a local directory carries no object tags at all
        return {}

    def _resolve(self, key: str) -> Path:
        resolved = (self._directory / key).resolve()
        if resolved != self._directory and self._directory not in resolved.parents:
            raise ValueError(f"{key!r} escapes the bucket directory")
        return resolved


# ---------------------------------------------------------------------------
# s3:// — optional boto3 dependency (the `s3` extra), the standard AWS
# credential chain only (mirroring `AmazonS3Builder::from_env()`,
# `src/ship.rs:212`) — never a bespoke credential path, never an argument
# this class could even accept an access key/secret through.
# ---------------------------------------------------------------------------

# Explicitly named per ADR 0007 §9: "invalid/expired/revoked credentials,
# and object-level permission denial." Everything NOT named here defaults
# to transient (below) — the same posture `src/ship.rs`'s `ShipError::Io`
# already takes ("assumed transient"), narrowed only by this explicit
# carve-out, never widened by guessing at unfamiliar codes.
_PERMANENT_ERROR_CODES: Final = frozenset(
    {
        "AccessDenied",
        "AllAccessDisabled",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "ExpiredToken",
        "TokenRefreshRequired",
        "InvalidToken",
        "AuthFailure",
        "UnauthorizedAccess",
        "AccountProblem",
        "InvalidSecurity",
        # A missing bucket is a configuration error (a typo, a bucket
        # never created), not one object vanishing mid-pass — it must
        # NOT be classified as `ObjectNotFoundError`, which `list()`'s own
        # contract promises never to raise (below).
        "NoSuchBucket",
    }
)

_NOT_FOUND_ERROR_CODES: Final = frozenset({"NoSuchKey", "NoSuchVersion", "404"})


def _classify_client_error(error: ClientError) -> Exception:
    response = getattr(error, "response", {}) or {}
    code = response.get("Error", {}).get("Code", "")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    # Code-based permanent classification (`NoSuchBucket` included) wins
    # over the blanket "status == 404" not-found check below — S3 answers
    # a missing bucket with HTTP 404 too, and a bare status code alone
    # cannot distinguish "this bucket doesn't exist" from "this object
    # doesn't exist."
    if code in _PERMANENT_ERROR_CODES:
        return PermanentStoreError(str(error))
    if code in _NOT_FOUND_ERROR_CODES or status == 404:
        return ObjectNotFoundError(str(error))
    if status in (401, 403):
        return PermanentStoreError(str(error))
    return TransientStoreError(str(error))


class S3ObjectStore:
    """Reads/lists one S3 (or S3-compatible) bucket via ``boto3``, built
    from the standard AWS credential chain only — ``endpoint_url`` (for an
    S3-compatible service, e.g. MinIO/R2), ``region_name``, and
    ``profile_name`` are the only construction knobs, and none of them is
    credential material itself; there is deliberately no parameter this
    class could accept an access key or secret through.

    ``client`` is an escape hatch for a caller who already has a configured
    ``boto3`` S3 client (custom retry/timeout config, or — this package's
    own tests — a ``botocore.stub.Stubber``-wrapped client with no network
    access at all) — mutually exclusive with ``endpoint_url``/
    ``region_name``/``profile_name``, which exist only to build one."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        client: S3Client | None = None,
    ) -> None:
        if client is not None:
            if endpoint_url is not None or region_name is not None or profile_name is not None:
                raise ValueError(
                    "client is mutually exclusive with endpoint_url/region_name/profile_name"
                )
            self._bucket = bucket
            self._client = client
            return
        if boto3 is None:
            raise ImportError(
                'S3ObjectStore requires boto3 — install with pip install "langchain-taguru[s3]"'
            ) from _BOTO3_IMPORT_ERROR
        if endpoint_url is not None:
            parsed = urlsplit(endpoint_url)
            if parsed.username or parsed.password:
                raise ValueError("endpoint_url must not embed credentials")
        self._bucket = bucket
        session = boto3.Session(profile_name=profile_name)
        self._client = session.client("s3", endpoint_url=endpoint_url, region_name=region_name)

    @property
    def base_uri(self) -> str:
        return f"s3://{self._bucket}"

    def _versioning_enabled(self) -> bool:
        # A best-effort check, not itself part of §9's two-way retry
        # split: ANY failure here (permission denial included) degrades to
        # "assume not versioned" rather than propagating, since a listing
        # this class can otherwise complete must not fail over a narrower
        # permission this one optional check needs.
        try:
            response = self._client.get_bucket_versioning(Bucket=self._bucket)
        except Exception:  # noqa: BLE001 - degrade to the plain listing path
            return False
        return response.get("Status") == "Enabled"

    def list(self, prefix: str) -> Iterator[ObjectMeta]:
        try:
            if self._versioning_enabled():
                yield from self._list_versions(prefix)
            else:
                yield from self._list_v2(prefix)
        except ClientError as error:
            raise _classify_client_error(error) from error
        except NoCredentialsError as error:
            raise PermanentStoreError(str(error)) from error
        except (
            EndpointConnectionError,
            ConnectTimeoutError,
            ReadTimeoutError,
            ConnectionClosedError,
        ) as error:
            raise TransientStoreError(str(error)) from error

    def _list_v2(self, prefix: str) -> Iterator[ObjectMeta]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield ObjectMeta(
                    key=obj["Key"],
                    size=obj["Size"],
                    last_modified=obj.get("LastModified"),
                    etag=_clean_etag(obj.get("ETag")),
                )

    def _list_versions(self, prefix: str) -> Iterator[ObjectMeta]:
        # `IsLatest` alone selects each key's current content version and
        # naturally excludes a key whose latest entry is a delete marker
        # (delete markers are returned separately, under `DeleteMarkers`,
        # never inside `Versions`) — so a deleted-then-recreated-elsewhere
        # key never reappears here as a false "still present."
        paginator = self._client.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Versions", []):
                if not obj.get("IsLatest"):
                    continue
                yield ObjectMeta(
                    key=obj["Key"],
                    size=obj["Size"],
                    last_modified=obj.get("LastModified"),
                    etag=_clean_etag(obj.get("ETag")),
                    version_id=obj.get("VersionId"),
                )

    def get(self, key: str, *, version_id: str | None = None) -> FetchedObject:
        try:
            response = (
                self._client.get_object(Bucket=self._bucket, Key=key, VersionId=version_id)
                if version_id is not None
                else self._client.get_object(Bucket=self._bucket, Key=key)
            )
            body: bytes = response["Body"].read()
            return FetchedObject(body=body, content_type=response.get("ContentType"))
        except ClientError as error:
            raise _classify_client_error(error) from error
        except NoCredentialsError as error:
            raise PermanentStoreError(str(error)) from error
        except (
            EndpointConnectionError,
            ConnectTimeoutError,
            ReadTimeoutError,
            ConnectionClosedError,
        ) as error:
            raise TransientStoreError(str(error)) from error

    def object_tags(self, key: str) -> Mapping[str, str]:
        try:
            response = self._client.get_object_tagging(Bucket=self._bucket, Key=key)
        except ClientError as error:
            classified = _classify_client_error(error)
            if isinstance(classified, PermanentStoreError):
                # Tags specifically are unreadable (e.g. a policy that
                # grants s3:GetObject but not s3:GetObjectTagging) — the
                # object itself is still readable, so this degrades to "no
                # tags" rather than failing the whole object (ADR 0007 §9).
                return {}
            if isinstance(classified, ObjectNotFoundError):
                return {}
            raise classified from error
        except NoCredentialsError:
            return {}
        return {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}


def _clean_etag(etag: str | None) -> str | None:
    """S3 wraps ``ETag`` in literal double quotes in every listing/response
    API; strip them so a caller comparing values never has to know that."""
    if etag is None:
        return None
    return etag.strip('"') or None


def open_object_store(
    url: str,
    *,
    endpoint_url: str | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
) -> tuple[ObjectStore, str]:
    """Opens the store ``url`` names, mirroring `src/ship.rs:176`'s own
    ``open_store`` in shape AND scheme set (``s3://``, ``gs://``,
    ``az://``, ``file://``): parse the URL, dispatch on scheme, return
    ``(store, prefix)`` — a caller lists ``store.list(prefix)`` to
    enumerate exactly what the URL named. For the cloud schemes the
    prefix is the URL's own path (the bucket/container is the netloc);
    for ``file://`` the whole path IS the bucket directory, so the
    prefix is always ``""``. ``endpoint_url``/``region_name``/
    ``profile_name`` apply to ``s3://`` only (an S3-compatible endpoint,
    or a non-default region/profile) and are rejected everywhere else —
    ``gs://``/``az://`` read their standard credential chains, and a
    caller needing an emulator endpoint or a preconfigured client
    constructs :class:`~taguru_langchain.ingest_connectors.
    objectstore_gcs.GCSObjectStore`/:class:`~taguru_langchain.
    ingest_connectors.objectstore_azure.AzureBlobObjectStore` directly.
    The two cloud imports are lazy, so neither optional extra has to be
    installed to open a URL of the other's scheme."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme == "s3":
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"{url}: s3:// URL must name a bucket, e.g. s3://my-bucket/prefix")
        prefix = unquote(parsed.path).lstrip("/")
        store: ObjectStore = S3ObjectStore(
            bucket, endpoint_url=endpoint_url, region_name=region_name, profile_name=profile_name
        )
        return store, prefix
    if endpoint_url is not None or region_name is not None or profile_name is not None:
        raise ValueError(
            f"{scheme}:// does not accept endpoint_url/region_name/profile_name — "
            "construct the store class directly for a custom endpoint or client"
        )
    if scheme == "gs":
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"{url}: gs:// URL must name a bucket, e.g. gs://my-bucket/prefix")
        from .objectstore_gcs import GCSObjectStore

        return GCSObjectStore(bucket), unquote(parsed.path).lstrip("/")
    if scheme == "az":
        container = parsed.netloc
        if not container:
            raise ValueError(
                f"{url}: az:// URL must name a container, e.g. az://my-container/prefix"
            )
        from .objectstore_azure import AzureBlobObjectStore

        return AzureBlobObjectStore(container), unquote(parsed.path).lstrip("/")
    if scheme == "file":
        directory = Path(unquote(parsed.path))
        return FileObjectStore(directory), ""
    raise ValueError(
        f"{url}: unsupported object-storage scheme — use s3://, gs://, az://, or file://"
    )
