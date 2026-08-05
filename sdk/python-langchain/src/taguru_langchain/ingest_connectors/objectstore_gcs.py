"""The ``gs://`` backend of the object-storage boundary (ADR 0007 §13's
deferred GCS connector, issue #414): :class:`GCSObjectStore` implements the
same :class:`~taguru_langchain.ingest_connectors.objectstore.ObjectStore`
protocol :class:`~taguru_langchain.ingest_connectors.objectstore.
S3ObjectStore` does, as an optional ``google-cloud-storage`` dependency via
the ``gcs`` extra — so `src/ship.rs`'s ``open_store`` scheme set (``s3://``,
``gs://``, ``az://``, ``file://``) and this package's
:func:`~taguru_langchain.ingest_connectors.objectstore.open_object_store`
stop disagreeing about which clouds exist.

Credentials come from Application Default Credentials only — the GCS
analog of `AmazonS3Builder::from_env()` (`src/ship.rs:212`) and of
:class:`S3ObjectStore`'s standard-chain-only posture. ``project`` is the
one construction knob and is not credential material; there is
deliberately no parameter this class could accept a service-account key
through.

Two GCS-specific mappings, both strictly better fingerprints than S3's:

- ``version_id`` carries the blob's **generation** on every listing —
  GCS stamps a new generation on every content overwrite whether or not
  the bucket has object versioning enabled, so the top fingerprint tier
  (ADR 0007 §9) is always available here, not gated on a bucket setting
  the way S3's ``VersionId`` is. Fetching a listed generation after the
  object was overwritten (versioning off) surfaces as
  :class:`~taguru_langchain.ingest_connectors.objectstore.
  ObjectNotFoundError` — exactly the "skip this object this pass" the
  protocol already means by it; the new content lists under its new
  generation on the next pass.
- ``checksum`` is populated AT LISTING TIME (``crc32c``, falling back to
  ``md5_hash``) — the tier :class:`~taguru_langchain.ingest_connectors.
  objectstore.ObjectMeta` documents as "only when a store cheaply
  returns one at listing time; real S3 does not." GCS does.

``object_tags`` maps to the blob's **custom metadata** (user key/value
pairs): GCS has no separate object-tagging API, and custom metadata is
the same operator-attached, non-content key/value surface S3 tags are —
the downstream use (source tags, ``_mapped_tags``) reads identically.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Final

from .objectstore import (
    FetchedObject,
    ObjectMeta,
    ObjectNotFoundError,
    PermanentStoreError,
    TransientStoreError,
)

if TYPE_CHECKING:
    from google.cloud.storage import Blob, Client

try:
    from google.api_core.exceptions import GoogleAPICallError
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import storage
except ImportError as _gcs_import_error:  # pragma: no cover - exercised only without the extra
    storage = None
    GoogleAPICallError = Exception  # type: ignore[assignment, misc]
    GoogleAuthError = Exception  # type: ignore[assignment, misc]
    _GCS_IMPORT_ERROR: ImportError | None = _gcs_import_error
else:
    _GCS_IMPORT_ERROR = None

# ADR 0007 §9's explicit carve-out, GCS dress: credential problems and
# permission denial are terminal; everything else keeps `src/ship.rs`'s
# "assumed transient" default. 404 is NOT here — a missing bucket is
# permanent but a missing object is `ObjectNotFoundError`, and only the
# call site knows which of the two it asked about (`_classify` takes
# `not_found`).
_PERMANENT_STATUS: Final = frozenset({401, 403})


def _classify(error: Exception, *, not_found: Exception) -> Exception:
    """One GCS failure → the protocol's exception vocabulary. `not_found`
    is what a 404 means AT THIS CALL SITE: `PermanentStoreError` for a
    listing (the bucket itself is a typo or was never created — retrying
    forever would mask a fixable configuration problem),
    `ObjectNotFoundError` for a fetch (the object vanished between
    listing and get — skip it this pass)."""
    if isinstance(error, GoogleAuthError):
        return PermanentStoreError(str(error))
    if isinstance(error, GoogleAPICallError):
        status = error.code
        if status in _PERMANENT_STATUS:
            return PermanentStoreError(str(error))
        if status == 404:
            return not_found
        return TransientStoreError(str(error))
    # Everything unnamed stays transient (connection resets, timeouts —
    # requests-level failures all subclass OSError), the same narrowed-
    # only-by-carve-out posture `_classify_client_error` takes for S3.
    return TransientStoreError(str(error))


class GCSObjectStore:
    """Reads/lists one GCS bucket via ``google-cloud-storage``, built from
    Application Default Credentials only — ``project`` is the only
    construction knob, and it is not credential material; there is
    deliberately no parameter this class could accept a service-account
    key through.

    ``client`` is the same escape hatch :class:`~taguru_langchain.
    ingest_connectors.objectstore.S3ObjectStore` documents: a caller who
    already has a configured ``google.cloud.storage.Client`` (custom
    retry/timeout config, an emulator endpoint, or — this package's own
    tests — a network-free fake) passes it here, mutually exclusive with
    ``project``, which exists only to build one."""

    def __init__(
        self,
        bucket: str,
        *,
        project: str | None = None,
        client: Client | None = None,
    ) -> None:
        if client is not None:
            if project is not None:
                raise ValueError("client is mutually exclusive with project")
        elif storage is None:
            raise ImportError(
                "GCSObjectStore requires google-cloud-storage — install with "
                'pip install "langchain-taguru[gcs]"'
            ) from _GCS_IMPORT_ERROR
        self._bucket = bucket
        self._project = project
        self._client = client

    @property
    def base_uri(self) -> str:
        return f"gs://{self._bucket}"

    def _client_or_build(self) -> Client:
        """Built on first use, not in ``__init__``: ``storage.Client``
        resolves Application Default Credentials eagerly, and a missing/
        broken credential is an OPERATION-time :class:`PermanentStoreError`
        under ADR 0007 §9's split — constructing a store object should
        neither touch the environment nor raise anything but the
        misconfiguration errors ``__init__`` already names."""
        if self._client is None:
            try:
                self._client = storage.Client(project=self._project)
            except (GoogleAuthError, OSError) as error:
                raise PermanentStoreError(str(error)) from error
        return self._client

    def list(self, prefix: str) -> Iterator[ObjectMeta]:
        client = self._client_or_build()
        try:
            for blob in client.list_blobs(self._bucket, prefix=prefix):
                yield _meta_of(blob)
        except (GoogleAuthError, GoogleAPICallError, OSError) as error:
            # A 404 here is the BUCKET missing — a configuration error
            # (`NoSuchBucket`'s posture in the S3 backend), never an
            # `ObjectNotFoundError`, which `list()`'s contract promises
            # not to raise.
            raise _classify(error, not_found=PermanentStoreError(str(error))) from error

    def get(self, key: str, *, version_id: str | None = None) -> FetchedObject:
        generation = int(version_id) if version_id is not None else None
        client = self._client_or_build()
        try:
            blob = client.bucket(self._bucket).get_blob(key, generation=generation)
            if blob is None:
                raise ObjectNotFoundError(f"gs://{self._bucket}/{key}: no such object")
            body: bytes = blob.download_as_bytes()
            return FetchedObject(body=body, content_type=blob.content_type)
        except (GoogleAuthError, GoogleAPICallError, OSError) as error:
            raise _classify(error, not_found=ObjectNotFoundError(str(error))) from error

    def object_tags(self, key: str) -> Mapping[str, str]:
        """The blob's custom metadata — GCS's operator-attached key/value
        surface, standing in for S3's object tags (no separate tagging
        API exists to be denied narrowly, but the same degrade applies:
        any failure reading it yields ``{}``, never a failed object)."""
        try:
            blob = self._client_or_build().bucket(self._bucket).get_blob(key)
        except PermanentStoreError:
            return {}
        except (GoogleAuthError, GoogleAPICallError, OSError) as error:
            classified = _classify(error, not_found=ObjectNotFoundError(str(error)))
            if isinstance(classified, TransientStoreError):
                # Transient failures re-raise (retry next pass) exactly as
                # the S3 backend's `object_tags` does; only the permanent/
                # not-found cases degrade to "no tags."
                raise classified from error
            return {}
        if blob is None or blob.metadata is None:
            return {}
        return dict(blob.metadata)


def _meta_of(blob: Blob) -> ObjectMeta:
    generation = blob.generation
    if blob.crc32c:
        checksum: str | None = f"crc32c:{blob.crc32c}"
    elif blob.md5_hash:
        checksum = f"md5:{blob.md5_hash}"
    else:
        checksum = None
    return ObjectMeta(
        key=blob.name or "",
        size=blob.size or 0,
        last_modified=blob.updated,
        etag=blob.etag,
        version_id=str(generation) if generation else None,
        checksum=checksum,
    )
