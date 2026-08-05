"""The ``az://`` backend of the object-storage boundary (ADR 0007 §13's
deferred Azure Blob connector, issue #414): :class:`AzureBlobObjectStore`
implements the same :class:`~taguru_langchain.ingest_connectors.
objectstore.ObjectStore` protocol :class:`~taguru_langchain.
ingest_connectors.objectstore.S3ObjectStore` does, as an optional
``azure-storage-blob`` + ``azure-identity`` dependency via the ``azure``
extra — completing `src/ship.rs` ``open_store``'s scheme set (``s3://``,
``gs://``, ``az://``, ``file://``) on this package's side.

Credentials come from ``DefaultAzureCredential`` only — Azure's standard
chain (environment, workload identity, managed identity, Azure CLI, …),
the analog of `AmazonS3Builder::from_env()` (`src/ship.rs:212`) and of
:class:`S3ObjectStore`'s standard-chain-only posture. ``account`` (the
storage account NAME, not a key) and ``endpoint_url`` (an Azurite or
sovereign-cloud endpoint) are the only construction knobs, and neither is
credential material; there is deliberately no parameter this class could
accept an account key or SAS token through. When ``account`` is omitted it
falls back to ``$AZURE_STORAGE_ACCOUNT_NAME`` (the variable
`MicrosoftAzureBuilder::from_env()` reads in `src/ship.rs`'s Rust path)
and then ``$AZURE_STORAGE_ACCOUNT`` (the Azure CLI's own spelling).

Mappings onto the protocol's fingerprint tiers (ADR 0007 §9):
``version_id`` is populated only when the container's versioning stamps
one on the listing — the same versioning-gated semantics as S3;
``checksum`` carries the listing's ``Content-MD5`` when the service
stored one; ``etag`` stays the last resort. ``object_tags`` maps to
**blob index tags** — Azure's direct analog of S3 object tags, a separate
API a policy can deny narrowly, so the same tags-only degrade applies.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from .objectstore import (
    FetchedObject,
    ObjectMeta,
    ObjectNotFoundError,
    PermanentStoreError,
    TransientStoreError,
)

if TYPE_CHECKING:
    from azure.storage.blob import BlobProperties, ContainerClient

try:
    from azure.core.exceptions import (
        AzureError,
        ClientAuthenticationError,
        HttpResponseError,
        ResourceNotFoundError,
    )
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import ContainerClient as _ContainerClient
except ImportError as _azure_import_error:  # pragma: no cover - exercised only without the extra
    _ContainerClient = None  # type: ignore[assignment, misc]
    DefaultAzureCredential = None  # type: ignore[assignment, misc]
    AzureError = Exception  # type: ignore[assignment, misc]
    ClientAuthenticationError = Exception  # type: ignore[assignment, misc]
    HttpResponseError = Exception  # type: ignore[assignment, misc]
    ResourceNotFoundError = Exception  # type: ignore[assignment, misc]
    _AZURE_IMPORT_ERROR: ImportError | None = _azure_import_error
else:
    _AZURE_IMPORT_ERROR = None

# ADR 0007 §9's explicit carve-out, Azure dress. `ContainerNotFound` sits
# beside the credential codes for the same reason `NoSuchBucket` does in
# the S3 backend: a missing container is a configuration error (a typo, a
# container never created), not one object vanishing mid-pass.
_PERMANENT_ERROR_CODES: Final = frozenset(
    {
        "AuthenticationFailed",
        "AuthorizationFailure",
        "AuthorizationPermissionMismatch",
        "InsufficientAccountPermissions",
        "AccountIsDisabled",
        "InvalidAuthenticationInfo",
        "ContainerNotFound",
    }
)

_PERMANENT_STATUS: Final = frozenset({401, 403})


def _error_code(error: Exception) -> str:
    code = getattr(error, "error_code", None)
    return str(code) if code else ""


def _classify(error: Exception, *, not_found: Exception) -> Exception:
    """One Azure failure → the protocol's exception vocabulary. As in the
    GCS backend, `not_found` is what "does not exist" means at this call
    site — but Azure names the missing thing itself (`ContainerNotFound`
    vs `BlobNotFound`), so a missing container classifies permanent even
    on the fetch path."""
    if isinstance(error, ClientAuthenticationError):
        # `azure-identity`'s CredentialUnavailableError subclasses this
        # too: no credential at all is as terminal as a rejected one.
        return PermanentStoreError(str(error))
    code = _error_code(error)
    if code in _PERMANENT_ERROR_CODES:
        return PermanentStoreError(str(error))
    if isinstance(error, ResourceNotFoundError):
        return not_found
    if isinstance(error, HttpResponseError):
        status = error.status_code
        if status in _PERMANENT_STATUS:
            return PermanentStoreError(str(error))
        if status == 404:
            return not_found
        return TransientStoreError(str(error))
    # ServiceRequestError (connection/DNS/timeout) and every other
    # AzureError stays transient — `src/ship.rs`'s "assumed transient"
    # default, narrowed only by the carve-outs above.
    return TransientStoreError(str(error))


def _account_url(account: str | None, endpoint_url: str | None) -> str:
    if endpoint_url is not None:
        parsed = urlsplit(endpoint_url)
        if parsed.username or parsed.password:
            raise ValueError("endpoint_url must not embed credentials")
        return endpoint_url
    account = (
        account
        or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        or os.environ.get("AZURE_STORAGE_ACCOUNT")
    )
    if not account:
        raise ValueError(
            "AzureBlobObjectStore needs the storage account name — pass account=, or set "
            "AZURE_STORAGE_ACCOUNT_NAME (the same variable the server's replication path reads)"
        )
    return f"https://{account}.blob.core.windows.net"


class AzureBlobObjectStore:
    """Reads/lists one Azure Blob container via ``azure-storage-blob``,
    authenticated through ``DefaultAzureCredential`` only (the standard
    chain) — ``account`` names the storage account, ``endpoint_url``
    overrides the whole account URL for an Azurite or sovereign-cloud
    endpoint, and neither is credential material; there is deliberately
    no parameter this class could accept an account key or SAS token
    through.

    ``client`` is the same escape hatch the S3/GCS backends document: a
    caller who already has a configured ``ContainerClient`` (custom
    retry/transport config, a connection-string-built client of their
    own, or — this package's own tests — a network-free fake) passes it
    here, mutually exclusive with ``account``/``endpoint_url``, which
    exist only to build one."""

    def __init__(
        self,
        container: str,
        *,
        account: str | None = None,
        endpoint_url: str | None = None,
        client: ContainerClient | None = None,
    ) -> None:
        if client is not None:
            if account is not None or endpoint_url is not None:
                raise ValueError("client is mutually exclusive with account/endpoint_url")
            self._container = container
            self._client = client
            return
        if _ContainerClient is None:
            raise ImportError(
                "AzureBlobObjectStore requires azure-storage-blob and azure-identity — "
                'install with pip install "langchain-taguru[azure]"'
            ) from _AZURE_IMPORT_ERROR
        self._container = container
        self._client = _ContainerClient(
            _account_url(account, endpoint_url),
            container,
            credential=DefaultAzureCredential(),
        )

    @property
    def base_uri(self) -> str:
        return f"az://{self._container}"

    def list(self, prefix: str) -> Iterator[ObjectMeta]:
        try:
            for blob in self._client.list_blobs(name_starts_with=prefix):
                yield _meta_of(blob)
        except AzureError as error:
            # A 404 here is the CONTAINER missing — permanent, like
            # `NoSuchBucket`; `list()`'s contract never raises
            # `ObjectNotFoundError`.
            raise _classify(error, not_found=PermanentStoreError(str(error))) from error

    def get(self, key: str, *, version_id: str | None = None) -> FetchedObject:
        try:
            downloader = (
                self._client.download_blob(key, version_id=version_id)
                if version_id is not None
                else self._client.download_blob(key)
            )
            body: bytes = downloader.readall()
            settings = downloader.properties.content_settings
            return FetchedObject(body=body, content_type=settings.content_type)
        except AzureError as error:
            raise _classify(error, not_found=ObjectNotFoundError(str(error))) from error

    def object_tags(self, key: str) -> Mapping[str, str]:
        """The blob's index tags — Azure's direct analog of S3 object
        tags, down to being a separate permission (``Microsoft.Storage/…/
        blobs/tags/read``) a policy can deny while the blob itself stays
        readable, so the same tags-only degrade to ``{}`` applies (ADR
        0007 §9)."""
        try:
            tags = self._client.get_blob_client(key).get_blob_tags()
        except AzureError as error:
            classified = _classify(error, not_found=ObjectNotFoundError(str(error)))
            if isinstance(classified, TransientStoreError):
                raise classified from error
            return {}
        return dict(tags or {})


def _meta_of(blob: BlobProperties) -> ObjectMeta:
    content_md5 = getattr(blob.content_settings, "content_md5", None)
    checksum = f"md5:{bytes(content_md5).hex()}" if content_md5 else None
    return ObjectMeta(
        key=blob.name,
        size=blob.size,
        last_modified=blob.last_modified,
        etag=_clean_etag(blob.etag),
        version_id=getattr(blob, "version_id", None),
        checksum=checksum,
    )


def _clean_etag(etag: str | None) -> str | None:
    """Azure quotes ``ETag`` the same way S3 does; strip for the same
    reason `objectstore._clean_etag` documents."""
    if etag is None:
        return None
    return etag.strip('"') or None
