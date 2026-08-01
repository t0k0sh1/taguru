"""Standard ingest connectors (ADR 0007, issue #347): the normalized
document contract every connector — present or future — produces, and the
reference ``.md``/``.txt``/``.pdf``/``.html``/``.docx``/S3 connectors that
prove the contract reaches
:class:`~taguru_langchain.ingest.TaguruIngester` end to end.

A submodule beside ``ingest.py``, per ADR 0007 §3/§4's packaging decision —
not a new top-level package, and no new Rust dependency anywhere: parsing a
PDF/HTML/DOCX/S3 object (#348-#351) stays entirely client-side, exactly as
this module's own ``.md``/``.txt`` reference connector already does.

Seven pieces:

- :class:`ConnectorDocument` (``document.py``) — the wire-independent shape
  a connector produces: ``text`` plus paragraph-indexed ``locators``/
  ``sections``, ``metadata``, ``fingerprint_inputs``, and ``diagnostics``.
- Source id derivation and URL canonicalization (``sources.py``, ADR 0007
  §6.1).
- :class:`ConnectorCheckpoint` (``checkpoint.py``, ADR 0007 §6.3) — "did I
  already fetch/parse this object," composing with (never replacing)
  :class:`~taguru_langchain.checkpoints.CheckpointStore`.
- :class:`Connector` (``protocol.py``) — the structural interface a format
  connector implements.
- :class:`TextFileConnector` (``text.py``), :class:`PdfConnector`
  (``pdf.py``, issue #348, optional ``pypdf`` dependency via the ``pdf``
  extra), :class:`HtmlConnector` (``html.py``, issue #349, local files and
  ``http(s)://`` URLs, stdlib-only parsing), and :class:`DocxConnector`
  (``docx.py``, issue #350, optional ``python-docx`` dependency via the
  ``docx`` extra) — the reference implementations.
- :mod:`~taguru_langchain.ingest_connectors.objectstore` (issue #351) — the
  object-storage boundary: :class:`ObjectStore`, its ``s3://``
  (:class:`S3ObjectStore`, optional ``boto3`` dependency via the ``s3``
  extra) and ``file://`` (:class:`FileObjectStore`, stdlib-only, the
  test/air-gapped backend) implementations, and
  :func:`object_fingerprint` (ADR 0007 §9's checkpoint-fingerprint
  priority).
- :class:`S3Connector`/:func:`sync_object_storage` (``s3.py``, issue #351)
  — dispatches one object-storage object to whichever format connector
  above handles it, and syncs a whole bucket/prefix into a
  :class:`~taguru_langchain.ingest.TaguruIngester` with a two-layer
  checkpoint and a never-destructive-by-default deletion policy.
- :func:`ingest_connector_document`/:func:`ingest_connector_documents`
  (``bridge.py``) — the one-way bridge from a :class:`ConnectorDocument`
  into ``TaguruIngester.ingest_text``; :func:`aingest_connector_document`/
  :func:`aingest_connector_documents` are the async twins, bridging into
  ``TaguruIngester.aingest_text`` instead.
"""

from __future__ import annotations

from .bridge import (
    aingest_connector_document,
    aingest_connector_documents,
    ingest_connector_document,
    ingest_connector_documents,
)
from .checkpoint import ConnectorCheckpoint
from .document import (
    CONNECTOR_DOCUMENT_VERSION,
    DIAGNOSTIC_CODES,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    DiagnosticCode,
    FingerprintInputs,
    LocatorEntry,
    SectionEntry,
    options_digest,
)
from .docx import DocxConnector
from .html import HtmlConnector
from .objectstore import (
    FetchedObject,
    FileObjectStore,
    FingerprintTier,
    ObjectMeta,
    ObjectNotFoundError,
    ObjectStore,
    PermanentStoreError,
    S3ObjectStore,
    TransientStoreError,
    object_fingerprint,
    open_object_store,
)
from .pdf import PdfConnector
from .protocol import Connector
from .s3 import (
    DeletionPolicy,
    Phase,
    S3Connector,
    S3ObjectCheckpoint,
    S3SyncReport,
    SourceEvent,
    sync_object_storage,
)
from .sources import (
    SourceIdRegistry,
    canonicalize_url,
    check_source_id,
    file_source_id,
    sub_source_id,
)
from .text import TextFileConnector

__all__ = [
    "CONNECTOR_DOCUMENT_VERSION",
    "DIAGNOSTIC_CODES",
    "Connector",
    "ConnectorCheckpoint",
    "ConnectorDocument",
    "ConnectorMetadata",
    "DeletionPolicy",
    "Diagnostic",
    "DiagnosticCode",
    "DocxConnector",
    "FetchedObject",
    "FileObjectStore",
    "FingerprintInputs",
    "FingerprintTier",
    "HtmlConnector",
    "LocatorEntry",
    "ObjectMeta",
    "ObjectNotFoundError",
    "ObjectStore",
    "PdfConnector",
    "PermanentStoreError",
    "Phase",
    "S3Connector",
    "S3ObjectCheckpoint",
    "S3ObjectStore",
    "S3SyncReport",
    "SectionEntry",
    "SourceEvent",
    "SourceIdRegistry",
    "TextFileConnector",
    "TransientStoreError",
    "aingest_connector_document",
    "aingest_connector_documents",
    "canonicalize_url",
    "check_source_id",
    "file_source_id",
    "ingest_connector_document",
    "ingest_connector_documents",
    "object_fingerprint",
    "open_object_store",
    "options_digest",
    "sub_source_id",
    "sync_object_storage",
]
