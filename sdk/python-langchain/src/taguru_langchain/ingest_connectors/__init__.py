"""Standard ingest connectors (ADR 0007, issue #347): the normalized
document contract every connector — present or future — produces, and the
reference ``.md``/``.txt``/``.pdf``/``.html``/``.docx``/``.pptx``/S3
connectors that prove the contract reaches
:class:`~taguru_langchain.ingest.TaguruIngester` end to end.

A submodule beside ``ingest.py``, per ADR 0007 §3/§4's packaging decision —
not a new top-level package, and no new Rust dependency anywhere: parsing a
PDF/HTML/DOCX/PPTX/S3 object (#348-#352) stays entirely client-side, exactly
as this module's own ``.md``/``.txt`` reference connector already does.

Eleven pieces:

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
  ``http(s)://`` URLs, stdlib-only parsing), :class:`DocxConnector`
  (``docx.py``, issue #350, optional ``python-docx`` dependency via the
  ``docx`` extra), and :class:`PptxConnector` (``pptx.py``, issue #352,
  optional ``python-pptx`` dependency via the ``pptx`` extra) — the
  reference implementations.
- :class:`OcrAdapter` (``ocr.py``, ADR 0007 §10, issue #352) — the external
  OCR engine boundary a connector calls out to when one is configured; no
  OCR engine ships here or in any connector. :class:`PdfConnector` is the
  one connector that calls out to a configured adapter today.
- :mod:`~taguru_langchain.ingest_connectors.objectstore` (issue #351) — the
  object-storage boundary: :class:`ObjectStore`, its ``s3://``
  (:class:`S3ObjectStore`, optional ``boto3`` dependency via the ``s3``
  extra) and ``file://`` (:class:`FileObjectStore`, stdlib-only, the
  test/air-gapped backend) implementations, and
  :func:`object_fingerprint` (ADR 0007 §9's checkpoint-fingerprint
  priority). Issue #414 completes `src/ship.rs`'s scheme set with
  ``gs://`` (:class:`GCSObjectStore`, the ``gcs`` extra) and ``az://``
  (:class:`AzureBlobObjectStore`, the ``azure`` extra), each in its own
  module, each reading only its cloud's standard credential chain.
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
- :mod:`~taguru_langchain.ingest_connectors.observability` (ADR 0007 §11,
  issue #353) — the one event/summary shape every connector driver emits:
  :class:`RunReport`, :class:`SourceEvent`, :class:`RunRecorder` (the
  shared bookkeeping), and :class:`SourceEventSink` (the append-only
  per-source JSONL sidecar). ``S3SyncReport`` is now a deprecated alias of
  :class:`RunReport`.
- :func:`sync_references`/:func:`plan_references`/:func:`default_connectors`
  (``references.py``, issue #353) — the cross-connector driver for every
  local-file/``http(s)://`` reference (``.md``/``.txt``/PDF/HTML/DOCX/
  PPTX): dispatch, ``dry_run``, and a :class:`RunReport`, the non-S3 twin
  of :func:`sync_object_storage`.
"""

from __future__ import annotations

from .bridge import (
    aingest_connector_document,
    aingest_connector_documents,
    ingest_connector_document,
    ingest_connector_documents,
)
from .checkpoint import ConnectorCheckpoint, FileProbe, FileProbeCheckpoint
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
from .objectstore_azure import AzureBlobObjectStore
from .objectstore_gcs import GCSObjectStore
from .observability import (
    PHASES,
    RUN_SUMMARY_VERSION,
    Phase,
    RunRecorder,
    RunReport,
    S3SyncReport,
    SourceEvent,
    SourceEventSink,
)
from .ocr import OcrAdapter, OcrRecoveredUnit, OcrRequest, OcrResult
from .pdf import PdfConnector
from .pptx import PptxConnector
from .protocol import Connector
from .references import (
    ReferenceKind,
    ReferencePlan,
    default_connectors,
    plan_references,
    sync_references,
)
from .s3 import DeletionPolicy, S3Connector, S3ObjectCheckpoint, sync_object_storage
from .sources import (
    SourceIdRegistry,
    canonicalize_url,
    check_source_id,
    file_source_id,
    sub_source_id,
)
from .text import TextFileConnector
from .watch import watch_directory

__all__ = [
    "CONNECTOR_DOCUMENT_VERSION",
    "DIAGNOSTIC_CODES",
    "PHASES",
    "RUN_SUMMARY_VERSION",
    "Connector",
    "ConnectorCheckpoint",
    "AzureBlobObjectStore",
    "ConnectorDocument",
    "ConnectorMetadata",
    "DeletionPolicy",
    "Diagnostic",
    "DiagnosticCode",
    "DocxConnector",
    "FetchedObject",
    "FileObjectStore",
    "FileProbe",
    "FileProbeCheckpoint",
    "FingerprintInputs",
    "FingerprintTier",
    "GCSObjectStore",
    "HtmlConnector",
    "LocatorEntry",
    "ObjectMeta",
    "ObjectNotFoundError",
    "ObjectStore",
    "OcrAdapter",
    "OcrRecoveredUnit",
    "OcrRequest",
    "OcrResult",
    "PdfConnector",
    "PermanentStoreError",
    "Phase",
    "PptxConnector",
    "ReferenceKind",
    "ReferencePlan",
    "RunRecorder",
    "RunReport",
    "S3Connector",
    "S3ObjectCheckpoint",
    "S3ObjectStore",
    "S3SyncReport",
    "SectionEntry",
    "SourceEvent",
    "SourceEventSink",
    "SourceIdRegistry",
    "TextFileConnector",
    "TransientStoreError",
    "aingest_connector_document",
    "aingest_connector_documents",
    "canonicalize_url",
    "check_source_id",
    "default_connectors",
    "file_source_id",
    "ingest_connector_document",
    "ingest_connector_documents",
    "object_fingerprint",
    "open_object_store",
    "options_digest",
    "plan_references",
    "sub_source_id",
    "sync_object_storage",
    "sync_references",
    "watch_directory",
]
