"""Standard ingest connectors (ADR 0007, issue #347): the normalized
document contract every connector — present or future — produces, and the
reference ``.md``/``.txt`` connector that proves the contract reaches
:class:`~taguru_langchain.ingest.TaguruIngester` end to end.

A submodule beside ``ingest.py``, per ADR 0007 §3/§4's packaging decision —
not a new top-level package, and no new Rust dependency anywhere: parsing a
PDF/HTML/DOCX/S3 object (#348-#351) stays entirely client-side, exactly as
this module's own ``.md``/``.txt`` reference connector already does.

Five pieces, one per file:

- :class:`ConnectorDocument` (``document.py``) — the wire-independent shape
  a connector produces: ``text`` plus paragraph-indexed ``locators``/
  ``sections``, ``metadata``, ``fingerprint_inputs``, and ``diagnostics``.
- Source id derivation and URL canonicalization (``sources.py``, ADR 0007
  §6.1).
- :class:`ConnectorCheckpoint` (``checkpoint.py``, ADR 0007 §6.3) — "did I
  already fetch/parse this object," composing with (never replacing)
  :class:`~taguru_langchain.checkpoints.CheckpointStore`.
- :class:`Connector` (``protocol.py``) — the structural interface a format
  connector implements, and :class:`TextFileConnector` (``text.py``), the
  reference implementation.
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
from .protocol import Connector
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
    "Diagnostic",
    "DiagnosticCode",
    "FingerprintInputs",
    "LocatorEntry",
    "SectionEntry",
    "SourceIdRegistry",
    "TextFileConnector",
    "aingest_connector_document",
    "aingest_connector_documents",
    "canonicalize_url",
    "check_source_id",
    "file_source_id",
    "ingest_connector_document",
    "ingest_connector_documents",
    "options_digest",
    "sub_source_id",
]
