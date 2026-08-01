"""LangChain integration for the Taguru long-term semantic memory server.

Three entry points:

- :class:`TaguruRetriever` — the documented retrieval loop (graph lane +
  text lane, RRF-merged) as a LangChain ``BaseRetriever``.
- :class:`TaguruIngester` — the LangChain twin of ``taguru extract``: a chat
  model decomposes Documents into associations under the protocol's ingest
  discipline, applied via ``POST /import`` (per-source replace, idempotent).
- :mod:`taguru_langchain.ingest_connectors` (ADR 0007, issue #347) — the
  normalized document contract standard ingest connectors (``.md``/``.txt``
  today; PDF/HTML/DOCX/S3 in follow-up issues) produce, and the bridge that
  feeds a :class:`~taguru_langchain.ingest_connectors.document.
  ConnectorDocument` into :class:`TaguruIngester` instead of a bare string.

Deliberately NOT provided: a VectorStore facade (Taguru's retrieval is
structural-first; forcing it behind ``similarity_search`` would misrepresent
resolve/activate semantics), a Memory class (deprecated upstream in favor of
LangGraph state), and agent Tools (the MCP bridge ``taguru-mcp`` already
serves the identical tool vocabulary — pair it with ``langchain-mcp-adapters``).
"""

from __future__ import annotations

from ._extract import MODEL_OUTPUT_JSON_SCHEMA
from .checkpoints import CheckpointStore, FilesystemCheckpointStore
from .events import (
    AttemptFailed,
    AttemptStarted,
    ChunkCompleted,
    ChunkStarted,
    DocumentStarted,
    EmbeddingRefreshCompleted,
    EmbeddingRefreshStarted,
    EmbeddingRefreshWarning,
    ImportCompleted,
    ImportStarted,
    IngestEvent,
    IngestEventCallback,
    ProviderMetadata,
)
from .ingest import IngestOutcome, TaguruIngester
from .ingest_connectors import (
    CONNECTOR_DOCUMENT_VERSION,
    DIAGNOSTIC_CODES,
    Connector,
    ConnectorCheckpoint,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    DiagnosticCode,
    FingerprintInputs,
    LocatorEntry,
    SectionEntry,
    SourceIdRegistry,
    TextFileConnector,
    aingest_connector_document,
    aingest_connector_documents,
    canonicalize_url,
    check_source_id,
    file_source_id,
    ingest_connector_document,
    ingest_connector_documents,
    options_digest,
    sub_source_id,
)
from .retrievers import TaguruRetriever

__version__ = "0.6.0"

__all__ = [
    "TaguruRetriever",
    "TaguruIngester",
    "IngestOutcome",
    "MODEL_OUTPUT_JSON_SCHEMA",
    "CheckpointStore",
    "FilesystemCheckpointStore",
    "IngestEvent",
    "IngestEventCallback",
    "ProviderMetadata",
    "DocumentStarted",
    "ChunkStarted",
    "AttemptStarted",
    "AttemptFailed",
    "ChunkCompleted",
    "ImportStarted",
    "ImportCompleted",
    "EmbeddingRefreshStarted",
    "EmbeddingRefreshCompleted",
    "EmbeddingRefreshWarning",
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
    "__version__",
]
