/**
 * LangChain.js integration for the Taguru long-term semantic memory server.
 * The mechanical mirror of the Python `taguru_langchain` package — identical
 * structure, vocabulary, and behavior (TS is async-only, so no separate
 * `a`-prefixed methods).
 */

export { TaguruRetriever, type TaguruRetrieverFields } from "./retrievers.js";
export {
  TaguruIngester,
  type TaguruIngesterFields,
  type IngestOutcome,
  type ShouldStop,
} from "./ingest.js";
export { FilesystemCheckpointStore, type CheckpointStore } from "./checkpoints.js";
export { MODEL_OUTPUT_JSON_SCHEMA } from "./extract.js";
export type {
  IngestEvent,
  IngestEventCallback,
  ProviderMetadata,
  DocumentStarted,
  ChunkStarted,
  AttemptStage,
  AttemptStarted,
  AttemptFailed,
  ChunkCompleted,
  ImportStarted,
  ImportCompleted,
  EmbeddingRefreshStarted,
  EmbeddingRefreshCompleted,
  EmbeddingRefreshWarning,
} from "./events.js";

// Standard ingest connectors (ADR 0007, issues #347-#354/#414; TypeScript
// parity: issue #415) — the same root-level surface the Python package
// re-exports from `taguru_langchain.ingest_connectors`.
export {
  CONNECTOR_DOCUMENT_VERSION,
  DIAGNOSTIC_CODES,
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  type DiagnosticCode,
  FingerprintInputs,
  LocatorEntry,
  SectionEntry,
  optionsDigest,
} from "./ingest-connectors/document.js";
export type { Connector } from "./ingest-connectors/protocol.js";
export {
  SourceIdRegistry,
  canonicalizeUrl,
  checkSourceId,
  fileSourceId,
  subSourceId,
} from "./ingest-connectors/sources.js";
export {
  ConnectorCheckpoint,
  FileProbe,
  FileProbeCheckpoint,
} from "./ingest-connectors/checkpoint.js";
export { TextFileConnector } from "./ingest-connectors/text.js";
export { HtmlConnector, type HtmlConnectorOptions } from "./ingest-connectors/html.js";
export { PdfConnector } from "./ingest-connectors/pdf.js";
export { DocxConnector, type DocxConnectorOptions } from "./ingest-connectors/docx.js";
export { PptxConnector, type PptxConnectorOptions } from "./ingest-connectors/pptx.js";
export {
  type OcrAdapter,
  OcrRecoveredUnit,
  OcrRequest,
  OcrResult,
} from "./ingest-connectors/ocr.js";
export {
  FetchedObject,
  FileObjectStore,
  type FingerprintTier,
  ObjectMeta,
  ObjectNotFoundError,
  type ObjectStore,
  PermanentStoreError,
  TransientStoreError,
  objectFingerprint,
  openObjectStore,
} from "./ingest-connectors/objectstore.js";
export { S3ObjectStore } from "./ingest-connectors/objectstore-s3.js";
export { GCSObjectStore } from "./ingest-connectors/objectstore-gcs.js";
export { AzureBlobObjectStore } from "./ingest-connectors/objectstore-azure.js";
export {
  PHASES,
  RUN_SUMMARY_VERSION,
  type Phase,
  RunRecorder,
  RunReport,
  S3SyncReport,
  SourceEvent,
  SourceEventSink,
} from "./ingest-connectors/observability.js";
export {
  ingestConnectorDocument,
  ingestConnectorDocuments,
} from "./ingest-connectors/bridge.js";
export {
  type ReferenceKind,
  ReferencePlan,
  defaultConnectors,
  planReferences,
  syncReferences,
} from "./ingest-connectors/references.js";
export {
  type DeletionPolicy,
  S3Connector,
  S3ObjectCheckpoint,
  syncObjectStorage,
} from "./ingest-connectors/s3.js";
export { watchDirectory } from "./ingest-connectors/watch.js";
