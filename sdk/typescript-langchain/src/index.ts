/**
 * LangChain.js integration for the Taguru long-term semantic memory server.
 * The mechanical mirror of the Python `taguru_langchain` package — identical
 * structure, vocabulary, and behavior (TS is async-only, so no separate
 * `a`-prefixed methods).
 */

export { TaguruRetriever, type TaguruRetrieverFields } from "./retrievers.js";
export { TaguruIngester, type TaguruIngesterFields, type IngestOutcome } from "./ingest.js";
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
