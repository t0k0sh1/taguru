/**
 * Progress/diagnostic events for TaguruIngester (ingest.ts). A discriminated
 * union of readonly interfaces, one per stage of an ingest run (document ->
 * chunk -> attempt -> import -> embedding refresh). Every variant carries a
 * `kind` literal so callers can switch on it exhaustively. The mechanical
 * mirror of the Python twin's `taguru_langchain.events` — see that module
 * for the full design rationale; field names are snake_case here too,
 * matching the SDKs' wire-vocabulary convention.
 */

/**
 * Provider-reported details for one LLM call, when the model exposes them.
 *
 * `finish_reason` is read from `response_metadata` under whichever of
 * `done_reason`/`finish_reason`/`stop_reason` the provider used. Token
 * counts come from LangChain's normalized `usage_metadata`.
 */
export interface ProviderMetadata {
  finish_reason: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
}

export interface DocumentStarted {
  kind: "document_started";
  source: string;
  text_bytes: number;
}

export interface ChunkStarted {
  kind: "chunk_started";
  source: string;
  index: number;
  total: number;
}

/**
 * Which corrective loop this attempt belongs to: `"item"` is the per-chunk
 * Stage 1 loop (syntax/validity), `"cross_chunk"` is the single targeted
 * Stage 2 correction crossOutputIssues can trigger once every chunk's
 * output is known (issue #180's ADR 0001 §8 bucket 2, ported from the Rust
 * twin's cross_output_issues).
 */
export type AttemptStage = "item" | "cross_chunk";

export interface AttemptStarted {
  kind: "attempt_started";
  source: string;
  chunk_index: number;
  attempt: number;
  max_attempts: number;
  stage: AttemptStage;
}

export interface AttemptFailed {
  kind: "attempt_failed";
  source: string;
  chunk_index: number;
  attempt: number;
  max_attempts: number;
  parse_error: string;
  elapsed_seconds: number;
  provider_metadata: ProviderMetadata | null;
  /**
   * Whether `provider_metadata.finish_reason` says this answer was cut off
   * at the provider's output cap (see extract.ts's `indicatesLengthLimit`)
   * — the next attempt's corrective turn asks for a shorter answer instead
   * of repeating the same ask verbatim when this is true.
   */
  length_limited: boolean;
  /** See AttemptStarted's `stage`. */
  stage: AttemptStage;
  /**
   * The path-addressed issues (e.g. `"associations[1].weight: expected
   * finite non-zero number, got string \"strong\""`) that failed this
   * attempt, when it was syntactically valid JSON but not a valid
   * extraction (extract.ts's InvalidFault) or a Stage 2 cross-chunk alias
   * problem. `null` for every other failure kind — `parse_error` already
   * carries a human-readable diagnosis for all of them.
   */
  validation_issues: string[] | null;
}

/**
 * Raw per-chunk proposal counts, from before cross-chunk merge/dedup.
 *
 * These intentionally do not match IngestOutcome's final tallies
 * (associations, aliases, ...): merge() dedupes across chunks, so a
 * per-chunk "accepted" count doesn't exist until every chunk is in.
 */
export interface ChunkCompleted {
  kind: "chunk_completed";
  source: string;
  index: number;
  total: number;
  associations_proposed: number;
  aliases_proposed: number;
  questions_proposed: number;
  llm_calls: number;
  elapsed_seconds: number;
}

export interface ImportStarted {
  kind: "import_started";
  source: string;
}

export interface ImportCompleted {
  kind: "import_completed";
  source: string;
  elapsed_seconds: number;
}

export interface EmbeddingRefreshStarted {
  kind: "embedding_refresh_started";
  source: string;
}

/**
 * Terminal, non-error outcome of a refresh attempt.
 *
 * `configured: false` covers the 501 "no provider configured" case, which
 * is expected steady state for most deployments, not a failure.
 */
export interface EmbeddingRefreshCompleted {
  kind: "embedding_refresh_completed";
  source: string;
  configured: boolean;
  embedded: number;
  total: number;
}

export interface EmbeddingRefreshWarning {
  kind: "embedding_refresh_warning";
  source: string;
  message: string;
}

export type IngestEvent =
  | DocumentStarted
  | ChunkStarted
  | AttemptStarted
  | AttemptFailed
  | ChunkCompleted
  | ImportStarted
  | ImportCompleted
  | EmbeddingRefreshStarted
  | EmbeddingRefreshCompleted
  | EmbeddingRefreshWarning;

/**
 * Must be synchronous and non-blocking — ingestText calls it directly,
 * without a microtask hop or an await, so a slow callback stalls the
 * ingest. Exceptions raised here are caught and reported via
 * `console.warn`; they never interrupt the ingest they were reporting on.
 */
export type IngestEventCallback = (event: IngestEvent) => void;
