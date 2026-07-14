/**
 * TypeScript/JavaScript client SDK for the Taguru long-term semantic memory
 * server.
 *
 * Quick start:
 * ```ts
 * import { Taguru } from "taguru";
 *
 * const client = new Taguru(); // TAGURU_URL / TAGURU_API_TOKEN, else localhost:8248
 * const ctx = client.context("sake");
 * const hits = await ctx.searchPassages("酒蔵の創業年", { limit: 5 });
 * ```
 *
 * The behavioral contract is the server's own protocol document:
 * `await client.protocol()` (GET /protocol).
 */

export { Taguru, Contexts, Groups, Context, type TaguruOptions } from "./client.js";
export {
  TaguruError,
  AuthenticationError,
  PermissionDeniedError,
  NotFoundError,
  ConflictError,
  ValidationError,
  PayloadTooLargeError,
  RequestTimeoutError,
  RateLimitError,
  ServerError,
  ServiceUnavailableError,
  StorageFullError,
  EmbeddingUnavailableError,
  TransportError,
  UnexpectedStatusError,
} from "./errors.js";
export {
  citationKey,
  type Activation,
  type ActivationPage,
  type AliasEntry,
  type AliasPage,
  type AssocOp,
  type Association,
  type Attribution,
  type BatchApplyResult,
  type Bm25Explain,
  type Citation,
  type CompactOutcome,
  type ConceptDescription,
  type ContextMeta,
  type ContextPage,
  type ContextStats,
  type ContextUsage,
  type CrossAssociation,
  type CrossMatchCursor,
  type CrossMatchPage,
  type CrossPassageHit,
  type DirectoryEntry,
  type ExploreCursor,
  type ExplorePage,
  type GroupEntry,
  type GroupPage,
  type ImportOutcome,
  type LabelPage,
  type LabelUsage,
  type LaneEvidence,
  type LexicalExplain,
  type MatchCursor,
  type MatchPage,
  type NearestGloss,
  type NearestResolution,
  type NearestSpellings,
  type OneOrMany,
  type PassageHit,
  type PassageLanes,
  type PassageLookup,
  type QuestionSpec,
  type RankingExplain,
  type Recollection,
  type RefreshBreakdown,
  type RefreshOutcome,
  type ResolveExplanation,
  type ResolveRanking,
  type RetractAssociationOutcome,
  type RetractOutcome,
  type RetrievalResult,
  type SearchExplanation,
  type SectionSpec,
  type SemanticExplain,
  type SourcePage,
  type StoredPassages,
  type TermContribution,
  type TieredResolution,
  type TwinPair,
  type VectorExplain,
  type VocabularyAudit,
} from "./models.js";
