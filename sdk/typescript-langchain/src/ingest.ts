/**
 * TaguruIngester: LLM-driven document decomposition into a Taguru context.
 * The mechanical mirror of the Python `taguru_langchain.TaguruIngester` —
 * see that module for the full design rationale (import-based per-source
 * replace, no source-id fallback, live vocabulary seeding, dry_run).
 */

import type { DocumentInterface } from "@langchain/core/documents";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import type { Runnable } from "@langchain/core/runnables";
import {
  EmbeddingUnavailableError,
  NotFoundError,
  Taguru,
  type ImportOutcome,
  type SchemaDocument,
} from "taguru";

import {
  CHUNK_BYTES,
  MAX_PASSAGE_BYTES,
  MAX_QUESTIONS_PER_PARAGRAPH,
  MODEL_OUTPUT_JSON_SCHEMA,
  PROMPT_VERSION,
  VOCABULARY_CAP,
  chunk,
  combinedCrossOutputIssues,
  correctiveAssistantTurnContent,
  correctiveMessage,
  correctiveValidationMessage,
  effectiveItemRules,
  emptyAnswerDiagnosis,
  evaluateAnswer,
  indicatesLengthLimit,
  indicatesRefusal,
  interpretModelOutput,
  InvalidFault,
  isEmptyAnswer,
  labeledDocument,
  merge,
  renderBatch,
  reparseBatch,
  splitParagraphs,
  SyntaxFault,
  systemPrompt,
  userMessage,
  type ItemRules,
  type ModelOutput,
} from "./extract.js";
import type {
  IngestEvent,
  IngestEventCallback,
  ProviderMetadata,
} from "./events.js";
import {
  deriveModelIdentity,
  sha256Hex,
  DocumentCheckpoints,
  type CheckpointFingerprint,
  type CheckpointStore,
} from "./checkpoints.js";

/** Total attempts (1 initial + corrections) at the JSON object per chunk. */
const DEFAULT_MAX_ATTEMPTS = 2;

/**
 * Hard ceiling on max_attempts: a misconfigured value must not be able to
 * turn one stubborn chunk into an unbounded number of model calls. Kept
 * in sync with src/extract.rs's MAX_EXTRACT_ATTEMPTS.
 */
const MAX_ATTEMPTS_CEILING = 10;

/** What one document's ingest amounted to. */
export interface IngestOutcome {
  source: string;
  ok: boolean;
  ndjson: string | null;
  created: boolean;
  retracted: number;
  associations: number;
  aliases: number;
  passage_stored: boolean;
  questions_stored: number;
  /** Under the strict default this counts only merge()'s policy trims
   * (per-paragraph question-cap overflow, a volunteered question when
   * none was requested) — a business-rule-invalid item is corrected or
   * fails the source before merge() ever runs (issue #181). Under
   * `lossy: true` it is the old drop-and-proceed tally: every item
   * merge() silently discarded. */
  duplicates_dropped: number;
  invalid_dropped: number;
  llm_calls: number;
  chunks: number;
  /** How many corrective turns (Stage 1 syntax/validity retries plus any
   * Stage 2 cross-chunk correction) this ingest needed — 0 means
   * every chunk's first answer was accepted as-is. */
  correction_attempts: number;
  /** Labels of automatic, information-preserving JSON repairs applied
   * across every accepted answer (e.g. "trailing_comma", "bom",
   * "code_fence", "braces_slice" — see candidateJson). Empty when no
   * repair was ever needed. */
  lossless_repairs: string[];
  error: string | null;
  embeddings_refresh_warning: string | null;
  /**
   * True when a `should_stop` check was honored between chunks (or, for
   * `ingestDocuments`, between documents) — not a failure: `ok` stays
   * `false` and nothing was imported, but every chunk validated before the
   * stop is durably checkpointed for the next run to resume from (issue
   * #212, the TypeScript twin of #211/#210 for #179).
   */
  interrupted: boolean;
  /** How many chunks were satisfied from a checkpoint store instead of a
   * fresh model call. */
  chunks_reused: number;
}

const emptyOutcome = (source: string): IngestOutcome => ({
  source,
  ok: false,
  ndjson: null,
  created: false,
  retracted: 0,
  associations: 0,
  aliases: 0,
  passage_stored: false,
  questions_stored: 0,
  duplicates_dropped: 0,
  invalid_dropped: 0,
  llm_calls: 0,
  chunks: 0,
  correction_attempts: 0,
  lossless_repairs: [],
  error: null,
  embeddings_refresh_warning: null,
  interrupted: false,
  chunks_reused: 0,
});

/**
 * A caller-supplied cooperative-stop check for `ingestText`/
 * `ingestDocuments`: either a zero-argument predicate, or an `AbortSignal`
 * (the JS-idiomatic analogue of a `threading.Event` — the same cancellation
 * vocabulary `fetch`/`setTimeout` already use). Checked only between chunks
 * and between documents, never mid-model-call: an in-flight LLM call always
 * finishes. Deliberately not named `stop` — LangChain uses that word for
 * stop sequences.
 */
export type ShouldStop = (() => boolean) | AbortSignal;

/**
 * Stable (sorted-key) JSON serialization — object key order is otherwise
 * insertion order, which would make two structurally identical schema
 * documents hash differently depending on how the server happened to
 * serialize `types`/`relations` this time.
 */
function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) =>
      a < b ? -1 : a > b ? 1 : 0,
    );
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/**
 * A canonical digest of one fetched schema document, folded into the
 * checkpoint fingerprint (ADR 0009 §11.1, mirrors src/extract.rs's own
 * `schema_digest`) so swapping in a different schema document re-extracts
 * exactly like any other computation input changing. `""` when no schema
 * was fetched — the same "no schema" value a fresh schema-less run
 * resolves to.
 */
async function schemaDigest(schema: SchemaDocument | null): Promise<string> {
  if (schema === null) {
    return "";
  }
  return sha256Hex(stableStringify(schema));
}

function normalizeStop(shouldStop: ShouldStop | undefined): () => boolean {
  if (shouldStop === undefined) {
    return () => false;
  }
  if (typeof shouldStop === "function") {
    return shouldStop;
  }
  return () => shouldStop.aborted;
}

export interface TaguruIngesterFields {
  context: string;
  llm: BaseChatModel;
  client?: Taguru;
  base_url?: string;
  api_key?: string;
  create_context?: boolean;
  context_description?: string;
  source_key?: string;
  questions?: number;
  fact_budget?: number;
  max_attempts?: number;
  corrective_context_bytes?: number;
  include_passage?: boolean;
  chunk_bytes?: number;
  vocabulary_cap?: number;
  refresh_embeddings?: boolean;
  raise_on_error?: boolean;
  structured_output?: boolean;
  /**
   * Restore the pre-issue-#181 drop-and-proceed behavior: a
   * business-rule-invalid item (bad weight, dangling alias, out-of-range
   * question, ...) is silently dropped and the source still reports
   * success, exactly like merge() always did. Default `false` (ADR 0001
   * §8's never-silent-drop default): an invalid item instead earns one
   * targeted, path-addressed corrective turn, and the source fails
   * outright (no /import call) if it is still invalid afterward. See
   * `IngestOutcome.invalid_dropped`.
   */
  lossy?: boolean;
  /**
   * Live progress/diagnostics: fires typed events (document/chunk/attempt
   * started, attempt failed with parse error + provider metadata, chunk
   * completed, import started/completed, embedding refresh started/
   * completed/warning) from ingestText, so a caller can show progress and
   * see *why* a corrective attempt fired without copying the private
   * extraction helpers. Must be synchronous and non-blocking (see
   * IngestEventCallback). Default undefined: no callback, no behavior
   * difference.
   */
  on_event?: IngestEventCallback;
  /**
   * Enables durable per-chunk checkpoints (issue #212): each chunk's
   * validated output is persisted here as soon as it is accepted, so an
   * interrupted document (process killed, spot instance reclaimed) can
   * resume without recalling the model for chunks already validated.
   * Default undefined: no checkpointing, unchanged behavior.
   */
  checkpoint_store?: CheckpointStore;
  /**
   * Overrides the model identity baked into the checkpoint fingerprint.
   * Required when `checkpoint_store` is set and `llm` exposes none of
   * `model`/`modelName`/`modelId`/`deploymentName` (checked at
   * construction — see `deriveModelIdentity` in checkpoints.ts); optional
   * otherwise.
   */
  checkpoint_model_id?: string;
}

/**
 * The shape withStructuredOutput(schema, { includeRaw: true }) resolves to
 * for a plain (non-Zod) JSON Schema like MODEL_OUTPUT_JSON_SCHEMA: RunOutput
 * stays at its Record<string, any> default. The .d.ts types `parsed` as
 * always-present, but the implementation's own includeRaw:true fallback
 * (@langchain/core's assembleStructuredOutputPipeline) can still resolve it
 * to null on a failed extraction — attemptOnce() below checks for that.
 */
type StructuredOutputResult = { raw: BaseMessage; parsed: Record<string, unknown> | null };

// -- one attempt's §7-style classification (issue #181, mirrors extract.rs's ---
// -- classify_attempt / evaluate_answer and the Python twin's _Attempt) --------

type AttemptKind = "valid" | "length_limited" | "refusal" | "empty" | "syntax" | "invalid";

/**
 * One model call's outcome, classified from provider metadata BEFORE any
 * parse-level interpretation — a `length`-terminated answer is
 * length-limited even when its prefix happens to parse, since a valid
 * prefix of a cut-off extraction is exactly the "deleted-subset called
 * complete" ADR 0001 forbids.
 */
interface AttemptResult {
  kind: AttemptKind;
  content: string;
  output: ModelOutput | null;
  repairs: string[];
  error: string | null;
  issues: string[] | null;
  metadata: ProviderMetadata | null;
}

function classifyText(
  content: string,
  metadata: ProviderMetadata | null,
  rules: ItemRules | null,
): AttemptResult {
  const finishReason = metadata?.finish_reason ?? undefined;
  if (indicatesLengthLimit(finishReason)) {
    return {
      kind: "length_limited",
      content,
      output: null,
      repairs: [],
      error: "the answer hit the provider's output limit",
      issues: null,
      metadata,
    };
  }
  if (indicatesRefusal(finishReason)) {
    return {
      kind: "refusal",
      content,
      output: null,
      repairs: [],
      error: finishReason!,
      issues: null,
      metadata,
    };
  }
  if (isEmptyAnswer(content)) {
    return {
      kind: "empty",
      content,
      output: null,
      repairs: [],
      error: emptyAnswerDiagnosis(),
      issues: null,
      metadata,
    };
  }
  try {
    const { output, repairs } = evaluateAnswer(content, rules);
    return { kind: "valid", content, output, repairs, error: null, issues: null, metadata };
  } catch (error) {
    if (error instanceof InvalidFault) {
      return {
        kind: "invalid",
        content,
        output: null,
        repairs: [],
        error: null,
        issues: error.issues,
        metadata,
      };
    }
    if (error instanceof SyntaxFault) {
      return {
        kind: "syntax",
        content,
        output: null,
        repairs: [],
        error: error.message,
        issues: null,
        metadata,
      };
    }
    throw error;
  }
}

/**
 * The structured-output twin of classifyText: the provider has already
 * parsed (or failed to parse) its own tool-call args against
 * MODEL_OUTPUT_JSON_SCHEMA, so there is no raw JSON text here to
 * lossless-repair, and `content` is conventionally empty for a
 * tool-calling response — isEmptyAnswer would misfire on it, so emptiness
 * is judged from `parsed` instead.
 */
function classifyStructured(
  content: string,
  metadata: ProviderMetadata | null,
  parsed: Record<string, unknown> | null | undefined,
  rules: ItemRules | null,
): AttemptResult {
  const finishReason = metadata?.finish_reason ?? undefined;
  if (indicatesLengthLimit(finishReason)) {
    return {
      kind: "length_limited",
      content,
      output: null,
      repairs: [],
      error: "the answer hit the provider's output limit",
      issues: null,
      metadata,
    };
  }
  if (indicatesRefusal(finishReason)) {
    return {
      kind: "refusal",
      content,
      output: null,
      repairs: [],
      error: finishReason!,
      issues: null,
      metadata,
    };
  }
  if (parsed === undefined || parsed === null) {
    return {
      kind: "syntax",
      content,
      output: null,
      repairs: [],
      error:
        "the model's structured-output call produced no parsed result (LangChain.js surfaces " +
        "no further detail here — see assembleStructuredOutputPipeline's includeRaw:true fallback)",
      issues: null,
      metadata,
    };
  }
  const { output, issues } = interpretModelOutput(parsed, effectiveItemRules(rules));
  if (rules !== null && issues.length > 0) {
    return { kind: "invalid", content, output: null, repairs: [], error: null, issues, metadata };
  }
  return { kind: "valid", content, output, repairs: [], error: null, issues: null, metadata };
}

/** The raw, unwrapped human-readable reason one attempt failed — used for
 * the corrective ask and (for "invalid"/"empty") the final failure message
 * verbatim. */
function diagnosisFor(result: AttemptResult): string {
  if (result.kind === "invalid") {
    const issues = result.issues!;
    return `the answer left ${issues.length} invalid item(s) uncorrected: ${issues.join("; ")}`;
  }
  return result.error ?? "";
}

/**
 * The message thrown when a chunk's attempts are exhausted without a valid
 * answer. "invalid"/"empty" stay unwrapped (extract.rs's
 * AttemptOutcome::Invalid/Empty never get the generic wrapper either);
 * "syntax"/"length_limited" get the same "the model would not produce the
 * JSON object" wrapper today's (pre-#181) behavior always used.
 */
function finalMessageFor(result: AttemptResult): string {
  const diagnosis = diagnosisFor(result);
  if (result.kind === "invalid" || result.kind === "empty") {
    return diagnosis;
  }
  return `the model would not produce the JSON object: ${diagnosis}`;
}

/**
 * The next attempt's user-facing ask, addressed to whichever kind of
 * failure `result` was. "empty" reuses correctiveMessage's ordinary
 * (non-length-limited) text with the empty diagnosis as its parseError.
 */
function correctiveAskFor(result: AttemptResult, factBudget: number): string {
  if (result.kind === "invalid") {
    return correctiveValidationMessage(result.issues!);
  }
  const lengthLimited = result.kind === "length_limited";
  return correctiveMessage(result.error ?? "", lengthLimited, factBudget);
}

/**
 * The Stage 2 (cross-chunk correction) terminal message for one
 * offending chunk's non-valid reply — mirrors extract.rs's
 * correct_cross_output_issues per-kind texts verbatim.
 */
function crossChunkFailureMessage(label: string, result: AttemptResult): string {
  if (result.kind === "length_limited") {
    return (
      `${label}: the cross-chunk correction was cut off at the output limit — ` +
      "failing the source rather than importing a truncated correction"
    );
  }
  if (result.kind === "refusal") {
    return `${label}: the provider refused the cross-chunk correction (finish_reason ${result.error})`;
  }
  if (result.kind === "empty") {
    return `${label}: ${result.error}`;
  }
  if (result.kind === "invalid") {
    const issues = result.issues!;
    return (
      `${label}: the cross-chunk correction still left ${issues.length} invalid item(s) ` +
      `uncorrected: ${issues.join("; ")}`
    );
  }
  return `${label}: the cross-chunk correction was not the JSON object asked for (${result.error})`;
}

/**
 * One chunk's accepted output, plus everything Stage 2's single targeted
 * corrective turn needs to rebuild THAT chunk's own conversation (never
 * the whole document's) if crossOutputIssues flags it.
 */
interface ChunkRecord {
  output: ModelOutput;
  chunkIndex: number;
  user: string;
  answer: string;
}

/**
 * Decompose LangChain Documents into one Taguru context via a chat model.
 * See the Python twin's docstring for parameter semantics — identical here,
 * modulo TS being async-only (no separate `a`-prefixed variants).
 */
export class TaguruIngester {
  readonly context: string;
  readonly source_key: string;
  readonly questions: number;
  readonly fact_budget: number;
  readonly max_attempts: number;
  readonly corrective_context_bytes: number | undefined;
  readonly include_passage: boolean;
  readonly chunk_bytes: number;
  readonly vocabulary_cap: number;
  readonly refresh_embeddings: boolean;
  readonly raise_on_error: boolean;
  readonly structured_output: boolean;
  readonly lossy: boolean;
  readonly checkpoint_store: CheckpointStore | undefined;
  readonly checkpoint_model_id: string | undefined;

  private readonly llm: BaseChatModel;
  private readonly client: Taguru;
  private readonly create_context: boolean;
  private readonly context_description: string | undefined;
  private readonly structuredLlm: Runnable<BaseLanguageModelInput, StructuredOutputResult> | null;
  private readonly on_event: IngestEventCallback | undefined;

  constructor(fields: TaguruIngesterFields) {
    if (fields.create_context && fields.context_description === undefined) {
      throw new Error("create_context: true requires context_description");
    }
    const questions = fields.questions ?? 0;
    if (questions < 0 || questions > MAX_QUESTIONS_PER_PARAGRAPH) {
      throw new Error(`questions must be between 0 and ${MAX_QUESTIONS_PER_PARAGRAPH}`);
    }
    if (fields.fact_budget !== undefined && fields.fact_budget < 1) {
      throw new Error(
        "fact_budget must be a positive integer, or undefined to leave it unbounded",
      );
    }
    const maxAttempts = fields.max_attempts ?? DEFAULT_MAX_ATTEMPTS;
    if (maxAttempts < 1 || maxAttempts > MAX_ATTEMPTS_CEILING) {
      throw new Error(`max_attempts must be between 1 and ${MAX_ATTEMPTS_CEILING}`);
    }
    if (fields.corrective_context_bytes !== undefined && fields.corrective_context_bytes < 0) {
      throw new Error(
        "corrective_context_bytes must be a non-negative integer, or undefined to replay " +
          "the prior bad answer in full",
      );
    }
    this.context = fields.context;
    this.llm = fields.llm;
    this.client =
      fields.client ?? new Taguru({ base_url: fields.base_url, api_key: fields.api_key });
    this.create_context = fields.create_context ?? false;
    this.context_description = fields.context_description;
    this.source_key = fields.source_key ?? "source";
    this.questions = questions;
    this.fact_budget = fields.fact_budget ?? 0;
    this.max_attempts = maxAttempts;
    // Passed through as-is: undefined (replay the prior bad answer in
    // full) and 0 (omit it behind a placeholder) are distinct policies —
    // a `?? 0` here would silently collapse the default into omission.
    this.corrective_context_bytes = fields.corrective_context_bytes;
    this.include_passage = fields.include_passage ?? true;
    this.chunk_bytes = fields.chunk_bytes ?? CHUNK_BYTES;
    this.vocabulary_cap = fields.vocabulary_cap ?? VOCABULARY_CAP;
    this.refresh_embeddings = fields.refresh_embeddings ?? true;
    this.raise_on_error = fields.raise_on_error ?? false;
    this.structured_output = fields.structured_output ?? false;
    this.lossy = fields.lossy ?? false;
    this.on_event = fields.on_event;
    this.checkpoint_store = fields.checkpoint_store;
    this.checkpoint_model_id = fields.checkpoint_model_id;
    if (
      this.checkpoint_store !== undefined &&
      this.checkpoint_model_id === undefined &&
      deriveModelIdentity(fields.llm) === null
    ) {
      throw new Error(
        "checkpoint_store requires checkpoint_model_id: this chat model exposes none of " +
          "model/modelName/modelId/deploymentName, so the checkpoint fingerprint has no " +
          "stable model identity to key on",
      );
    }
    // Opt-in only (see the Python twin's docstring for the full rationale):
    // built once here so a chat model that cannot bind tools fails fast, at
    // construction, rather than surfacing later as a per-attempt failure.
    // name: "ModelOutput" keeps the bound tool's name aligned with the Rust
    // and Python producers instead of withStructuredOutput()'s "extract"
    // default (see MODEL_OUTPUT_JSON_SCHEMA's doc comment in extract.ts).
    this.structuredLlm = this.structured_output
      ? fields.llm.withStructuredOutput(MODEL_OUTPUT_JSON_SCHEMA, {
          includeRaw: true,
          name: "ModelOutput",
        })
      : null;
  }

  /**
   * Reports one progress/diagnostic event to `on_event`, if set. A raising
   * callback is caught and reported via `console.warn` rather than
   * breaking the ingest it was reporting on — mirrors the Python twin's
   * `_emit`.
   */
  private emit(event: IngestEvent): void {
    if (this.on_event === undefined) {
      return;
    }
    try {
      this.on_event(event);
    } catch (error) {
      console.warn(
        `TaguruIngester on_event callback raised ${String(error)}; ingest continues`,
      );
    }
  }

  /**
   * Runs one model call for one attempt and classifies it (issue #181's
   * §7-style state machine: length -> refusal -> empty ->
   * syntax/invalid/valid). Goes through the withStructuredOutput()
   * pipeline built in the constructor when structured_output is on, else
   * today's plain invoke() + the free-text validation walk.
   */
  private async attemptOnce(
    messages: BaseMessage[],
    rules: ItemRules | null,
  ): Promise<AttemptResult> {
    if (this.structuredLlm !== null) {
      const result = await this.structuredLlm.invoke(messages);
      const content = contentText(result.raw);
      const metadata = providerMetadata(result.raw);
      return classifyStructured(content, metadata, result.parsed, rules);
    }
    const response = await this.llm.invoke(messages);
    const content = contentText(response);
    const metadata = providerMetadata(response);
    return classifyText(content, metadata, rules);
  }

  /**
   * `null` under `lossy: true` — evaluateAnswer/interpretModelOutput then
   * parse leniently and discard whatever they'd have flagged, reproducing
   * pre-issue-#181 behavior byte for byte (merge() alone decides what
   * survives).
   */
  private itemRules(paragraphCount: number): ItemRules | null {
    if (this.lossy) {
      return null;
    }
    return { paragraphCount, questionsRequested: this.questions > 0 };
  }

  /**
   * Issue #181 Stage 2: one targeted corrective turn per output
   * crossOutputIssues flags, rebuilding THAT chunk's own conversation base
   * (never the whole document's) and replaying its own final answer as
   * the prior bad turn. Bounded to exactly one extra call per offending
   * chunk regardless of max_attempts: a still-invalid, still-cross-
   * conflicting, length-limited, refused, or empty reply fails the source
   * outright — Stage 2 never loops a second round. Mirrors extract.rs's
   * correct_cross_output_issues.
   */
  private async correctCrossChunkIssues(
    system: string,
    records: ChunkRecord[],
    rules: ItemRules | null,
    chunkTotal: number,
    outcome: IngestOutcome,
    schema: SchemaDocument | null,
  ): Promise<void> {
    for (const [recordIndex, issues] of combinedCrossOutputIssues(
      records.map((r) => r.output),
      schema,
    )) {
      const chunkRecord = records[recordIndex]!;
      const label = `chunk ${chunkRecord.chunkIndex + 1}/${chunkTotal}`;
      const messages: BaseMessage[] = [
        new SystemMessage(system),
        new HumanMessage(chunkRecord.user),
        new AIMessage(
          correctiveAssistantTurnContent(chunkRecord.answer, this.corrective_context_bytes),
        ),
        new HumanMessage(correctiveValidationMessage(issues)),
      ];
      this.emit({
        kind: "attempt_started",
        source: outcome.source,
        chunk_index: chunkRecord.chunkIndex,
        attempt: 1,
        max_attempts: 1,
        stage: "cross_chunk",
      });
      const attemptStartedAt = performance.now();
      const result = await this.attemptOnce(messages, rules);
      outcome.llm_calls += 1;
      outcome.correction_attempts += 1;
      if (result.kind === "valid") {
        records[recordIndex] = {
          output: result.output!,
          chunkIndex: chunkRecord.chunkIndex,
          user: chunkRecord.user,
          answer: result.content,
        };
        outcome.lossless_repairs.push(...result.repairs);
        continue;
      }
      const message = crossChunkFailureMessage(label, result);
      this.emit({
        kind: "attempt_failed",
        source: outcome.source,
        chunk_index: chunkRecord.chunkIndex,
        attempt: 1,
        max_attempts: 1,
        parse_error: message,
        elapsed_seconds: (performance.now() - attemptStartedAt) / 1000,
        provider_metadata: result.metadata,
        length_limited: result.kind === "length_limited",
        stage: "cross_chunk",
        validation_issues: result.issues,
      });
      throw new Error(message);
    }

    // Re-check rather than trust the single corrective turn blindly: a
    // correction can rename an association another chunk's alias depended
    // on, introducing a FRESH cross-chunk issue. This is the bounded
    // re-check, not a second round — any issue here fails the source.
    const recheck = combinedCrossOutputIssues(
      records.map((r) => r.output),
      schema,
    );
    if (recheck.length > 0) {
      const [recordIndex, issues] = recheck[0]!;
      const chunkIndex = records[recordIndex]!.chunkIndex;
      throw new Error(
        `chunk ${chunkIndex + 1}/${chunkTotal}: still has ${issues.length} cross-chunk ` +
          `issue(s) after correction: ${issues.join("; ")}`,
      );
    }
  }

  /**
   * Ingest one text under one source id. Throws on failure (unlike
   * `ingestDocuments`, there is no "continue with the rest" here).
   */
  async ingestText(
    text: string,
    options: { source: string; dry_run?: boolean; should_stop?: ShouldStop },
  ): Promise<IngestOutcome> {
    const stop = normalizeStop(options.should_stop);
    const outcome = emptyOutcome(options.source);
    if (this.include_passage && Buffer.byteLength(text, "utf-8") > MAX_PASSAGE_BYTES) {
      throw new Error(`document exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`);
    }
    this.emit({
      kind: "document_started",
      source: options.source,
      text_bytes: Buffer.byteLength(text, "utf-8"),
    });

    const vocabulary = await this.fetchVocabulary();
    const schema = await this.fetchSchema();
    const system = systemPrompt(vocabulary, this.questions, this.fact_budget, schema);
    const paragraphCount = splitParagraphs(text).length;
    const rules = this.itemRules(paragraphCount);
    const chunks = chunk(labeledDocument(text, this.chunk_bytes), this.chunk_bytes);
    outcome.chunks = chunks.length;
    const checkpoints = await this.loadCheckpoints(
      options.source,
      text,
      await schemaDigest(schema),
    );

    const records: ChunkRecord[] = [];
    for (let index = 0; index < chunks.length; index += 1) {
      if (stop()) {
        outcome.interrupted = true;
        return outcome;
      }
      this.emit({
        kind: "chunk_started",
        source: options.source,
        index,
        total: chunks.length,
      });
      const chunkStartedAt = performance.now();
      const cached = await this.checkpointedUnit(checkpoints, chunks[index]!);
      if (cached !== null) {
        records.push(cached);
        outcome.chunks_reused += 1;
        this.emit({
          kind: "chunk_completed",
          source: options.source,
          index,
          total: chunks.length,
          associations_proposed: cached.output.associations.length,
          aliases_proposed: cached.output.aliases.length,
          questions_proposed: cached.output.questions.length,
          llm_calls: 0,
          elapsed_seconds: (performance.now() - chunkStartedAt) / 1000,
          reused: true,
        });
        continue;
      }
      let chunkLlmCalls = 0;
      const user = userMessage(options.source, index, chunks.length, chunks[index]!);
      const base: BaseMessage[] = [new SystemMessage(system), new HumanMessage(user)];
      let chunkRecord: ChunkRecord | null = null;
      // Each retry rebuilds the conversation from the system/user base and
      // appends only the most recent bad turn — never the whole history —
      // so corrective_context_bytes bounds every retry alike, not just the
      // first. At the all-defaults policy this reproduces the previous
      // fixed implementation's request bodies exactly.
      let pendingAsk: string | null = null;
      let priorBadAnswer: string | null = null;
      let lastDiagnosis = "";
      let emptyCorrected = false;

      for (let attempt = 0; attempt < this.max_attempts && chunkRecord === null; attempt += 1) {
        const messages: BaseMessage[] =
          priorBadAnswer === null
            ? base
            : [
                ...base,
                new AIMessage(
                  correctiveAssistantTurnContent(priorBadAnswer, this.corrective_context_bytes),
                ),
                new HumanMessage(pendingAsk ?? ""),
              ];
        this.emit({
          kind: "attempt_started",
          source: options.source,
          chunk_index: index,
          attempt: attempt + 1,
          max_attempts: this.max_attempts,
          stage: "item",
        });
        const attemptStartedAt = performance.now();
        const result = await this.attemptOnce(messages, rules);
        outcome.llm_calls += 1;
        chunkLlmCalls += 1;

        if (result.kind === "valid") {
          chunkRecord = { output: result.output!, chunkIndex: index, user, answer: result.content };
          outcome.lossless_repairs.push(...result.repairs);
          break;
        }

        if (result.kind === "refusal") {
          const message =
            `the provider refused this content (finish_reason ${result.error}) — a policy ` +
            "refusal is terminal; no corrective turn can change it";
          this.emit({
            kind: "attempt_failed",
            source: options.source,
            chunk_index: index,
            attempt: attempt + 1,
            max_attempts: this.max_attempts,
            parse_error: message,
            elapsed_seconds: (performance.now() - attemptStartedAt) / 1000,
            provider_metadata: result.metadata,
            length_limited: false,
            stage: "item",
            validation_issues: null,
          });
          throw new Error(message);
        }

        if (result.kind === "empty" && emptyCorrected) {
          const message = diagnosisFor(result);
          this.emit({
            kind: "attempt_failed",
            source: options.source,
            chunk_index: index,
            attempt: attempt + 1,
            max_attempts: this.max_attempts,
            parse_error: message,
            elapsed_seconds: (performance.now() - attemptStartedAt) / 1000,
            provider_metadata: result.metadata,
            length_limited: false,
            stage: "item",
            validation_issues: null,
          });
          throw new Error(message);
        }
        if (result.kind === "empty") {
          emptyCorrected = true;
        }

        lastDiagnosis = finalMessageFor(result);
        this.emit({
          kind: "attempt_failed",
          source: options.source,
          chunk_index: index,
          attempt: attempt + 1,
          max_attempts: this.max_attempts,
          parse_error: diagnosisFor(result),
          elapsed_seconds: (performance.now() - attemptStartedAt) / 1000,
          provider_metadata: result.metadata,
          length_limited: result.kind === "length_limited",
          stage: "item",
          validation_issues: result.issues,
        });
        outcome.correction_attempts += 1;
        pendingAsk = correctiveAskFor(result, this.fact_budget);
        priorBadAnswer = result.content;
      }

      if (chunkRecord === null) {
        throw new Error(lastDiagnosis);
      }
      records.push(chunkRecord);
      await this.recordCheckpoint(checkpoints, options.source, chunks[index]!, chunkRecord);
      this.emit({
        kind: "chunk_completed",
        source: options.source,
        index,
        total: chunks.length,
        associations_proposed: chunkRecord.output.associations.length,
        aliases_proposed: chunkRecord.output.aliases.length,
        questions_proposed: chunkRecord.output.questions.length,
        llm_calls: chunkLlmCalls,
        elapsed_seconds: (performance.now() - chunkStartedAt) / 1000,
        reused: false,
      });
    }

    if (!this.lossy) {
      await this.correctCrossChunkIssues(system, records, rules, chunks.length, outcome, schema);
    }

    const extraction = merge(
      records.map((r) => r.output),
      this.questions,
      paragraphCount,
    );
    outcome.duplicates_dropped = extraction.duplicates;
    outcome.invalid_dropped = extraction.dropped;
    const description = this.create_context ? (this.context_description ?? null) : null;
    const ndjson = renderBatch(
      this.context,
      options.source,
      description,
      extraction,
      this.include_passage ? text : null,
    );
    reparseBatch(ndjson);
    outcome.ndjson = ndjson;

    if (options.dry_run) {
      outcome.ok = true;
      return outcome;
    }

    this.emit({ kind: "import_started", source: options.source });
    const importStartedAt = performance.now();
    const applied = await this.client.importBatches(ndjson);
    record(outcome, applied.batches[0]!);
    outcome.ok = true;
    this.emit({
      kind: "import_completed",
      source: options.source,
      elapsed_seconds: (performance.now() - importStartedAt) / 1000,
    });
    // Deletion happens here — after the import lands, before the
    // best-effort embeddings refresh below — so a refresh warning can
    // never resurrect a checkpoint this document no longer needs.
    await this.deleteCheckpoints(options.source);

    if (this.refresh_embeddings) {
      this.emit({ kind: "embedding_refresh_started", source: options.source });
      try {
        const refreshResult = await this.client.context(this.context).refreshEmbeddings();
        this.emit({
          kind: "embedding_refresh_completed",
          source: options.source,
          configured: true,
          embedded: refreshResult.embedded,
          total: refreshResult.total,
        });
      } catch (error) {
        if (error instanceof EmbeddingUnavailableError && error.reason === "provider_error") {
          outcome.embeddings_refresh_warning = error.message;
          this.emit({
            kind: "embedding_refresh_warning",
            source: options.source,
            message: error.message,
          });
        } else if (error instanceof EmbeddingUnavailableError) {
          this.emit({
            kind: "embedding_refresh_completed",
            source: options.source,
            configured: false,
            embedded: 0,
            total: 0,
          });
        } else {
          throw error;
        }
      }
    }
    return outcome;
  }

  /**
   * Ingest each document independently; one failure never stops the rest
   * (set `raise_on_error: true` to fail fast).
   */
  async ingestDocuments(
    documents: DocumentInterface[],
    options: { dry_run?: boolean; should_stop?: ShouldStop } = {},
  ): Promise<IngestOutcome[]> {
    const stop = normalizeStop(options.should_stop);
    const outcomes: IngestOutcome[] = [];
    for (const document of documents) {
      if (stop()) {
        break;
      }
      let source: string;
      try {
        source = this.sourceOf(document);
      } catch (error) {
        if (this.raise_on_error) {
          throw error;
        }
        const failed = emptyOutcome("");
        failed.error = (error as Error).message;
        outcomes.push(failed);
        continue;
      }
      try {
        const outcome = await this.ingestText(document.pageContent, {
          source,
          dry_run: options.dry_run,
          should_stop: options.should_stop,
        });
        outcomes.push(outcome);
        if (outcome.interrupted) {
          break;
        }
      } catch (error) {
        if (this.raise_on_error) {
          throw error;
        }
        const failed = emptyOutcome(source);
        failed.error = error instanceof Error ? error.message : String(error);
        outcomes.push(failed);
      }
    }
    return outcomes;
  }

  private sourceOf(document: DocumentInterface): string {
    const source = document.metadata[this.source_key];
    if (typeof source !== "string" || source === "") {
      throw new Error(
        `document metadata lacks a '${this.source_key}' key — the source id is the ` +
          "retract-then-apply idempotency unit and cannot be invented",
      );
    }
    return source;
  }

  /**
   * The context's live relation vocabulary — an advantage the offline
   * extractor structurally lacks. Best-effort: an absent context is fine.
   */
  private async fetchVocabulary(): Promise<string[]> {
    try {
      const page = await this.client.context(this.context).listLabels({
        limit: this.vocabulary_cap,
      });
      return page.labels;
    } catch (error) {
      if (error instanceof NotFoundError) {
        return [];
      }
      throw error;
    }
  }

  /**
   * The context's schema document (ADR 0009 §11.4), same best-effort
   * posture as `fetchVocabulary`: a schema-unaware server or a schema-free
   * context is fine, and this ingester works unchanged either way.
   */
  private async fetchSchema(): Promise<SchemaDocument | null> {
    try {
      return await this.client.context(this.context).getSchema();
    } catch (error) {
      if (error instanceof NotFoundError) {
        return null;
      }
      throw error;
    }
  }

  // -- checkpoint/resume (issue #212, the TypeScript twin of #211/#210) --

  /**
   * The compatibility inputs that gate reuse of a cached chunk: content
   * hash of the WHOLE raw document (not per-chunk), model identity, and
   * every output-shaping setting. See CheckpointFingerprint's doc comment
   * for what is deliberately excluded and why.
   */
  private async checkpointFingerprint(
    text: string,
    digest: string,
  ): Promise<CheckpointFingerprint> {
    return {
      sha256: await sha256Hex(text),
      model: this.checkpoint_model_id ?? deriveModelIdentity(this.llm)!,
      prompt_version: PROMPT_VERSION,
      context: this.context,
      questions_n: this.questions,
      no_passage: !this.include_passage,
      description: this.create_context ? (this.context_description ?? "") : "",
      fact_budget: this.fact_budget,
      structured_output: this.structured_output ? "langchain" : "",
      lossy: this.lossy,
      schema_digest: digest,
    };
  }

  /**
   * `null` when no store is configured. A store `load` failure is caught
   * and reported via `console.warn` — the run proceeds as though nothing
   * were cached, never as a hard failure.
   */
  private async loadCheckpoints(
    source: string,
    text: string,
    digest: string,
  ): Promise<DocumentCheckpoints | null> {
    if (this.checkpoint_store === undefined) {
      return null;
    }
    const fingerprint = await this.checkpointFingerprint(text, digest);
    let data: Uint8Array | null;
    try {
      data = await this.checkpoint_store.load(source);
    } catch (error) {
      console.warn(
        `TaguruIngester checkpoint store failed to load ${JSON.stringify(source)}: ` +
          `${String(error)} — starting this document without any cached chunks`,
      );
      data = null;
    }
    return DocumentCheckpoints.fromBytes(data, fingerprint);
  }

  /**
   * Looks up `piece` (the labeled chunk text exactly as fed to the loop —
   * never its index) by content hash and, on a hit, reconstructs the
   * `ChunkRecord` from the STORED user/answer/output rather than
   * recomputing anything.
   */
  private async checkpointedUnit(
    checkpoints: DocumentCheckpoints | null,
    piece: string,
  ): Promise<ChunkRecord | null> {
    if (checkpoints === null) {
      return null;
    }
    const unit = checkpoints.lookup(await sha256Hex(piece));
    if (unit === null) {
      return null;
    }
    return { output: unit.output, chunkIndex: unit.chunk_index, user: unit.user, answer: unit.answer };
  }

  /**
   * Records one validated chunk and rewrites the whole checkpoint file —
   * small enough that a read-modify-write-whole-file is simpler, and just
   * as durable, as an append-only log. A store `save` failure is caught
   * and reported via `console.warn`: the chunk still counts for this run,
   * but a resume may repeat it.
   */
  private async recordCheckpoint(
    checkpoints: DocumentCheckpoints | null,
    source: string,
    piece: string,
    chunkRecord: ChunkRecord,
  ): Promise<void> {
    if (checkpoints === null || this.checkpoint_store === undefined) {
      return;
    }
    checkpoints.record(await sha256Hex(piece), {
      chunk_index: chunkRecord.chunkIndex,
      output: chunkRecord.output,
      user: chunkRecord.user,
      answer: chunkRecord.answer,
    });
    try {
      await this.checkpoint_store.save(source, checkpoints.toBytes());
    } catch (error) {
      console.warn(
        `TaguruIngester failed to save a checkpoint for ${JSON.stringify(source)}: ` +
          `${String(error)} — the completed chunk still counts this run; a resume may repeat it`,
      );
    }
  }

  /**
   * Best-effort cleanup once a document's batch has actually landed —
   * failures are silently ignored (the save/delete asymmetry, inherited
   * from the Rust twin: nothing correctness-critical depends on prompt
   * cleanup, unlike a save that a resume would otherwise repeat).
   */
  private async deleteCheckpoints(source: string): Promise<void> {
    if (this.checkpoint_store === undefined) {
      return;
    }
    try {
      await this.checkpoint_store.delete(source);
    } catch {
      // silently ignored
    }
  }
}

function record(outcome: IngestOutcome, applied: ImportOutcome): void {
  outcome.created = applied.created;
  outcome.retracted = applied.retracted;
  outcome.associations = applied.associations;
  outcome.aliases = applied.aliases;
  outcome.passage_stored = applied.passage_stored;
  outcome.questions_stored = applied.questions_stored;
}

/**
 * The provider's finish reason, read from AIMessage.response_metadata
 * under whichever key the integration used — done_reason (Ollama),
 * finish_reason (OpenAI and most others), stop_reason (Anthropic) — the
 * same priority order as the Python twin's _provider_metadata. Narrowed
 * defensively from unknown (like contentText below) rather than typed
 * against BaseMessage's generic metadata field.
 */
const FINISH_REASON_KEYS = ["done_reason", "finish_reason", "stop_reason"] as const;

function readTokenCount(usage: Record<string, unknown> | null, key: string): number | null {
  const value = usage?.[key];
  return typeof value === "number" ? value : null;
}

/**
 * Provider-reported details for one model call, read from
 * `response_metadata`/`usage_metadata` on an AIMessage — the shared
 * metadata reader `AttemptFailed` events and diagnostics both draw from.
 * Mirrors the Python twin's `_provider_metadata`: returns `null` (not a
 * ProviderMetadata of nulls) when there is neither a finish reason nor
 * usage metadata at all.
 */
function providerMetadata(message: unknown): ProviderMetadata | null {
  const responseMetadata = (message as { response_metadata?: unknown }).response_metadata;
  let finishReason: string | null = null;
  if (typeof responseMetadata === "object" && responseMetadata !== null) {
    const record = responseMetadata as Record<string, unknown>;
    for (const key of FINISH_REASON_KEYS) {
      const value = record[key];
      if (value !== undefined && value !== null) {
        finishReason = String(value);
        break;
      }
    }
  }
  const usageMetadata = (message as { usage_metadata?: unknown }).usage_metadata;
  const usage =
    typeof usageMetadata === "object" && usageMetadata !== null
      ? (usageMetadata as Record<string, unknown>)
      : null;
  if (finishReason === null && usage === null) {
    return null;
  }
  return {
    finish_reason: finishReason,
    input_tokens: readTokenCount(usage, "input_tokens"),
    output_tokens: readTokenCount(usage, "output_tokens"),
    total_tokens: readTokenCount(usage, "total_tokens"),
  };
}

function contentText(message: unknown): string {
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) =>
        typeof part === "object" && part !== null && "text" in part
          ? String((part as { text: unknown }).text)
          : String(part),
      )
      .join("");
  }
  return String(content ?? "");
}
