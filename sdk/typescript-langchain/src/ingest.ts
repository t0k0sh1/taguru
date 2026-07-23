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
} from "taguru";

import {
  CHUNK_BYTES,
  MAX_PASSAGE_BYTES,
  MAX_QUESTIONS_PER_PARAGRAPH,
  MODEL_OUTPUT_JSON_SCHEMA,
  VOCABULARY_CAP,
  chunk,
  coerceOutput,
  correctiveAssistantTurnContent,
  correctiveMessage,
  indicatesLengthLimit,
  labeledDocument,
  merge,
  parseModelOutput,
  renderBatch,
  reparseBatch,
  splitParagraphs,
  systemPrompt,
  userMessage,
  type ModelOutput,
} from "./extract.js";

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
  duplicates_dropped: number;
  invalid_dropped: number;
  llm_calls: number;
  chunks: number;
  error: string | null;
  embeddings_refresh_warning: string | null;
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
  error: null,
  embeddings_refresh_warning: null,
});

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
}

/**
 * The shape withStructuredOutput(schema, { includeRaw: true }) resolves to
 * for a plain (non-Zod) JSON Schema like MODEL_OUTPUT_JSON_SCHEMA: RunOutput
 * stays at its Record<string, any> default. The .d.ts types `parsed` as
 * always-present, but the implementation's own includeRaw:true fallback
 * (@langchain/core's assembleStructuredOutputPipeline) can still resolve it
 * to null on a failed extraction — invokeForOutput() below checks for that.
 */
type StructuredOutputResult = { raw: BaseMessage; parsed: Record<string, unknown> };

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

  private readonly llm: BaseChatModel;
  private readonly client: Taguru;
  private readonly create_context: boolean;
  private readonly context_description: string | undefined;
  private readonly structuredLlm: Runnable<BaseLanguageModelInput, StructuredOutputResult> | null;

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
   * One model call for one attempt, returning `{ content, output, error,
   * finishReason }` with `output === null` iff `error !== null`. Goes
   * through the withStructuredOutput() pipeline built in the constructor
   * when structured_output is on, else today's plain invoke() +
   * parseModelOutput(). Either path lands in a ModelOutput the same way —
   * coerceOutput() — so schema-constrained generation narrows the model's
   * answer without skipping revalidation. finishReason comes from the
   * provider regardless of parse success, so the retry loop can tell a
   * length-limited answer from an ordinary malformed one.
   */
  private async invokeForOutput(messages: BaseMessage[]): Promise<{
    content: string;
    output: ModelOutput | null;
    error: Error | null;
    finishReason: string | undefined;
  }> {
    if (this.structuredLlm !== null) {
      const result = await this.structuredLlm.invoke(messages);
      const content = contentText(result.raw);
      const finishReason = extractFinishReason(result.raw);
      // Narrow through unknown first — see StructuredOutputResult's comment
      // on why `parsed` can be null here despite its declared type.
      const parsed: unknown = result.parsed;
      if (parsed === undefined || parsed === null) {
        return {
          content,
          output: null,
          error: new Error(
            "the model's structured-output call produced no parsed result " +
              "(LangChain.js surfaces no further detail here — see " +
              "assembleStructuredOutputPipeline's includeRaw:true fallback)",
          ),
          finishReason,
        };
      }
      try {
        return { content, output: coerceOutput(parsed), error: null, finishReason };
      } catch (error) {
        return { content, output: null, error: error as Error, finishReason };
      }
    }
    const response = await this.llm.invoke(messages);
    const content = contentText(response);
    const finishReason = extractFinishReason(response);
    try {
      return { content, output: parseModelOutput(content), error: null, finishReason };
    } catch (error) {
      return { content, output: null, error: error as Error, finishReason };
    }
  }

  /**
   * Ingest one text under one source id. Throws on failure (unlike
   * `ingestDocuments`, there is no "continue with the rest" here).
   */
  async ingestText(
    text: string,
    options: { source: string; dry_run?: boolean },
  ): Promise<IngestOutcome> {
    const outcome = emptyOutcome(options.source);
    if (this.include_passage && Buffer.byteLength(text, "utf-8") > MAX_PASSAGE_BYTES) {
      throw new Error(`document exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`);
    }

    const vocabulary = await this.fetchVocabulary();
    const system = systemPrompt(vocabulary, this.questions, this.fact_budget);
    const chunks = chunk(labeledDocument(text, this.chunk_bytes), this.chunk_bytes);
    outcome.chunks = chunks.length;

    const outputs: ModelOutput[] = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const user = userMessage(options.source, index, chunks.length, chunks[index]!);
      const base: BaseMessage[] = [new SystemMessage(system), new HumanMessage(user)];
      let output: ModelOutput | null = null;
      // Each retry rebuilds the conversation from the system/user base and
      // appends only the most recent bad turn — never the whole history —
      // so corrective_context_bytes bounds every retry alike, not just the
      // first. At the all-defaults policy this reproduces the previous
      // fixed implementation's request bodies exactly.
      let priorBadTurn: { content: string; error: Error; lengthLimited: boolean } | null = null;
      for (let attempt = 0; attempt < this.max_attempts && output === null; attempt += 1) {
        const messages: BaseMessage[] =
          priorBadTurn === null
            ? base
            : [
                ...base,
                new AIMessage(
                  correctiveAssistantTurnContent(
                    priorBadTurn.content,
                    this.corrective_context_bytes,
                  ),
                ),
                new HumanMessage(
                  correctiveMessage(
                    priorBadTurn.error.message,
                    priorBadTurn.lengthLimited,
                    this.fact_budget,
                  ),
                ),
              ];
        const { content, output: parsed, error, finishReason } = await this.invokeForOutput(messages);
        outcome.llm_calls += 1;
        if (error === null) {
          output = parsed;
        } else {
          priorBadTurn = { content, error, lengthLimited: indicatesLengthLimit(finishReason) };
        }
      }
      if (output === null) {
        throw new Error(
          `the model would not produce the JSON object: ${priorBadTurn?.error.message}`,
        );
      }
      outputs.push(output);
    }

    const paragraphCount = splitParagraphs(text).length;
    const extraction = merge(outputs, this.questions, paragraphCount);
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

    const applied = await this.client.importBatches(ndjson);
    record(outcome, applied.batches[0]!);
    outcome.ok = true;

    if (this.refresh_embeddings) {
      try {
        await this.client.context(this.context).refreshEmbeddings();
      } catch (error) {
        if (error instanceof EmbeddingUnavailableError && error.reason === "provider_error") {
          outcome.embeddings_refresh_warning = error.message;
        } else if (!(error instanceof EmbeddingUnavailableError)) {
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
    options: { dry_run?: boolean } = {},
  ): Promise<IngestOutcome[]> {
    const outcomes: IngestOutcome[] = [];
    for (const document of documents) {
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
        outcomes.push(
          await this.ingestText(document.pageContent, { source, dry_run: options.dry_run }),
        );
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

function extractFinishReason(message: unknown): string | undefined {
  const metadata = (message as { response_metadata?: unknown }).response_metadata;
  if (typeof metadata !== "object" || metadata === null) {
    return undefined;
  }
  const record = metadata as Record<string, unknown>;
  for (const key of FINISH_REASON_KEYS) {
    const value = record[key];
    if (value !== undefined && value !== null) {
      return String(value);
    }
  }
  return undefined;
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
