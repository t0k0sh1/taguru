/**
 * TaguruIngester: LLM-driven document decomposition into a Taguru context.
 * The mechanical mirror of the Python `taguru_langchain.TaguruIngester` —
 * see that module for the full design rationale (import-based per-source
 * replace, no source-id fallback, live vocabulary seeding, dry_run).
 */

import type { DocumentInterface } from "@langchain/core/documents";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, SystemMessage, type BaseMessage } from "@langchain/core/messages";
import {
  EmbeddingUnavailableError,
  NotFoundError,
  Taguru,
  TaguruError,
  type ImportOutcome,
} from "taguru";

import {
  CHUNK_BYTES,
  MAX_PASSAGE_BYTES,
  MAX_QUESTIONS_PER_PARAGRAPH,
  VOCABULARY_CAP,
  chunk,
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
  include_passage?: boolean;
  chunk_bytes?: number;
  vocabulary_cap?: number;
  refresh_embeddings?: boolean;
  raise_on_error?: boolean;
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
  readonly include_passage: boolean;
  readonly chunk_bytes: number;
  readonly vocabulary_cap: number;
  readonly refresh_embeddings: boolean;
  readonly raise_on_error: boolean;

  private readonly llm: BaseChatModel;
  private readonly client: Taguru;
  private readonly create_context: boolean;
  private readonly context_description: string | undefined;

  constructor(fields: TaguruIngesterFields) {
    if (fields.create_context && fields.context_description === undefined) {
      throw new Error("create_context: true requires context_description");
    }
    const questions = fields.questions ?? 0;
    if (questions < 0 || questions > MAX_QUESTIONS_PER_PARAGRAPH) {
      throw new Error(`questions must be between 0 and ${MAX_QUESTIONS_PER_PARAGRAPH}`);
    }
    this.context = fields.context;
    this.llm = fields.llm;
    this.client =
      fields.client ?? new Taguru({ base_url: fields.base_url, api_key: fields.api_key });
    this.create_context = fields.create_context ?? false;
    this.context_description = fields.context_description;
    this.source_key = fields.source_key ?? "source";
    this.questions = questions;
    this.include_passage = fields.include_passage ?? true;
    this.chunk_bytes = fields.chunk_bytes ?? CHUNK_BYTES;
    this.vocabulary_cap = fields.vocabulary_cap ?? VOCABULARY_CAP;
    this.refresh_embeddings = fields.refresh_embeddings ?? true;
    this.raise_on_error = fields.raise_on_error ?? false;
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
    if (Buffer.byteLength(text, "utf-8") > MAX_PASSAGE_BYTES) {
      throw new Error(`document exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`);
    }

    const vocabulary = await this.fetchVocabulary();
    const system = systemPrompt(vocabulary, this.questions);
    const chunks = chunk(labeledDocument(text), this.chunk_bytes);
    outcome.chunks = chunks.length;

    const outputs: ModelOutput[] = [];
    for (let index = 0; index < chunks.length; index += 1) {
      const user = userMessage(options.source, index, chunks.length, chunks[index]!);
      let messages: BaseMessage[] = [
        new SystemMessage(system),
        new HumanMessage(user),
      ];
      let output: ModelOutput | null = null;
      let lastError: Error | null = null;
      for (let attempt = 0; attempt < 2 && output === null; attempt += 1) {
        const response = await this.llm.invoke(messages);
        outcome.llm_calls += 1;
        const content = contentText(response);
        try {
          output = parseModelOutput(content);
        } catch (error) {
          lastError = error as Error;
          messages = [
            ...messages,
            new AIMessage(content),
            new HumanMessage(
              `That was not the single JSON object asked for (${lastError.message}). ` +
                "Answer again with only the JSON object.",
            ),
          ];
        }
      }
      if (output === null) {
        throw new Error(`the model would not produce the JSON object: ${lastError?.message}`);
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
        if (this.raise_on_error || !(error instanceof Error && isIngestFailure(error))) {
          throw error;
        }
        const failed = emptyOutcome(source);
        failed.error = error.message;
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

function isIngestFailure(error: Error): boolean {
  return error instanceof TaguruError || error.name === "Error";
}

function record(outcome: IngestOutcome, applied: ImportOutcome): void {
  outcome.created = applied.created;
  outcome.retracted = applied.retracted;
  outcome.associations = applied.associations;
  outcome.aliases = applied.aliases;
  outcome.passage_stored = applied.passage_stored;
  outcome.questions_stored = applied.questions_stored;
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
