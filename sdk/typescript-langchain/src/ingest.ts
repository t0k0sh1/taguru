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
  correctiveAssistantTurnContent,
  correctiveMessage,
  correctiveValidationMessage,
  crossOutputIssues,
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
   * Stage 2 cross-chunk alias correction) this ingest needed — 0 means
   * every chunk's first answer was accepted as-is. */
  correction_attempts: number;
  /** Labels of automatic, information-preserving JSON repairs applied
   * across every accepted answer (e.g. "trailing_comma", "bom",
   * "code_fence", "braces_slice" — see candidateJson). Empty when no
   * repair was ever needed. */
  lossless_repairs: string[];
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
  correction_attempts: 0,
  lossless_repairs: [],
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
}

function classifyText(
  content: string,
  finishReason: string | undefined,
  rules: ItemRules | null,
): AttemptResult {
  if (indicatesLengthLimit(finishReason)) {
    return {
      kind: "length_limited",
      content,
      output: null,
      repairs: [],
      error: "the answer hit the provider's output limit",
      issues: null,
    };
  }
  if (indicatesRefusal(finishReason)) {
    return { kind: "refusal", content, output: null, repairs: [], error: finishReason!, issues: null };
  }
  if (isEmptyAnswer(content)) {
    return {
      kind: "empty",
      content,
      output: null,
      repairs: [],
      error: emptyAnswerDiagnosis(),
      issues: null,
    };
  }
  try {
    const { output, repairs } = evaluateAnswer(content, rules);
    return { kind: "valid", content, output, repairs, error: null, issues: null };
  } catch (error) {
    if (error instanceof InvalidFault) {
      return { kind: "invalid", content, output: null, repairs: [], error: null, issues: error.issues };
    }
    if (error instanceof SyntaxFault) {
      return {
        kind: "syntax",
        content,
        output: null,
        repairs: [],
        error: error.message,
        issues: null,
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
  finishReason: string | undefined,
  parsed: Record<string, unknown> | null | undefined,
  rules: ItemRules | null,
): AttemptResult {
  if (indicatesLengthLimit(finishReason)) {
    return {
      kind: "length_limited",
      content,
      output: null,
      repairs: [],
      error: "the answer hit the provider's output limit",
      issues: null,
    };
  }
  if (indicatesRefusal(finishReason)) {
    return { kind: "refusal", content, output: null, repairs: [], error: finishReason!, issues: null };
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
    };
  }
  const { output, issues } = interpretModelOutput(parsed, effectiveItemRules(rules));
  if (rules !== null && issues.length > 0) {
    return { kind: "invalid", content, output: null, repairs: [], error: null, issues };
  }
  return { kind: "valid", content, output, repairs: [], error: null, issues: null };
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
 * The Stage 2 (cross-chunk alias correction) terminal message for one
 * offending chunk's non-valid reply — mirrors extract.rs's
 * correct_cross_output_issues per-kind texts verbatim.
 */
function crossChunkFailureMessage(label: string, result: AttemptResult): string {
  if (result.kind === "length_limited") {
    return (
      `${label}: the cross-chunk alias correction was cut off at the output limit — ` +
      "failing the source rather than importing a truncated correction"
    );
  }
  if (result.kind === "refusal") {
    return `${label}: the provider refused the cross-chunk alias correction (finish_reason ${result.error})`;
  }
  if (result.kind === "empty") {
    return `${label}: ${result.error}`;
  }
  if (result.kind === "invalid") {
    const issues = result.issues!;
    return (
      `${label}: the cross-chunk alias correction still left ${issues.length} invalid item(s) ` +
      `uncorrected: ${issues.join("; ")}`
    );
  }
  return `${label}: the cross-chunk alias correction was not the JSON object asked for (${result.error})`;
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
    this.lossy = fields.lossy ?? false;
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
      const finishReason = extractFinishReason(result.raw);
      return classifyStructured(content, finishReason, result.parsed, rules);
    }
    const response = await this.llm.invoke(messages);
    const content = contentText(response);
    const finishReason = extractFinishReason(response);
    return classifyText(content, finishReason, rules);
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
  ): Promise<void> {
    for (const [recordIndex, issues] of crossOutputIssues(records.map((r) => r.output))) {
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
      throw new Error(crossChunkFailureMessage(label, result));
    }

    // Re-check rather than trust the single corrective turn blindly: a
    // correction can rename an association another chunk's alias depended
    // on, introducing a FRESH cross-chunk issue. This is the bounded
    // re-check, not a second round — any issue here fails the source.
    const recheck = crossOutputIssues(records.map((r) => r.output));
    if (recheck.length > 0) {
      const [recordIndex, issues] = recheck[0]!;
      const chunkIndex = records[recordIndex]!.chunkIndex;
      throw new Error(
        `chunk ${chunkIndex + 1}/${chunkTotal}: still has ${issues.length} cross-chunk alias ` +
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
    options: { source: string; dry_run?: boolean },
  ): Promise<IngestOutcome> {
    const outcome = emptyOutcome(options.source);
    if (this.include_passage && Buffer.byteLength(text, "utf-8") > MAX_PASSAGE_BYTES) {
      throw new Error(`document exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`);
    }

    const vocabulary = await this.fetchVocabulary();
    const system = systemPrompt(vocabulary, this.questions, this.fact_budget);
    const paragraphCount = splitParagraphs(text).length;
    const rules = this.itemRules(paragraphCount);
    const chunks = chunk(labeledDocument(text, this.chunk_bytes), this.chunk_bytes);
    outcome.chunks = chunks.length;

    const records: ChunkRecord[] = [];
    for (let index = 0; index < chunks.length; index += 1) {
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
        const result = await this.attemptOnce(messages, rules);
        outcome.llm_calls += 1;

        if (result.kind === "valid") {
          chunkRecord = { output: result.output!, chunkIndex: index, user, answer: result.content };
          outcome.lossless_repairs.push(...result.repairs);
          break;
        }

        if (result.kind === "refusal") {
          throw new Error(
            `the provider refused this content (finish_reason ${result.error}) — a policy ` +
              "refusal is terminal; no corrective turn can change it",
          );
        }

        if (result.kind === "empty" && emptyCorrected) {
          throw new Error(diagnosisFor(result));
        }
        if (result.kind === "empty") {
          emptyCorrected = true;
        }

        lastDiagnosis = finalMessageFor(result);
        outcome.correction_attempts += 1;
        pendingAsk = correctiveAskFor(result, this.fact_budget);
        priorBadAnswer = result.content;
      }

      if (chunkRecord === null) {
        throw new Error(lastDiagnosis);
      }
      records.push(chunkRecord);
    }

    if (!this.lossy) {
      await this.correctCrossChunkIssues(system, records, rules, chunks.length, outcome);
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
