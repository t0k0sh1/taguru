/**
 * The Taguru client. Mirrors the Python SDK surface exactly: identical
 * structure, vocabulary, arguments, and returns — method names differ only by
 * casing convention (search_passages ↔ searchPassages), data fields are the
 * wire's own snake_case in both languages.
 */

import { NotFoundError, TaguruError, TransportError } from "./errors.js";
import type {
  Activation,
  ActivationPage,
  AliasEntry,
  AliasPage,
  AssocOp,
  Association,
  BatchApplyResult,
  Citation,
  CompactOutcome,
  ConceptDescription,
  ContextMeta,
  ContextPage,
  CrossMatchCursor,
  CrossMatchPage,
  CrossPassageHit,
  DirectoryEntry,
  ExploreCursor,
  ExplorePage,
  GroupEntry,
  GroupPage,
  ImportOutcome,
  LabelPage,
  MatchCursor,
  MatchPage,
  OneOrMany,
  PassageHit,
  PassageLookup,
  QuestionSpec,
  RefreshOutcome,
  RetractAssociationOutcome,
  RetractOutcome,
  RetrievalResult,
  SectionSpec,
  SourcePage,
  StoredPassages,
  TieredResolution,
  VocabularyAudit,
} from "./models.js";
import { citationKey } from "./models.js";
import {
  DEFAULT_RETRIES,
  type RetryClass,
  backoffDelay,
  parseRetryAfter,
  shouldRetryStatus,
  shouldRetryTransport,
} from "./retry.js";
import {
  DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT_SECS,
  ENV_TOKEN,
  ENV_URL,
  MAX_CHUNK_BYTES,
  MAX_OPS_PER_REQUEST,
  chunkAssociations,
  describeError,
  dropUndefined,
  encodeName,
  errorFromBody,
  isPreConnectFailure,
  normalizeImportOutcomes,
  sleep,
  unwrapEnvelope,
} from "./transport.js";

export interface TaguruOptions {
  /** Server base URL; defaults to $TAGURU_URL then http://127.0.0.1:8248. */
  base_url?: string;
  /**
   * Bearer token; defaults to $TAGURU_API_TOKEN (unset means the server runs
   * unauthenticated — dev mode).
   */
  api_key?: string;
  /**
   * Per-request budget in seconds. Matches the server's own default; raise
   * both together when the server calls an embedding provider.
   */
  timeout?: number;
  /** Additional attempts after the first, for retry-safe failures. */
  retries?: number;
  /** Extra headers sent on every request. */
  headers?: Record<string, string>;
  /** Injectable fetch (tests, non-Node runtimes). */
  fetch?: typeof fetch;
}

interface SendOptions {
  params?: Record<string, unknown>;
  jsonBody?: unknown;
  content?: string | Uint8Array;
  contentType?: string;
  retry?: RetryClass;
  retries?: number;
}

interface SentResponse {
  status: number;
  headers: Headers;
  text: string;
}

/** Client for one Taguru server. */
export class Taguru {
  readonly contexts: Contexts;
  readonly groups: Groups;

  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly retries: number;
  private readonly timeoutSecs: number;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof fetch;

  constructor(options: TaguruOptions = {}) {
    const env = typeof process !== "undefined" ? process.env : undefined;
    this.baseUrl = (options.base_url ?? env?.[ENV_URL] ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.api_key ?? env?.[ENV_TOKEN];
    this.retries = options.retries ?? DEFAULT_RETRIES;
    this.timeoutSecs = options.timeout ?? DEFAULT_TIMEOUT_SECS;
    this.headers = { ...(options.headers ?? {}) };
    if (this.apiKey) {
      this.headers["authorization"] = `Bearer ${this.apiKey}`;
    }
    this.fetchImpl = options.fetch ?? fetch;
    this.contexts = new Contexts(this);
    this.groups = new Groups(this);
  }

  // -- transport ---------------------------------------------------------

  /** @internal */
  async send(method: string, path: string, options: SendOptions = {}): Promise<SentResponse> {
    let url = this.baseUrl + path;
    if (options.params !== undefined) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(dropUndefined(options.params))) {
        params.set(key, String(value));
      }
      const query = params.toString();
      if (query) {
        url += `?${query}`;
      }
    }
    const headers: Record<string, string> = { ...this.headers };
    let body: string | Uint8Array | undefined = options.content;
    if (options.jsonBody !== undefined) {
      body = JSON.stringify(options.jsonBody);
      headers["content-type"] = "application/json";
    } else if (options.content !== undefined && options.contentType !== undefined) {
      headers["content-type"] = options.contentType;
    }
    const retryClass = options.retry ?? "safe";
    const maxAttempts = (options.retries ?? this.retries) + 1;
    let attempt = 0;
    for (;;) {
      let response: Response;
      try {
        response = await this.fetchImpl(url, {
          method,
          headers,
          body: body as BodyInit | undefined,
          signal: AbortSignal.timeout(this.timeoutSecs * 1000),
        });
      } catch (error) {
        // A pre-connect failure never reached the server; anything after
        // that is ambiguous (the request may have executed).
        const ambiguous = !isPreConnectFailure(error);
        attempt += 1;
        if (attempt < maxAttempts && shouldRetryTransport(ambiguous, retryClass)) {
          await sleep(backoffDelay(attempt - 1));
          continue;
        }
        throw new TransportError(describeError(error), { cause: error });
      }
      if (response.status >= 400) {
        const bodyText = await response.text();
        attempt += 1;
        if (attempt < maxAttempts && shouldRetryStatus(response.status, retryClass)) {
          const delay = parseRetryAfter(response.headers.get("retry-after"));
          await sleep(delay ?? backoffDelay(attempt - 1));
          continue;
        }
        throw errorFromBody(response.status, response.headers.get("retry-after"), bodyText);
      }
      return { status: response.status, headers: response.headers, text: await response.text() };
    }
  }

  /** @internal */
  async requestJson(method: string, path: string, options: SendOptions = {}): Promise<unknown> {
    const response = await this.send(method, path, options);
    return unwrapEnvelope(response.status, response.text);
  }

  /** @internal */
  streamUrl(path: string): { url: string; headers: Record<string, string>; fetchImpl: typeof fetch } {
    return { url: this.baseUrl + path, headers: { ...this.headers }, fetchImpl: this.fetchImpl };
  }

  // -- server-level operations -------------------------------------------

  /** Readiness probe: throws ServiceUnavailableError while degraded. */
  async health(): Promise<void> {
    await this.send("GET", "/health", { retries: 0 });
  }

  /** Liveness probe: succeeds as long as the process answers at all. */
  async live(): Promise<void> {
    await this.send("GET", "/live", { retries: 0 });
  }

  /** Prometheus exposition text (not the JSON envelope). */
  async metrics(): Promise<string> {
    return (await this.send("GET", "/metrics")).text;
  }

  /** The client protocol document (markdown) this server ships. */
  async protocol(): Promise<string> {
    return (await this.send("GET", "/protocol")).text;
  }

  /** Persist every dirty context now; returns their names (admin role). */
  async flush(): Promise<string[]> {
    const result = await this.requestJson("POST", "/flush");
    return (result as unknown[]).map(String);
  }

  /**
   * Apply an NDJSON batch stream (the format `export` produces). Each batch
   * is one source's retract-then-apply, so re-importing is idempotent. The
   * response is normalized to an array even for a single batch.
   */
  async importBatches(data: string | Uint8Array): Promise<ImportOutcome[]> {
    const response = await this.send("POST", "/import", {
      content: data,
      contentType: "application/x-ndjson",
    });
    return normalizeImportOutcomes(unwrapEnvelope(response.status, response.text));
  }

  /** Apply an NDJSON batch file (see `importBatches`). */
  async importFile(path: string): Promise<ImportOutcome[]> {
    const { readFile } = await import("node:fs/promises");
    return this.importBatches(await readFile(path));
  }

  /** Poll `live` then `health` until both pass or `timeout` elapses. */
  async waitUntilReady(options: { timeout?: number; interval?: number } = {}): Promise<void> {
    const timeout = options.timeout ?? 30.0;
    const interval = options.interval ?? 0.5;
    const deadline = Date.now() + timeout * 1000;
    let lastError: TaguruError | null = null;
    for (;;) {
      try {
        await this.live();
        await this.health();
        return;
      } catch (error) {
        lastError = error instanceof TaguruError ? error : new TaguruError(describeError(error));
      }
      if (Date.now() >= deadline) {
        throw lastError ?? new TaguruError(`server not ready after ${timeout} seconds`);
      }
      await sleep(interval);
    }
  }

  /** A handle bound to one context (no network call). */
  context(name: string): Context {
    return new Context(this, name);
  }

  // -- cross-context search ------------------------------------------------

  /**
   * Recall across several contexts at once, every match tagged. `contexts`
   * takes full names; each `groups` entry searches every context the group
   * reaches (nested children included), overlaps deduped. At least one of
   * the two must name something. Weights share one scale, so past the limit
   * the strongest |weight| survives exactly as within one context. `after`
   * resumes past the previous page's last match; `total` stays constant
   * across pages.
   */
  async recall(
    cue: string,
    options: {
      contexts?: string[];
      groups?: string[];
      limit?: number;
      after?: CrossMatchCursor;
    } = {},
  ): Promise<CrossMatchPage> {
    const result = await this.requestJson("POST", "/recall", {
      jsonBody: dropUndefined({
        contexts: options.contexts,
        groups: options.groups,
        cue,
        limit: options.limit,
        after: options.after,
      }),
    });
    return result as CrossMatchPage;
  }

  /**
   * Exact-position query across several contexts at once, matches tagged;
   * the same target contract as `recall`.
   */
  async query(
    options: {
      contexts?: string[];
      groups?: string[];
      subject?: OneOrMany;
      label?: OneOrMany;
      object?: OneOrMany;
      limit?: number;
      after?: CrossMatchCursor;
    } = {},
  ): Promise<CrossMatchPage> {
    const result = await this.requestJson("POST", "/query", {
      jsonBody: dropUndefined({
        contexts: options.contexts,
        groups: options.groups,
        subject: options.subject,
        label: options.label,
        object: options.object,
        limit: options.limit,
        after: options.after,
      }),
    });
    return result as CrossMatchPage;
  }

  /**
   * Paragraph search across several contexts at once, hits tagged. Passage
   * scores do NOT share a scale across contexts (BM25 statistics are
   * corpus-local), so the merged order is rank interleaving — every
   * context's best hit first; `score` compares within one context only.
   */
  async searchPassages(
    query: string,
    options: { contexts?: string[]; groups?: string[]; limit?: number } = {},
  ): Promise<CrossPassageHit[]> {
    const result = await this.requestJson("POST", "/sources/search", {
      jsonBody: dropUndefined({
        contexts: options.contexts,
        groups: options.groups,
        query,
        limit: options.limit,
      }),
    });
    return result as CrossPassageHit[];
  }

  /**
   * Release resources. The fetch-based transport holds no persistent state,
   * so this is a no-op kept for surface parity with the Python SDK.
   */
  close(): void {
    // no-op
  }
}

/** The context directory: collection-level CRUD. */
export class Contexts {
  constructor(private readonly client: Taguru) {}

  /** One directory page (keyset cursor: `after` = last name shown). */
  async list(options: { limit?: number; after?: string } = {}): Promise<ContextPage> {
    const result = await this.client.requestJson("GET", "/contexts", {
      params: { limit: options.limit, after: options.after },
    });
    return result as ContextPage;
  }

  /** Walk every directory page transparently. */
  async *iter(options: { limit?: number } = {}): AsyncGenerator<DirectoryEntry, void, undefined> {
    let after: string | undefined;
    for (;;) {
      const page = await this.list({ limit: options.limit, after });
      if (page.contexts.length === 0) {
        return;
      }
      yield* page.contexts;
      if (options.limit !== undefined && page.contexts.length < options.limit) {
        return;
      }
      after = page.contexts[page.contexts.length - 1]!.name;
    }
  }

  async get(name: string): Promise<DirectoryEntry> {
    const result = await this.client.requestJson("GET", `/contexts/${encodeName(name)}`);
    return result as DirectoryEntry;
  }

  async exists(name: string): Promise<boolean> {
    try {
      await this.get(name);
    } catch (error) {
      if (error instanceof NotFoundError) {
        return false;
      }
      throw error;
    }
    return true;
  }

  /** Create a context (409 ConflictError if it already exists). */
  async create(
    name: string,
    options: {
      description?: string;
      pinned?: boolean;
      dice_floor?: number;
      semantic_floor?: number;
    } = {},
  ): Promise<boolean> {
    const result = await this.client.requestJson("PUT", `/contexts/${encodeName(name)}`, {
      jsonBody: dropUndefined({
        description: options.description ?? "",
        pinned: options.pinned ?? false,
        dice_floor: options.dice_floor,
        semantic_floor: options.semantic_floor,
      }),
      retry: "unsafe_on_ambiguous",
    });
    return Boolean(result);
  }

  /** Update metadata; an omitted field is left unchanged. */
  async update(
    name: string,
    options: {
      description?: string;
      pinned?: boolean;
      dice_floor?: number;
      semantic_floor?: number;
    } = {},
  ): Promise<ContextMeta> {
    const result = await this.client.requestJson("PATCH", `/contexts/${encodeName(name)}`, {
      jsonBody: dropUndefined({
        description: options.description,
        pinned: options.pinned,
        dice_floor: options.dice_floor,
        semantic_floor: options.semantic_floor,
      }),
    });
    return result as ContextMeta;
  }

  /** Delete a context, files included (admin role). */
  async delete(name: string): Promise<boolean> {
    const result = await this.client.requestJson("DELETE", `/contexts/${encodeName(name)}`);
    return Boolean(result);
  }
}

/**
 * The group directory: flat context bundles (many-to-many) that may nest
 * child groups — a shallow DAG, at most 3 storeys, never cyclic — as the
 * addressing unit cross-context search builds on.
 */
export class Groups {
  constructor(private readonly client: Taguru) {}

  /** One directory page (keyset cursor: `after` = last name shown). */
  async list(options: { limit?: number; after?: string } = {}): Promise<GroupPage> {
    const result = await this.client.requestJson("GET", "/groups", {
      params: { limit: options.limit, after: options.after },
    });
    return result as GroupPage;
  }

  /** Walk every directory page transparently. */
  async *iter(options: { limit?: number } = {}): AsyncGenerator<GroupEntry, void, undefined> {
    let after: string | undefined;
    for (;;) {
      const page = await this.list({ limit: options.limit, after });
      if (page.groups.length === 0) {
        return;
      }
      yield* page.groups;
      if (options.limit !== undefined && page.groups.length < options.limit) {
        return;
      }
      after = page.groups[page.groups.length - 1]!.name;
    }
  }

  async get(name: string): Promise<GroupEntry> {
    const result = await this.client.requestJson("GET", `/groups/${encodeName(name)}`);
    return result as GroupEntry;
  }

  async exists(name: string): Promise<boolean> {
    try {
      await this.get(name);
    } catch (error) {
      if (error instanceof NotFoundError) {
        return false;
      }
      throw error;
    }
    return true;
  }

  /**
   * Create a group (409 ConflictError if it already exists). Every listed
   * member — context or child group — must already exist; contexts and
   * groups are separate namespaces.
   */
  async create(
    name: string,
    options: { description?: string; contexts?: string[]; groups?: string[] } = {},
  ): Promise<boolean> {
    const result = await this.client.requestJson("PUT", `/groups/${encodeName(name)}`, {
      jsonBody: dropUndefined({
        description: options.description ?? "",
        contexts: options.contexts,
        groups: options.groups,
      }),
      retry: "unsafe_on_ambiguous",
    });
    return Boolean(result);
  }

  /**
   * Delta membership update (removals first); returns the updated row.
   * Removing a non-member is an idempotent no-op; only additions demand the
   * member exists. The result holds at most 1,000 member contexts and 1,000
   * child groups — past that, split into nested child groups.
   */
  async update(
    name: string,
    options: {
      description?: string;
      add_contexts?: string[];
      remove_contexts?: string[];
      add_groups?: string[];
      remove_groups?: string[];
    } = {},
  ): Promise<GroupEntry> {
    const result = await this.client.requestJson("PATCH", `/groups/${encodeName(name)}`, {
      jsonBody: dropUndefined({
        description: options.description,
        add_contexts: options.add_contexts,
        remove_contexts: options.remove_contexts,
        add_groups: options.add_groups,
        remove_groups: options.remove_groups,
      }),
    });
    return result as GroupEntry;
  }

  /** Delete the bundling only — member contexts and child groups stay. */
  async delete(name: string): Promise<boolean> {
    const result = await this.client.requestJson("DELETE", `/groups/${encodeName(name)}`);
    return Boolean(result);
  }

  /**
   * The group as one import-stream record (a `taguru_group` JSON line);
   * `importBatches` restores it as a whole-record replace.
   */
  async export(name: string): Promise<string> {
    return (await this.client.send("GET", `/groups/${encodeName(name)}/export`)).text;
  }
}

/**
 * Operations bound to one context, named after the server's own vocabulary.
 * Method names mirror `GET /protocol` and the MCP tool names, so knowledge of
 * one surface transfers to the others.
 */
export class Context {
  readonly name: string;
  private readonly path: string;

  constructor(
    private readonly client: Taguru,
    name: string,
  ) {
    this.name = name;
    this.path = `/contexts/${encodeName(name)}`;
  }

  private async post(suffix: string, jsonBody?: unknown, retry?: RetryClass): Promise<unknown> {
    return this.client.requestJson("POST", this.path + suffix, { jsonBody, retry });
  }

  // -- entry resolution ---------------------------------------------------

  /**
   * Concept candidates for a cue. Read `gloss` before adopting a
   * containment/fuzzy hit — never adopt a lookalike on score alone.
   */
  async resolve(
    cue: string,
    options: { dice_floor?: number; semantic_floor?: number; limit?: number } = {},
  ): Promise<TieredResolution[]> {
    const result = await this.post(
      "/resolve",
      dropUndefined({
        cue,
        dice_floor: options.dice_floor,
        semantic_floor: options.semantic_floor,
        limit: options.limit,
      }),
    );
    return result as TieredResolution[];
  }

  /** Relation-label candidates for a cue. */
  async resolveLabel(
    cue: string,
    options: { dice_floor?: number; semantic_floor?: number; limit?: number } = {},
  ): Promise<TieredResolution[]> {
    const result = await this.post(
      "/resolve_label",
      dropUndefined({
        cue,
        dice_floor: options.dice_floor,
        semantic_floor: options.semantic_floor,
        limit: options.limit,
      }),
    );
    return result as TieredResolution[];
  }

  // -- graph reads ---------------------------------------------------------

  /**
   * Associations whose subject/object entry-matches the cue. `after`
   * resumes past the previous page's last match; `total` stays constant
   * across pages.
   */
  async recall(
    cue: string,
    options: { limit?: number; after?: MatchCursor } = {},
  ): Promise<MatchPage> {
    const result = await this.post(
      "/recall",
      dropUndefined({ cue, limit: options.limit, after: options.after }),
    );
    return result as MatchPage;
  }

  /** Exact-position query; each position takes one name or an OR-set. */
  async query(
    options: {
      subject?: OneOrMany;
      label?: OneOrMany;
      object?: OneOrMany;
      limit?: number;
      after?: MatchCursor;
    } = {},
  ): Promise<MatchPage> {
    const result = await this.post(
      "/query",
      dropUndefined({
        subject: options.subject,
        label: options.label,
        object: options.object,
        limit: options.limit,
        after: options.after,
      }),
    );
    return result as MatchPage;
  }

  /** Label outline (counts per role); null for an unknown concept. */
  async describe(concept: string): Promise<ConceptDescription | null> {
    const result = await this.post("/describe", { concept });
    return (result ?? null) as ConceptDescription | null;
  }

  /**
   * Exhaustive hop-annotated walk (truncation keeps the nearest). `after`
   * resumes past the previous page's last recollection; `total` stays
   * constant across pages.
   */
  async explore(
    origins: string | string[],
    options: { max_depth?: number; limit?: number; after?: ExploreCursor } = {},
  ): Promise<ExplorePage> {
    const result = await this.post(
      "/explore",
      dropUndefined({
        origins: typeof origins === "string" ? [origins] : origins,
        max_depth: options.max_depth,
        limit: options.limit,
        after: options.after,
      }),
    );
    return result as ExplorePage;
  }

  /** Spreading activation from origins, strongest first. */
  async activate(
    origins: string | string[],
    options: { decay?: number; limit?: number } = {},
  ): Promise<ActivationPage> {
    const result = await this.post(
      "/activate",
      dropUndefined({
        origins: typeof origins === "string" ? [origins] : origins,
        decay: options.decay,
        limit: options.limit,
      }),
    );
    return result as ActivationPage;
  }

  /**
   * Coverage audit: associations not reachable from the origins. `after`
   * resumes past the previous page's last match; `total` stays constant
   * across pages.
   */
  async unreachableFrom(
    origins: string | string[],
    options: { limit?: number; after?: MatchCursor } = {},
  ): Promise<MatchPage> {
    const result = await this.post(
      "/unreachable_from",
      dropUndefined({
        origins: typeof origins === "string" ? [origins] : origins,
        limit: options.limit,
        after: options.after,
      }),
    );
    return result as MatchPage;
  }

  /** One page of the relation vocabulary (canonical labels only). */
  async listLabels(options: { limit?: number; after?: string } = {}): Promise<LabelPage> {
    const result = await this.client.requestJson("GET", `${this.path}/labels`, {
      params: { limit: options.limit, after: options.after },
    });
    return result as LabelPage;
  }

  async *iterLabels(options: { limit?: number } = {}): AsyncGenerator<string, void, undefined> {
    let after: string | undefined;
    for (;;) {
      const page = await this.listLabels({ limit: options.limit, after });
      if (page.labels.length === 0) {
        return;
      }
      yield* page.labels;
      if (options.limit !== undefined && page.labels.length < options.limit) {
        return;
      }
      after = page.labels[page.labels.length - 1]!;
    }
  }

  // -- graph writes ---------------------------------------------------------

  /**
   * Assert a batch of associations; returns the applied count.
   *
   * Weight ACCUMULATES on re-assertion, so this call is never blindly retried
   * after an ambiguous transport failure. Server cap: 10,000 per request (use
   * `addAssociationsBatched` to auto-chunk).
   */
  async addAssociations(associations: AssocOp[]): Promise<number> {
    const result = await this.post("/associations", associations, "unsafe_on_ambiguous");
    return Number(result);
  }

  /**
   * Chunked `addAssociations` for arbitrarily large batches. Chunks are
   * independent requests: a failure mid-way leaves earlier chunks applied
   * (that is why this is a separate, opt-in method).
   */
  async addAssociationsBatched(
    associations: AssocOp[],
    options: { chunk_size?: number; max_chunk_bytes?: number } = {},
  ): Promise<BatchApplyResult> {
    const chunkSize = options.chunk_size ?? MAX_OPS_PER_REQUEST;
    const maxChunkBytes = options.max_chunk_bytes ?? MAX_CHUNK_BYTES;
    let applied = 0;
    let chunks = 0;
    for (const chunk of chunkAssociations(associations, chunkSize, maxChunkBytes)) {
      applied += await this.addAssociations(chunk);
      chunks += 1;
    }
    return { applied, chunks };
  }

  /**
   * Withdraw one (subject, label, object) association outright. Every
   * source's contribution to that one edge goes — where `retractSource`
   * withdraws a whole document's. Names resolve through aliases;
   * `retracted: false` means the triple named no live edge and nothing
   * changed. The surgical correction for a fact that should never have been
   * asserted; a fact that is merely CONTESTED wants a negative-weight
   * assertion instead.
   */
  async retractAssociation(
    subject: string,
    label: string,
    object: string,
  ): Promise<RetractAssociationOutcome> {
    const result = await this.post("/associations/retract", { subject, label, object });
    return result as RetractAssociationOutcome;
  }

  // -- passages / sources ----------------------------------------------------

  /**
   * Register source-id → full-text passages (replaces per source). Store the
   * document as-is: the server splits paragraphs on blank lines.
   * `questions`/`sections` attach per-paragraph doc2query questions and
   * section labels.
   */
  async storePassages(
    passages: Record<string, string>,
    options: {
      questions?: Record<string, QuestionSpec[]>;
      sections?: Record<string, SectionSpec[]>;
    } = {},
  ): Promise<StoredPassages> {
    const body: Record<string, unknown> = { passages };
    if (options.questions !== undefined) {
      body["questions"] = options.questions;
    }
    if (options.sections !== undefined) {
      body["sections"] = options.sections;
    }
    const result = await this.post("/sources", body);
    return result as StoredPassages;
  }

  /** Fetch whole passages by source id. */
  async lookupPassages(sources: string[]): Promise<PassageLookup> {
    const result = await this.post("/sources/lookup", { sources });
    return result as PassageLookup;
  }

  /**
   * Paragraph search (BM25 fused with embeddings where configured). Phrase
   * the query as an answer, not a question — a plausible declarative sentence
   * lands nearer the text you hope to find.
   */
  async searchPassages(query: string, options: { limit?: number } = {}): Promise<PassageHit[]> {
    const result = await this.post(
      "/sources/search",
      dropUndefined({ query, limit: options.limit }),
    );
    return result as PassageHit[];
  }

  /** Withdraw one source's contributions (diff sync before re-ingest). */
  async retractSource(source: string): Promise<RetractOutcome> {
    const result = await this.post("/sources/retract", { source });
    return result as RetractOutcome;
  }

  async listSources(options: { limit?: number; after?: string } = {}): Promise<SourcePage> {
    const result = await this.client.requestJson("GET", `${this.path}/sources`, {
      params: { limit: options.limit, after: options.after },
    });
    return result as SourcePage;
  }

  async *iterSources(options: { limit?: number } = {}): AsyncGenerator<string, void, undefined> {
    let after: string | undefined;
    for (;;) {
      const page = await this.listSources({ limit: options.limit, after });
      if (page.sources.length === 0) {
        return;
      }
      yield* page.sources;
      if (options.limit !== undefined && page.sources.length < options.limit) {
        return;
      }
      after = page.sources[page.sources.length - 1]!;
    }
  }

  /** One verbatim paragraph by source and paragraph locator. */
  async citePassage(source: string, paragraph: number): Promise<Citation> {
    const result = await this.post("/citations", { source, paragraph });
    return result as Citation;
  }

  // -- aliases -----------------------------------------------------------------

  /**
   * One alias page; the cursor spans both namespaces (concepts first), so
   * `after` takes "concept:<alias>" or "label:<alias>".
   */
  async getAliases(options: { limit?: number; after?: string } = {}): Promise<AliasPage> {
    const result = await this.client.requestJson("GET", `${this.path}/aliases`, {
      params: { limit: options.limit, after: options.after },
    });
    return result as AliasPage;
  }

  /** Walk both alias namespaces as a flat stream of entries. */
  async *iterAliases(options: { limit?: number } = {}): AsyncGenerator<AliasEntry, void, undefined> {
    let after: string | undefined;
    for (;;) {
      const page = await this.getAliases({ limit: options.limit, after });
      const count = Object.keys(page.concepts).length + Object.keys(page.labels).length;
      if (count === 0) {
        return;
      }
      let last = after;
      for (const [alias, canonical] of Object.entries(page.concepts)) {
        yield { namespace: "concept", alias, canonical };
        last = `concept:${alias}`;
      }
      for (const [alias, canonical] of Object.entries(page.labels)) {
        yield { namespace: "label", alias, canonical };
        last = `label:${alias}`;
      }
      if (options.limit !== undefined && count < options.limit) {
        return;
      }
      after = last;
    }
  }

  /**
   * Register alias → canonical spellings; returns the applied count. Aliases
   * are entry-only: results always carry the canonical spelling.
   * Re-registering an identical pair succeeds as a no-op (verified against
   * the server), so this call is retry-safe.
   */
  async addAliases(
    options: { concepts?: Record<string, string>; labels?: Record<string, string> } = {},
  ): Promise<number> {
    const result = await this.post("/aliases", {
      concepts: options.concepts ?? {},
      labels: options.labels ?? {},
    });
    return Number(result);
  }

  /** Withdraw alias spellings (canonical names are refused). */
  async removeAliases(
    options: { concepts?: string[]; labels?: string[] } = {},
  ): Promise<number> {
    const result = await this.client.requestJson("DELETE", `${this.path}/aliases`, {
      jsonBody: { concepts: options.concepts ?? [], labels: options.labels ?? [] },
    });
    return Number(result);
  }

  // -- maintenance ---------------------------------------------------------------

  /** Spelling/synonym fork candidates — candidates, not verdicts. */
  async auditVocabulary(
    options: { dice_floor?: number; cosine_floor?: number } = {},
  ): Promise<VocabularyAudit> {
    const result = await this.post(
      "/vocabulary/audit",
      dropUndefined({ dice_floor: options.dice_floor, cosine_floor: options.cosine_floor }),
    );
    return result as VocabularyAudit;
  }

  /**
   * Re-embed new/changed glosses (diff-only, idempotent). Throws
   * EmbeddingUnavailableError (501) when the server has no provider
   * configured.
   */
  async refreshEmbeddings(): Promise<RefreshOutcome> {
    const result = await this.post("/embeddings/refresh");
    return result as RefreshOutcome;
  }

  /** Rebuild the image without dead records (admin role). */
  async compact(): Promise<CompactOutcome> {
    const result = await this.post("/compact");
    return result as CompactOutcome;
  }

  // -- export ------------------------------------------------------------------------

  /** The context as an import batch stream (NDJSON text). */
  async export(): Promise<string> {
    return (await this.client.send("GET", `${this.path}/export`)).text;
  }

  /** Stream the export body without buffering it whole (no retry). */
  async *exportStream(): AsyncGenerator<Uint8Array, void, undefined> {
    const { url, headers, fetchImpl } = this.client.streamUrl(`${this.path}/export`);
    const response = await fetchImpl(url, { method: "GET", headers });
    if (response.status >= 400) {
      throw errorFromBody(
        response.status,
        response.headers.get("retry-after"),
        await response.text(),
      );
    }
    if (response.body === null) {
      return;
    }
    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          return;
        }
        yield value;
      }
    } finally {
      reader.releaseLock();
    }
  }

  /** Stream the export straight to a file. */
  async exportToFile(path: string): Promise<void> {
    const { open } = await import("node:fs/promises");
    const handle = await open(path, "w");
    try {
      for await (const chunk of this.exportStream()) {
        await handle.write(chunk);
      }
    } finally {
      await handle.close();
    }
  }

  // -- high-level retrieval loop -------------------------------------------------------

  /**
   * The documented retrieval loop as one call: resolve each cue →
   * (describe) → activate (and `query` when `labels` pins the facets) →
   * batch citations for every located attribution → optional text-lane
   * fallback.
   *
   * The cues must already be extracted entity names — decomposing a question,
   * judging lookalikes, and phrasing a declarative `text_fallback_query` are
   * the calling LLM's job. Every resolve candidate (gloss included) is
   * returned so an auto-picked anchor is never hidden.
   */
  async retrieve(
    origins: string | string[],
    options: {
      labels?: OneOrMany;
      dice_floor?: number;
      semantic_floor?: number;
      resolve_limit?: number;
      auto_pick?: boolean;
      activate_decay?: number;
      activate_limit?: number;
      describe_first?: boolean;
      fetch_citations?: boolean;
      text_fallback_query?: string;
      text_fallback_only_if_empty?: boolean;
      search_limit?: number;
    } = {},
  ): Promise<RetrievalResult> {
    const cues = typeof origins === "string" ? [origins] : [...origins];
    const autoPick = options.auto_pick ?? true;
    const describeFirst = options.describe_first ?? true;
    const fetchCitations = options.fetch_citations ?? true;
    const onlyIfEmpty = options.text_fallback_only_if_empty ?? true;

    const resolved: Record<string, TieredResolution[]> = {};
    const anchors: string[] = [];
    for (const cue of cues) {
      const candidates = await this.resolve(cue, {
        dice_floor: options.dice_floor,
        semantic_floor: options.semantic_floor,
        limit: options.resolve_limit,
      });
      resolved[cue] = candidates;
      const picked = autoPick ? candidates[0]?.name : cue;
      if (picked !== undefined && !anchors.includes(picked)) {
        anchors.push(picked);
      }
    }

    const outline: Record<string, ConceptDescription | null> = {};
    if (describeFirst) {
      for (const anchor of anchors) {
        outline[anchor] = await this.describe(anchor);
      }
    }

    let activations: Activation[] = [];
    const associations: Association[] = [];
    const seenTriples = new Set<string>();
    const tripleKey = (a: Association) => `${a.subject}\u0000${a.label}\u0000${a.object}`;
    if (anchors.length > 0) {
      if (options.labels !== undefined) {
        const matched = await this.query({ subject: anchors, label: options.labels });
        for (const match of matched.matches) {
          const key = tripleKey(match);
          if (!seenTriples.has(key)) {
            seenTriples.add(key);
            associations.push(match);
          }
        }
      }
      const page = await this.activate(anchors, {
        decay: options.activate_decay,
        limit: options.activate_limit,
      });
      activations = page.matches;
      for (const activation of activations) {
        const key = tripleKey(activation.association);
        if (!seenTriples.has(key)) {
          seenTriples.add(key);
          associations.push(activation.association);
        }
      }
    }

    const citations = new Map<string, Citation>();
    if (fetchCitations) {
      const wanted: Array<[string, number]> = [];
      const wantedKeys = new Set<string>();
      for (const association of associations) {
        for (const attribution of association.attributions) {
          if (attribution.paragraph === null) {
            continue;
          }
          const key = citationKey(attribution.source, attribution.paragraph);
          if (!wantedKeys.has(key)) {
            wantedKeys.add(key);
            wanted.push([attribution.source, attribution.paragraph]);
          }
        }
      }
      for (const [source, paragraph] of wanted) {
        try {
          citations.set(citationKey(source, paragraph), await this.citePassage(source, paragraph));
        } catch (error) {
          // The locator points at a passage that was never stored (or was
          // retracted) — the graph fact itself still stands.
          if (!(error instanceof NotFoundError)) {
            throw error;
          }
        }
      }
    }

    let passage_hits: PassageHit[] = [];
    if (
      options.text_fallback_query !== undefined &&
      (!onlyIfEmpty || associations.length === 0)
    ) {
      passage_hits = await this.searchPassages(options.text_fallback_query, {
        limit: options.search_limit,
      });
    }

    return { resolved, outline, associations, activations, citations, passage_hits };
  }
}
