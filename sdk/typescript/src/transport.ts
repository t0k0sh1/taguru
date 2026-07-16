/** Transport-independent pieces: envelope handling, error mapping, chunking. */

import { TaguruError, errorForStatus } from "./errors.js";
import type { AssocOp, GroupImportOutcome, ImportOutcome, ImportResult } from "./models.js";
import { parseRetryAfter } from "./retry.js";

export const DEFAULT_BASE_URL = "http://127.0.0.1:8248";
export const ENV_URL = "TAGURU_URL";
export const ENV_TOKEN = "TAGURU_API_TOKEN";
/**
 * Matches the server's own TAGURU_REQUEST_TIMEOUT_SECS default; raise both
 * together when the server has an embedding provider configured.
 */
export const DEFAULT_TIMEOUT_SECS = 30.0;
/** Server-enforced caps mirrored client-side by addAssociationsBatched. */
export const MAX_OPS_PER_REQUEST = 10_000;
export const MAX_CHUNK_BYTES = 8 * 1024 * 1024;

/** Omit absent optional fields instead of sending nulls. */
export function dropUndefined(mapping: Record<string, unknown>): Record<string, unknown> {
  const kept: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(mapping)) {
    if (value !== undefined && value !== null) {
      kept[key] = value;
    }
  }
  return kept;
}

/**
 * `Object.entries`, ordered by key byte order rather than `Object.entries`'
 * own insertion/numeric-first order (which sorts integer-like keys like
 * "2" and "10" numerically, ahead of any string key). Needed wherever a
 * map mirrors a server-side `BTreeMap<String, String>` whose byte order a
 * pagination cursor depends on.
 */
export function sortedEntries(mapping: Record<string, string>): Array<[string, string]> {
  return Object.entries(mapping).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

/** Split a batch by both element count and serialized body size. */
export function* chunkAssociations(
  ops: AssocOp[],
  chunkSize: number,
  maxChunkBytes: number,
): Generator<AssocOp[], void, undefined> {
  let chunk: AssocOp[] = [];
  let chunkBytes = 2; // "[" + "]"
  for (const op of ops) {
    const opBytes = Buffer.byteLength(JSON.stringify(op), "utf-8");
    let added = opBytes + (chunk.length > 0 ? 1 : 0); // separating comma
    if (chunk.length > 0 && (chunk.length >= chunkSize || chunkBytes + added > maxChunkBytes)) {
      yield chunk;
      chunk = [];
      chunkBytes = 2;
      added = opBytes;
    }
    chunk.push(op);
    chunkBytes += added;
  }
  if (chunk.length > 0) {
    yield chunk;
  }
}

/** Build the mapped error for a non-2xx response body. */
export function errorFromBody(
  status: number,
  retryAfterHeader: string | null,
  bodyText: string,
): TaguruError {
  const retry_after = parseRetryAfter(retryAfterHeader);
  let parsed: unknown;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    const message = bodyText.trim() || `HTTP ${status}`;
    return errorForStatus(status, message, { body: bodyText, retry_after });
  }
  let message = `HTTP ${status}`;
  let code: string | null = null;
  let time: number | null = null;
  if (typeof parsed === "object" && parsed !== null) {
    const shaped = parsed as { error?: unknown; code?: unknown; time?: unknown };
    if (typeof shaped.error === "string") {
      message = shaped.error;
    }
    if (typeof shaped.code === "string") {
      code = shaped.code;
    }
    if (typeof shaped.time === "number") {
      time = shaped.time;
    }
  }
  return errorForStatus(status, message, { body: parsed, code, time, retry_after });
}

/** Extract `result` from the `{"result", "status": "ok", "time"}` envelope. */
export function unwrapEnvelope(status: number, bodyText: string): unknown {
  let parsed: unknown;
  try {
    parsed = JSON.parse(bodyText);
  } catch (cause) {
    throw new TaguruError("expected a JSON envelope, got a non-JSON body", {
      status,
      body: bodyText,
      cause,
    });
  }
  if (typeof parsed === "object" && parsed !== null && "result" in parsed) {
    const shaped = parsed as { result: unknown; status?: unknown };
    if (shaped.status === "ok") {
      return shaped.result;
    }
  }
  throw new TaguruError("response is not the taguru envelope shape", {
    status,
    body: parsed,
  });
}

/**
 * Normalize /import's response to `{batches, groups}`. Current servers
 * always answer `{batches: [...], groups: [...]}` (`groups` omitted
 * entirely when the stream carried none); servers predating that change
 * answered a bare outcome for a single batch — both parse here, so callers
 * never branch on response shape.
 */
export function normalizeImportOutcomes(result: unknown): ImportResult {
  if (
    typeof result === "object" &&
    result !== null &&
    Array.isArray((result as { batches?: unknown }).batches)
  ) {
    const shaped = result as { batches: ImportOutcome[]; groups?: GroupImportOutcome[] };
    return { batches: shaped.batches, groups: shaped.groups ?? [] };
  }
  return { batches: [result as ImportOutcome], groups: [] };
}

/** Percent-encode one path segment (context names may be any UTF-8). */
export function encodeName(name: string): string {
  return encodeURIComponent(name);
}

/**
 * Whether a fetch failure certainly happened before the request was sent
 * (refused connection, unresolvable host, connect-phase timeout) — always
 * safe to retry. Anything else is ambiguous: the request may have executed
 * server-side.
 *
 * UND_ERR_CONNECT_TIMEOUT is undici's own connect-phase timeout, distinct
 * from the AbortSignal.timeout() `send` races against the whole request: that
 * one surfaces as an unqualified "TimeoutError" with no `code` at all and can
 * fire after the request already reached the server, so it stays ambiguous.
 * The undici error fires only while the TCP handshake itself is still
 * outstanding, which is why it belongs in this set and TimeoutError does not.
 */
export function isPreConnectFailure(error: unknown): boolean {
  const codes = new Set(["ECONNREFUSED", "ENOTFOUND", "EAI_AGAIN", "UND_ERR_CONNECT_TIMEOUT"]);
  const codeOf = (value: unknown): string | undefined => {
    if (typeof value === "object" && value !== null && "code" in value) {
      const code = (value as { code?: unknown }).code;
      return typeof code === "string" ? code : undefined;
    }
    return undefined;
  };
  const seen = new Set<unknown>();
  let current: unknown = error;
  while (current !== undefined && current !== null && !seen.has(current)) {
    seen.add(current);
    const code = codeOf(current);
    if (code !== undefined && codes.has(code)) {
      return true;
    }
    if (current instanceof AggregateError) {
      return current.errors.some((inner) => {
        const innerCode = codeOf(inner);
        return innerCode !== undefined && codes.has(innerCode);
      });
    }
    current = (current as { cause?: unknown }).cause;
  }
  return false;
}

export function describeError(error: unknown): string {
  if (error instanceof Error) {
    const cause = (error as { cause?: unknown }).cause;
    if (cause instanceof Error && cause.message) {
      return `${error.message}: ${cause.message}`;
    }
    return error.message || error.name;
  }
  return String(error);
}

export function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}
