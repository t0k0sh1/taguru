/** Transport-independent pieces: envelope handling, error mapping, chunking. */

import { TaguruError, errorForStatus } from "./errors.js";
import type {
  AssocOp,
  GroupImportOutcome,
  ImportOutcome,
  ImportResult,
  Issue,
  SchemaImportOutcome,
} from "./models.js";
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

/**
 * Lower-case every header key. HTTP header names are case-insensitive, but a
 * plain object used as a fetch() `headers` init is not: `Authorization` and
 * `authorization` survive as two distinct keys, and fetch's Headers-fill
 * algorithm appends both under one case-insensitive name instead of either
 * overwriting the other — comma-joining the two values into one broken
 * header. Normalizing here keeps caller-supplied headers from colliding with
 * the SDK's own lower-case keys (authorization, content-type).
 */
export function normalizeHeaders(headers: Record<string, string>): Record<string, string> {
  const normalized: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    normalized[key.toLowerCase()] = value;
  }
  return normalized;
}

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
  // `<` on strings compares UTF-16 code units, which diverges from UTF-8
  // byte order for supplementary-plane keys (an emoji sorts before
  // U+E000–U+FFFF in UTF-16, after in UTF-8) — enough to desync a
  // pagination cursor computed from this order.
  return Object.entries(mapping).sort(([a], [b]) =>
    Buffer.compare(Buffer.from(a, "utf-8"), Buffer.from(b, "utf-8")),
  );
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
  return unwrapEnvelopeFull(status, bodyText).result;
}

/**
 * `unwrapEnvelope` plus the envelope's warn-mode carrier: `issues` and
 * `schema_violations` are ADR 0009 §8.3's fields, riding *beside* `result`
 * on a `warn`-mode write whose associations violated the context's schema.
 * Both are empty/zero on every other response (and on servers predating the
 * fields), so result-only callers go through `unwrapEnvelope`.
 */
export function unwrapEnvelopeFull(
  status: number,
  bodyText: string,
): { result: unknown; issues: Issue[]; schema_violations: number } {
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
    const shaped = parsed as {
      result: unknown;
      status?: unknown;
      issues?: unknown;
      schema_violations?: unknown;
    };
    if (shaped.status === "ok") {
      return {
        result: shaped.result,
        issues: Array.isArray(shaped.issues) ? (shaped.issues as Issue[]) : [],
        schema_violations:
          typeof shaped.schema_violations === "number" ? shaped.schema_violations : 0,
      };
    }
  }
  throw new TaguruError("response is not the taguru envelope shape", {
    status,
    body: parsed,
  });
}

/**
 * Normalize /import's response to an `ImportResult`. Current servers always
 * answer `{batches: [...], groups: [...], schemas: [...]}`
 * (`groups`/`schemas` omitted entirely when the stream carried none);
 * servers predating that change answered a bare outcome for a single batch
 * — both parse here, so callers never branch on response shape.
 * `issues`/`schema_violations` are the response envelope's warn-mode
 * carrier, passed through by `importBatches`.
 */
export function normalizeImportOutcomes(
  result: unknown,
  issues: Issue[] = [],
  schema_violations = 0,
): ImportResult {
  if (
    typeof result === "object" &&
    result !== null &&
    Array.isArray((result as { batches?: unknown }).batches)
  ) {
    const shaped = result as {
      batches: ImportOutcome[];
      groups?: GroupImportOutcome[];
      schemas?: SchemaImportOutcome[];
    };
    return {
      batches: shaped.batches,
      groups: shaped.groups ?? [],
      schemas: shaped.schemas ?? [],
      issues,
      schema_violations,
    };
  }
  return {
    batches: [result as ImportOutcome],
    groups: [],
    schemas: [],
    issues,
    schema_violations,
  };
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
  if (error instanceof AggregateError) {
    // AggregateError's own `.message` defaults to "" — the real detail
    // lives in `.errors`, e.g. one entry per address a dual-stack connect
    // attempt failed to reach.
    return error.errors.map(describeError).join("; ") || error.message || error.name;
  }
  if (error instanceof Error) {
    const cause = (error as { cause?: unknown }).cause;
    if (cause instanceof AggregateError) {
      return `${error.message}: ${describeError(cause)}`;
    }
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
