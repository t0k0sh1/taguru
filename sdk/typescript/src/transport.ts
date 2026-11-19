/** Transport-independent pieces: envelope handling, error mapping, batching. */

import { TaguruError, errorForStatus } from "./errors.js";
import type {
  AssocOp,
  GroupImportOutcome,
  ImportOutcome,
  ImportResult,
  Issue,
  SchemaImportOutcome,
  Unreached,
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
export const MAX_BATCH_BYTES = 8 * 1024 * 1024;

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

/** Split associations into batches by both element count and serialized body size. */
export function* batchAssociations(
  ops: AssocOp[],
  batchSize: number,
  maxBatchBytes: number,
): Generator<AssocOp[], void, undefined> {
  let batch: AssocOp[] = [];
  let batchBytes = 2; // "[" + "]"
  for (const op of ops) {
    const opBytes = Buffer.byteLength(JSON.stringify(op), "utf-8");
    let added = opBytes + (batch.length > 0 ? 1 : 0); // separating comma
    if (batch.length > 0 && (batch.length >= batchSize || batchBytes + added > maxBatchBytes)) {
      yield batch;
      batch = [];
      batchBytes = 2;
      added = opBytes;
    }
    batch.push(op);
    batchBytes += added;
  }
  if (batch.length > 0) {
    yield batch;
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
 * `unwrapEnvelope` plus the envelope's carriers beside `result`: `issues`
 * and `schema_violations` are ADR 0009 §8.3's warn-mode fields, riding on a
 * `warn`-mode write whose associations violated the `context`'s schema;
 * `unreached` is a `taguru router`'s partial-fan-out report — the shards a
 * cross-`context` search or the `/contexts` listing could not reach, so the
 * result is real but missing their rows. All three are empty/zero on every
 * other response (and on servers predating the fields), so result-only
 * callers go through `unwrapEnvelope`.
 */
export function unwrapEnvelopeFull(
  status: number,
  bodyText: string,
): { result: unknown; issues: Issue[]; schema_violations: number; unreached: Unreached[] } {
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
      unreached?: unknown;
    };
    if (shaped.status === "ok") {
      return {
        result: shaped.result,
        issues: Array.isArray(shaped.issues) ? (shaped.issues as Issue[]) : [],
        schema_violations:
          typeof shaped.schema_violations === "number" ? shaped.schema_violations : 0,
        unreached: Array.isArray(shaped.unreached) ? (shaped.unreached as Unreached[]) : [],
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
 * servers predating that change answered a bare outcome for a single source
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

/** Percent-encode one path segment (`context` names may be any UTF-8). */
export function encodeName(name: string): string {
  return encodeURIComponent(name);
}

/**
 * Error codes that only ever come from establishing a connection — a
 * refused connect, an unresolvable host, undici's own connect-phase timeout
 * — so their presence alone proves the request was never sent.
 *
 * UND_ERR_CONNECT_TIMEOUT is undici's own connect-phase timeout, distinct
 * from the AbortSignal.timeout() `send` races against the whole request: that
 * one surfaces as an unqualified "TimeoutError" with no `code` at all and can
 * fire after the request already reached the server, so it stays ambiguous.
 * The undici error fires only while the TCP handshake itself is still
 * outstanding, which is why it belongs in this set and TimeoutError does not.
 */
const CONNECT_ONLY_CODES = new Set([
  "ECONNREFUSED",
  "ENOTFOUND",
  "EAI_AGAIN",
  "UND_ERR_CONNECT_TIMEOUT",
]);

/**
 * Error codes the OS raises on a connect attempt AND on an established
 * socket — an unreachable host or network, a TCP timeout. Pre-connect only
 * when Node tagged the error `syscall: "connect"`: the same code from a
 * `read`/`write` on a live connection means the request may already have
 * reached the server.
 */
const CONNECT_PHASE_CODES = new Set(["EHOSTUNREACH", "ENETUNREACH", "ETIMEDOUT"]);

/** Whether one error object, by itself, proves a pre-connect failure. */
function provesPreConnect(value: unknown): boolean {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const { code, syscall } = value as { code?: unknown; syscall?: unknown };
  if (typeof code !== "string") {
    return false;
  }
  return CONNECT_ONLY_CODES.has(code) || (CONNECT_PHASE_CODES.has(code) && syscall === "connect");
}

/**
 * Whether a fetch failure certainly happened before the request was sent
 * (refused connection, unresolvable host, unreachable host or network,
 * connect-phase timeout) — always safe to retry. Anything else is
 * ambiguous: the request may have executed server-side.
 */
export function isPreConnectFailure(error: unknown): boolean {
  const seen = new Set<unknown>();
  let current: unknown = error;
  while (current !== undefined && current !== null && !seen.has(current)) {
    seen.add(current);
    if (provesPreConnect(current)) {
      return true;
    }
    if (current instanceof AggregateError) {
      // One entry per address a dual-stack connect attempt tried.
      return current.errors.some(provesPreConnect);
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
