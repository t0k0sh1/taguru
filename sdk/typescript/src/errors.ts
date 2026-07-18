/**
 * Error hierarchy raised by the Taguru client.
 *
 * Every non-2xx HTTP response maps to a {@link TaguruError} subclass chosen by
 * HTTP status. The server's error body
 * (`{"status": "error", "code": "...", "error": "...", "time": ...}`) also
 * carries a stable machine-readable `code` (e.g. `"no_context"`,
 * `"over_limit"` — `GET /protocol` lists the vocabulary), surfaced here as
 * `.code` for branching finer than the class hierarchy; servers predating the
 * field yield `null`. {@link TransportError} covers failures where no HTTP
 * response was obtained at all. Class names match the Python SDK exactly.
 */

export interface TaguruErrorOptions {
  status?: number | null;
  code?: string | null;
  time?: number | null;
  body?: unknown;
  cause?: unknown;
}

export class TaguruError extends Error {
  /** HTTP status code, or null when no response was obtained. */
  readonly status: number | null;
  /**
   * The server's stable machine-readable failure kind (e.g.
   * "no_context"), or null when the response carried none.
   */
  readonly code: string | null;
  /** The server-reported handling time in seconds, when available. */
  readonly time: number | null;
  /** The parsed error body (object) or raw text, when available. */
  readonly body: unknown;

  constructor(message: string, options: TaguruErrorOptions = {}) {
    super(message, options.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = new.target.name;
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.time = options.time ?? null;
    this.body = options.body;
  }
}

/** 401 — missing or wrong bearer token. */
export class AuthenticationError extends TaguruError {}

/**
 * 403 — the key's role or context scope does not cover this operation.
 * Also every write sent to a read replica: `code` is then
 * `"read_only_replica"` and the message names the writer to send writes
 * to. Deliberate refusals both — never retried.
 */
export class PermissionDeniedError extends TaguruError {}

/** 404 — unknown context, source, paragraph, or route. */
export class NotFoundError extends TaguruError {}

/** 409 — duplicate create, alias conflict, or partial-write conflict. */
export class ConflictError extends TaguruError {}

/** 400/415/422 — malformed body, wrong content type, or mistyped JSON. */
export class ValidationError extends TaguruError {}

/**
 * 413 — request body over the server's cap. Current servers answer it in
 * the JSON error shape; servers predating that change answered axum's
 * plain text — both parse here.
 */
export class PayloadTooLargeError extends TaguruError {}

/** 408 — the server's own per-request budget expired. */
export class RequestTimeoutError extends TaguruError {}

/** 429 — this key is over its request budget; wait `retry_after` seconds. */
export class RateLimitError extends TaguruError {
  readonly retry_after: number | null;

  constructor(message: string, options: TaguruErrorOptions & { retry_after?: number | null } = {}) {
    super(message, options);
    this.retry_after = options.retry_after ?? null;
  }
}

/** 500 and any unmapped 5xx. */
export class ServerError extends TaguruError {}

/** 503 — load shedding or a degraded write path; honors `retry_after`. */
export class ServiceUnavailableError extends ServerError {
  readonly retry_after: number | null;

  constructor(message: string, options: TaguruErrorOptions & { retry_after?: number | null } = {}) {
    super(message, options);
    this.retry_after = options.retry_after ?? null;
  }
}

/** 507 — the write hit a storage cap and was NOT applied. */
export class StorageFullError extends ServerError {}

/**
 * 501/502 — no embedding provider configured, or the provider failed.
 * `reason` is "not_configured" for 501 and "provider_error" for 502.
 */
export class EmbeddingUnavailableError extends ServerError {
  readonly reason: "not_configured" | "provider_error";

  constructor(message: string, options: TaguruErrorOptions = {}) {
    super(message, options);
    this.reason = options.status === 501 ? "not_configured" : "provider_error";
  }
}

/** No HTTP response was obtained (DNS, refused connection, timeout, ...). */
export class TransportError extends TaguruError {}

/** Any status with no specific mapping (e.g. 405) — exhaustiveness fallback. */
export class UnexpectedStatusError extends TaguruError {}

/** Build the error for an HTTP status, endpoint-independently. */
export function errorForStatus(
  status: number,
  message: string,
  options: {
    code?: string | null;
    time?: number | null;
    body?: unknown;
    retry_after?: number | null;
  } = {},
): TaguruError {
  const base = { status, code: options.code, time: options.time, body: options.body };
  switch (status) {
    case 401:
      return new AuthenticationError(message, base);
    case 403:
      return new PermissionDeniedError(message, base);
    case 404:
      return new NotFoundError(message, base);
    case 409:
      return new ConflictError(message, base);
    case 400:
    case 415:
    case 422:
      return new ValidationError(message, base);
    case 413:
      return new PayloadTooLargeError(message, base);
    case 408:
      return new RequestTimeoutError(message, base);
    case 429:
      return new RateLimitError(message, { ...base, retry_after: options.retry_after });
    case 503:
      return new ServiceUnavailableError(message, { ...base, retry_after: options.retry_after });
    case 507:
      return new StorageFullError(message, base);
    case 501:
    case 502:
      return new EmbeddingUnavailableError(message, base);
    default:
      if (status >= 500 && status < 600) {
        return new ServerError(message, base);
      }
      return new UnexpectedStatusError(message, base);
  }
}
