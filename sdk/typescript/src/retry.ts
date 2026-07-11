/**
 * Retry policy: idempotency-first. Mirrors the Python SDK's `_retry.py`.
 *
 * Most of taguru's POST routes are reads-with-a-body or replace/diff writes
 * and are safe to retry blindly. The exceptions carry "unsafe_on_ambiguous":
 * a request that may already have executed (the response was lost mid-flight)
 * is NOT retried there, because re-applying would double state —
 * `addAssociations` accumulates weight. Statuses where the server states the
 * request was never executed (429 rate limit, 503 shed — both rejected before
 * the handler ran) retry regardless of class.
 */

export type RetryClass = "safe" | "unsafe_on_ambiguous";

export const DEFAULT_RETRIES = 2;
export const BACKOFF_BASE_SECS = 0.5;
export const BACKOFF_CAP_SECS = 8.0;

/**
 * Whether a non-2xx status is retryable for this route class.
 *
 * 429/503 mean the request was shed before executing — always retryable.
 * 502 is a transient embedding-provider failure ("retry later" per the
 * protocol) — it only occurs on read-shaped routes, all of which are safe.
 * 501 is static configuration; retrying cannot help.
 */
export function shouldRetryStatus(status: number, retryClass: RetryClass): boolean {
  if (status === 429 || status === 503) {
    return true;
  }
  if (status === 502) {
    return retryClass === "safe";
  }
  return false;
}

/**
 * Whether a transport failure is retryable. `ambiguous` means the request
 * may have reached the server (failure after the connection was established);
 * a pre-connect failure is always safe.
 */
export function shouldRetryTransport(ambiguous: boolean, retryClass: RetryClass): boolean {
  return !ambiguous || retryClass === "safe";
}

/** Full-jitter exponential backoff (seconds) for the given 0-based attempt. */
export function backoffDelay(attempt: number): number {
  return Math.random() * Math.min(BACKOFF_CAP_SECS, BACKOFF_BASE_SECS * 2 ** attempt);
}

/** Parse a Retry-After header. The server sends delay-seconds only. */
export function parseRetryAfter(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const seconds = Number.parseFloat(value.trim());
  if (Number.isNaN(seconds) || seconds < 0) {
    return null;
  }
  return seconds;
}
