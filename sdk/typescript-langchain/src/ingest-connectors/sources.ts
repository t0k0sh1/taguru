/**
 * Source id derivation and idempotency (ADR 0007 §6.1, issue #347) — the
 * mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.sources` module (issue #415).
 *
 * Extends the source id convention `taguru extract` already established
 * (`path.to_string_lossy()`, src/extract.rs:481) and docs/long-running.html
 * already documented (`manual.pdf#installation`) to URL and object-storage
 * connectors, without inventing a new grammar.
 */

import { byteLen, Diagnostic, MAX_NAME_BYTES } from "./document.js";

// A fixed, documented deny-list of well-known signed/temporary auth query
// parameters, matched case-insensitively on the key only — a URL connector
// strips these for two reasons: they are credential-shaped (ADR 0007 §9's
// "credential never reaches Taguru data" rule applies identically to a
// source id), and a rotating signature would otherwise make "the same
// resource" mint a different source id on every fetch, breaking §6.3's
// idempotency the same way any other unstable id breaks it.
const DENYLISTED_QUERY_KEYS: ReadonlySet<string> = new Set([
  "signature",
  "sig",
  "token",
  "access_token",
  "x-amz-signature",
  "x-amz-credential",
  "x-amz-security-token",
  "apikey",
  "api_key",
]);

/**
 * The local-file source id — unchanged from `taguru extract`'s own
 * `path.to_string_lossy()` (src/extract.rs:481): the path string,
 * verbatim.
 */
export function fileSourceId(path: string): string {
  return path;
}

/**
 * A sub-document unit within `source` — `manual.pdf#p12` or
 * `manual.pdf#installation` (already documented in docs/long-running.html),
 * extended by ADR 0007 §6.1 to every connector kind:
 * `s3://bucket/key#p12`, `https://example.com/report.html#p3`.
 */
export function subSourceId(source: string, fragment: string): string {
  return `${source}#${fragment}`;
}

/**
 * `null` when `source` fits within `MAX_NAME_BYTES`; otherwise the
 * `source_id_too_long` diagnostic a connector must refuse the object with
 * — NEVER a silently truncated id, since truncation risks two distinct
 * objects colliding on one source id (ADR 0007 §6.1).
 */
export function checkSourceId(source: string): Diagnostic | null {
  if (byteLen(source) <= MAX_NAME_BYTES) {
    return null;
  }
  return new Diagnostic({
    code: "source_id_too_long",
    message: `source id exceeds ${MAX_NAME_BYTES} bytes`,
    source,
  });
}

/**
 * Splits a URL into the five `urlsplit` components without any WHATWG
 * normalization — the WHATWG `URL` class lowercases the scheme and host,
 * punycodes, and resolves paths, all of which the Python twin's
 * `urllib.parse.urlsplit` deliberately does not do (`canonicalizeUrl`'s
 * doc: "case preserved — canonicalization here is about credentials and
 * stability, not host normalization").
 */
function urlSplit(url: string): {
  scheme: string;
  netloc: string;
  path: string;
  query: string;
  fragment: string;
} {
  let rest = url;
  let fragment = "";
  const hashIndex = rest.indexOf("#");
  if (hashIndex >= 0) {
    fragment = rest.slice(hashIndex + 1);
    rest = rest.slice(0, hashIndex);
  }
  let scheme = "";
  const schemeMatch = /^([A-Za-z][A-Za-z0-9+.-]*):/.exec(rest);
  if (schemeMatch !== null) {
    scheme = schemeMatch[1]!;
    rest = rest.slice(schemeMatch[0].length);
  }
  let netloc = "";
  if (rest.startsWith("//")) {
    rest = rest.slice(2);
    const netlocEnd = [rest.indexOf("/"), rest.indexOf("?")]
      .filter((index) => index >= 0)
      .reduce((a, b) => Math.min(a, b), rest.length);
    netloc = rest.slice(0, netlocEnd);
    rest = rest.slice(netlocEnd);
  }
  let query = "";
  const queryIndex = rest.indexOf("?");
  let path = rest;
  if (queryIndex >= 0) {
    query = rest.slice(queryIndex + 1);
    path = rest.slice(0, queryIndex);
  }
  return { scheme, netloc, path, query, fragment };
}

/** `unquote_plus` mirror: `+` to space, tolerant percent-decoding (an
 * invalid sequence stays verbatim rather than throwing). */
function unquotePlus(value: string): string {
  const plusDecoded = value.replaceAll("+", " ");
  return plusDecoded.replace(/(%[0-9A-Fa-f]{2})+/g, (match) => {
    try {
      return decodeURIComponent(match);
    } catch {
      return match;
    }
  });
}

/** `quote_plus` mirror: everything outside `[A-Za-z0-9_.~-]` is
 * percent-encoded from its UTF-8 bytes, except space which becomes `+`. */
function quotePlus(value: string): string {
  let result = "";
  for (const char of value) {
    if (/^[A-Za-z0-9_.~-]$/.test(char)) {
      result += char;
    } else if (char === " ") {
      result += "+";
    } else {
      for (const byte of new TextEncoder().encode(char)) {
        result += `%${byte.toString(16).toUpperCase().padStart(2, "0")}`;
      }
    }
  }
  return result;
}

/** `parse_qsl(..., keep_blank_values=True)` mirror: split on `&`, keep
 * blank values, drop fully-empty chunks. */
function parseQsl(query: string): Array<[string, string]> {
  const pairs: Array<[string, string]> = [];
  for (const chunk of query.split("&")) {
    if (!chunk) {
      continue;
    }
    const eqIndex = chunk.indexOf("=");
    if (eqIndex < 0) {
      pairs.push([unquotePlus(chunk), ""]);
    } else {
      pairs.push([unquotePlus(chunk.slice(0, eqIndex)), unquotePlus(chunk.slice(eqIndex + 1))]);
    }
  }
  return pairs;
}

/**
 * The one canonical, credential-stripped form of `url` — used for
 * identity, storage, AND display alike (ADR 0007 §6.1). There is no
 * separate "redacted display value": canonicalization already produces
 * one value safe for every purpose, so any URL appearing in a log line or
 * the observability summary uses this same form.
 *
 * - `userinfo` (`https://user:pass@host/...`) is always stripped — never
 *   meaningful to identity, always credential-shaped.
 * - The deny-listed query parameters (module-level
 *   `DENYLISTED_QUERY_KEYS`) are stripped; every other query parameter is
 *   kept, in its original order.
 * - Scheme, host (with port, and case preserved — canonicalization here
 *   is about credentials and stability, not host normalization), path,
 *   and fragment are otherwise untouched.
 *
 * The raw, uncanonicalized URL is the caller's to use for the fetch
 * itself; it is never returned here and must never be persisted.
 */
export function canonicalizeUrl(url: string): string {
  const parts = urlSplit(url);
  const atIndex = parts.netloc.lastIndexOf("@");
  const netloc = atIndex >= 0 ? parts.netloc.slice(atIndex + 1) : parts.netloc;
  const query = parseQsl(parts.query)
    .filter(([key]) => !DENYLISTED_QUERY_KEYS.has(key.toLowerCase()))
    .map(([key, value]) => `${quotePlus(key)}=${quotePlus(value)}`)
    .join("&");
  let result = "";
  if (parts.scheme) {
    result += `${parts.scheme}:`;
  }
  if (netloc || url.slice(result.length).startsWith("//")) {
    result += `//${netloc}`;
  }
  result += parts.path;
  if (query) {
    result += `?${query}`;
  }
  if (parts.fragment) {
    result += `#${parts.fragment}`;
  }
  return result;
}

/**
 * Detects "two distinct fetches canonicalize to the same source id" — a
 * rare collision (e.g. two otherwise-identical URLs differing only in a
 * stripped query parameter) this run must refuse the later one for,
 * rather than silently overwrite (ADR 0007 §6.1) — the same
 * collision-refusal `taguru extract`'s own `Run.claimed` map already
 * applies for batch file names (src/extract.rs:1273).
 *
 * Scoped to one run's lifetime; a connector constructs one instance per
 * enumeration pass. Deciding which diagnostic code names a refused
 * collision is left to the calling connector — ADR 0007 §8's closed
 * vocabulary has no code specific to "duplicate source id" today, so a
 * connector should pick the code that best matches why the object could
 * not be safely ingested under that identity.
 */
export class SourceIdRegistry {
  private readonly claimed = new Set<string>();

  /**
   * Returns `true` the first time `source` is claimed by this registry,
   * `false` on every subsequent claim of the same id.
   */
  claim(source: string): boolean {
    if (this.claimed.has(source)) {
      return false;
    }
    this.claimed.add(source);
    return true;
  }
}
