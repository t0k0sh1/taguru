/**
 * Wire-contract compatibility (ADR 0005 §3.8, §6).
 *
 * This SDK decodes HTTP; it owns no wire shape of its own (ADR 0005 §3). So
 * the only thing it ever compares against a server is `http_contract` —
 * never `server`'s own SemVer (a compatible patch/minor difference must
 * never be refused) and never `mcp_contract` (this SDK speaks no MCP; that
 * dimension covers tool schemas and JSON-RPC conventions it is
 * structurally blind to, so comparing it would only produce false
 * rejections).
 *
 * Fails closed only on POSITIVE proof the two ranges are disjoint. Every
 * absence of information — a 404 (pre-0.6 servers, which are
 * `http_contract: 1` in substance even though they predate this endpoint),
 * a non-JSON body, a missing `http_contract` key, an empty `supported`
 * array — is fail-open: none of those prove incompatibility, and refusing
 * on them would be a worse break than the one this check exists to
 * prevent.
 *
 * Deliberately NOT part of `sdk/spec/check_versions.py`'s lockstep: ADR
 * 0005 §3.8 makes the contract range independent of package version, on
 * purpose — routing it through that checker would silently re-couple them
 * and undo the point of the eight-dimension split.
 */

import { IncompatibleServerError } from "./errors.js";
import { VERSION } from "./version.js";

/**
 * The one `http_contract` version this SDK release decodes. An array of
 * accepted versions, not a `{min, max}` range: it mirrors `GET
 * /version`'s own `supported` array (ADR 0005 §6) so the check is a plain
 * set intersection, and it can express a future gap (e.g. `[1, 3]`) a
 * min/max pair cannot.
 */
export const SUPPORTED_HTTP_CONTRACTS: readonly number[] = [1];

/** ADR 0005 §6: exempt from auth like the other probes, always 200. */
export const VERSION_PATH = "/version";

/** The subset of `GET /version`'s body this SDK reads. */
export interface ServerContract {
  server: string | null;
  supported: readonly number[];
}

/**
 * Parse a `GET /version` body, or `null` if it can't be read.
 *
 * `null` covers every shape this SDK does not recognize (a pre-0.6
 * server's 404 body, a stray non-JSON response, a body missing
 * `http_contract` or carrying a non-array `supported`) — the caller
 * treats `null` as "learned nothing," which is fail-open by construction.
 */
export function parseVersionBody(payload: unknown): ServerContract | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }
  const httpContract = (payload as Record<string, unknown>).http_contract;
  if (typeof httpContract !== "object" || httpContract === null) {
    return null;
  }
  const supported = (httpContract as Record<string, unknown>).supported;
  if (!Array.isArray(supported) || !supported.every((item) => typeof item === "number")) {
    return null;
  }
  const server = (payload as Record<string, unknown>).server;
  return {
    server: typeof server === "string" ? server : null,
    supported: supported as number[],
  };
}

/**
 * The error to raise for `seen`, or `null` if it's compatible.
 *
 * Compatible means "shares at least one `http_contract` version with this
 * SDK" — an empty `seen.supported` is treated as no proof of anything
 * (fail-open), not as an empty intersection.
 */
export function incompatibility(
  seen: ServerContract,
  baseUrl: string,
  // Defaults to this SDK's real range; a test-only override — `const`
  // module bindings are immutable even from within their own module,
  // so unlike Python's `monkeypatch.setattr` there is no way to fake
  // `SUPPORTED_HTTP_CONTRACTS` short of this parameter.
  supported: readonly number[] = SUPPORTED_HTTP_CONTRACTS,
): IncompatibleServerError | null {
  if (seen.supported.length === 0) {
    return null;
  }
  const seenSet = new Set(seen.supported);
  if (supported.some((version) => seenSet.has(version))) {
    return null;
  }

  const sdkVersions = supported.join(", ");
  const serverVersions = seen.supported.join(", ");
  const serverNote = seen.server ? ` (taguru ${seen.server})` : "";

  let remedy: string;
  const seenMin = Math.min(...seen.supported);
  const seenMax = Math.max(...seen.supported);
  const sdkMin = Math.min(...supported);
  const sdkMax = Math.max(...supported);
  if (seenMin > sdkMax) {
    remedy = seen.server
      ? `Upgrade this SDK to a release that speaks http_contract ${serverVersions}: ` +
        `npm install taguru@^${seen.server}`
      : `Upgrade this SDK to a release that speaks http_contract ${serverVersions}.`;
  } else if (seen.server && seenMax < sdkMin) {
    const minor = seen.server.split(".").slice(0, 2).join(".");
    remedy =
      `Upgrade the server to a release that speaks http_contract ${sdkVersions}, or ` +
      `pin this SDK to the server's release: npm install taguru@${minor}.x`;
  } else {
    remedy =
      "Upgrade or downgrade one side to a pair that shares a contract version; this " +
      "SDK's range is declared as taguru's SUPPORTED_HTTP_CONTRACTS.";
  }

  const message =
    `taguru SDK ${VERSION} speaks http_contract ${sdkVersions}, but the server at ` +
    `${baseUrl}${serverNote} supports http_contract ${serverVersions} — no version ` +
    `in common. ${remedy}`;

  return new IncompatibleServerError(message, {
    sdk_version: VERSION,
    server_version: seen.server,
    supported_contracts: supported,
    server_contracts: seen.supported,
  });
}
