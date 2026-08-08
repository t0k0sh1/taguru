/** Source id derivation and URL canonicalization (ADR 0007 §6.1, issue
 * #347; TypeScript parity: issue #415). */

import { expect, test } from "vitest";

import {
  SourceIdRegistry,
  canonicalizeUrl,
  checkSourceId,
  fileSourceId,
  subSourceId,
} from "../../src/ingest-connectors/sources.js";

test("fileSourceId is the path verbatim", () => {
  // Unchanged from taguru extract's own path.to_string_lossy() (src/extract.rs:481).
  expect(fileSourceId("docs/manual.pdf")).toBe("docs/manual.pdf");
});

test("subSourceId appends a fragment", () => {
  expect(subSourceId("manual.pdf", "p12")).toBe("manual.pdf#p12");
  expect(subSourceId("manual.pdf", "installation")).toBe("manual.pdf#installation");
});

test("checkSourceId is null within the cap", () => {
  expect(checkSourceId("docs/manual.pdf")).toBeNull();
  expect(checkSourceId("x".repeat(1024))).toBeNull();
});

test("checkSourceId flags an oversized source without truncating", () => {
  const source = "x".repeat(1025);
  const diagnostic = checkSourceId(source);
  expect(diagnostic).not.toBeNull();
  expect(diagnostic!.code).toBe("source_id_too_long");
  expect(diagnostic!.source).toBe(source); // never truncated — collision risk
});

test("canonicalizeUrl strips userinfo", () => {
  expect(canonicalizeUrl("https://user:pass@example.com/report.html")).toBe(
    "https://example.com/report.html",
  );
});

test.each([
  "signature",
  "sig",
  "token",
  "access_token",
  "x-amz-signature",
  "x-amz-credential",
  "x-amz-security-token",
  "apikey",
  "api_key",
  "X-AMZ-SIGNATURE", // case-insensitive match on the key
])("canonicalizeUrl strips denylisted query parameter %s", (deniedKey) => {
  const url = `https://example.com/report.html?${deniedKey}=secret&kept=1`;
  const canonical = canonicalizeUrl(url);
  expect(canonical).not.toContain("secret");
  expect(canonical.toLowerCase()).not.toContain(deniedKey.toLowerCase() + "=");
  expect(canonical).toContain("kept=1");
});

test("canonicalizeUrl preserves order of kept query parameters", () => {
  const url = "https://example.com/report.html?b=2&a=1&token=secret&c=3";
  expect(canonicalizeUrl(url)).toBe("https://example.com/report.html?b=2&a=1&c=3");
});

test("canonicalizeUrl is stable and safe for display", () => {
  // ADR 0007 §6.1: one canonical value safe for identity, storage, AND
  // display — no separate redacted form.
  const url = "https://user:pass@example.com/report.html?token=abc&page=2";
  const first = canonicalizeUrl(url);
  const second = canonicalizeUrl(url);
  expect(first).toBe(second);
  expect(first).not.toContain("user");
  expect(first).not.toContain("pass");
  expect(first).not.toContain("abc");
});

test("canonicalizeUrl leaves an already-clean URL unchanged", () => {
  const url = "https://example.com/report.html?page=2";
  expect(canonicalizeUrl(url)).toBe(url);
});

test("SourceIdRegistry claims each id once", () => {
  const registry = new SourceIdRegistry();
  expect(registry.claim("docs/a.md")).toBe(true);
  expect(registry.claim("docs/b.md")).toBe(true);
  expect(registry.claim("docs/a.md")).toBe(false);
});
