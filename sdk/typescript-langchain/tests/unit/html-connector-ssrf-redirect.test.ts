/**
 * The SSRF guard re-runs per redirect hop (issue #737): with
 * `allowPrivateNetworks` left at its false default, an origin that
 * answers a redirect INTO a private/internal address must be refused
 * before that hop ever connects. The loopback test server plays the
 * "public" origin via a dns mock that declares exactly `127.0.0.1`
 * public (the socket still connects — Node skips dns for IP literals);
 * the redirect target hostname resolves, through the same mock, to the
 * cloud metadata address, so the refusal below is the REAL
 * `isBlockedIp` verdict on the hop, not a mocked one. Own file on
 * purpose: `vi.mock` is file-scoped, and the sibling fetch tests rely
 * on the real resolver's verdicts (127.0.0.1 blocked by default).
 */

import { expect, test, vi } from "vitest";

vi.mock("node:dns/promises", async (importOriginal) => {
  const real = await importOriginal<typeof import("node:dns/promises")>();
  const lookup = (async (hostname: string, options: unknown) => {
    if (hostname === "127.0.0.1") {
      return [{ address: "203.0.113.7", family: 4 }]; // TEST-NET-3: reads as public
    }
    if (hostname === "redirect-target.internal") {
      return [{ address: "169.254.169.254", family: 4 }]; // the metadata endpoint
    }
    return (real.lookup as unknown as (h: string, o: unknown) => Promise<unknown>)(
      hostname,
      options,
    );
  }) as unknown as typeof real.lookup;
  return { ...real, lookup };
});

import { HtmlConnector } from "../../src/ingest-connectors/html.js";
import { serve } from "../httpd.js";

test("a redirect to a private address is refused with the guard on", async () => {
  const server = await serve({
    "/start": { location: "http://redirect-target.internal/held" },
  });
  try {
    const document = await new HtmlConnector().read(`${server.baseUrl}/start`);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(["unreadable"]);
    expect(document.diagnostics[0]!.message).toContain("private/internal");
  } finally {
    await server.close();
  }
});
