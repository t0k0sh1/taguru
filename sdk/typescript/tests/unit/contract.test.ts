/**
 * The one-time `GET /version` compatibility preflight (ADR 0005 §3.8, §6).
 *
 * Every case here opts in with `checkContract: true` — the default stub
 * client pre-seeds the check as already done (see `stub.ts`) so the rest of
 * this suite's exact call-count and call-path assertions aren't disturbed
 * by a surprise probe.
 */

import { describe, expect, it } from "vitest";

import { Taguru } from "../../src/client.js";
import { incompatibility } from "../../src/contract.js";
import { IncompatibleServerError, TransportError } from "../../src/errors.js";
import { VERSION } from "../../src/version.js";
import { okBody, stubClient, type StubHandler, type StubResult } from "./stub.js";

const EMPTY_MATCHES = { total: 0, matches: [] };

function versionBody(payload: Record<string, unknown>, status = 200): StubResult {
  return { status, body: JSON.stringify(payload) };
}

function compatibleVersion(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { server: "0.6.0", http_contract: { current: 1, supported: [1] }, ...extra };
}

function incompatibleNewer(): Record<string, unknown> {
  return { server: "0.7.0", http_contract: { current: 2, supported: [2] } };
}

describe("preflight", () => {
  it("runs once per client", async () => {
    const paths: string[] = [];
    const handler: StubHandler = (req) => {
      paths.push(req.path);
      return req.path === "/version" ? versionBody(compatibleVersion()) : okBody(EMPTY_MATCHES);
    };
    const client = stubClient(handler, { checkContract: true });

    await client.context("sake").recall("cue");
    await client.context("sake").recall("cue");

    expect(paths.filter((path) => path === "/version")).toHaveLength(1);
    expect(paths).toEqual(["/version", "/contexts/sake/recall", "/contexts/sake/recall"]);
  });

  it("concurrent first calls share one in-flight probe — none races past it", async () => {
    // `stub.ts`'s StubHandler returns synchronously, so a manual fetch is
    // used here to actually delay the /version response until both
    // `recall()` calls are demonstrably in flight and racing against it —
    // proving neither raced past a merely-synchronous "checked" flag the
    // way a naive implementation would let the second one do.
    const paths: string[] = [];
    let releaseProbe: (() => void) | undefined;
    const probeGate = new Promise<void>((resolve) => {
      releaseProbe = resolve;
    });
    const fetchStub: typeof fetch = async (input) => {
      const path = new URL(String(input)).pathname;
      paths.push(path);
      if (path === "/version") {
        await probeGate;
        return new Response(JSON.stringify(incompatibleNewer()), { status: 200 });
      }
      throw new Error(`the real request must not run: ${path}`);
    };
    const client = new Taguru({ base_url: "http://test", api_key: "", fetch: fetchStub });

    const call1 = client
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    const call2 = client
      .context("tea")
      .recall("cue")
      .catch((caught: unknown) => caught);
    // Give both `recall()` calls' synchronous prefixes a chance to run —
    // including the second one observing the shared in-flight probe —
    // before releasing the probe's response.
    await Promise.resolve();
    await Promise.resolve();
    releaseProbe?.();

    const [result1, result2] = await Promise.all([call1, call2]);
    expect(result1).toBeInstanceOf(IncompatibleServerError);
    expect(result2).toBeInstanceOf(IncompatibleServerError);
    expect(paths).toEqual(["/version"]);
  });
});

describe("incompatibility", () => {
  it("a newer server raises with an upgrade-this-SDK remedy", async () => {
    const handler: StubHandler = (req) => {
      if (req.path === "/version") {
        return versionBody(incompatibleNewer());
      }
      throw new Error(`the real request must not run: ${req.path}`);
    };
    const client = stubClient(handler, { checkContract: true });

    const error = await client
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(IncompatibleServerError);
    const shaped = error as IncompatibleServerError;
    expect(shaped.status).toBeNull();
    expect(shaped.sdk_version).toBe(VERSION);
    expect(shaped.server_version).toBe("0.7.0");
    expect(shaped.supported_contracts).toEqual([1]);
    expect(shaped.server_contracts).toEqual([2]);
    expect(shaped.message).toContain("0.7.0");
    expect(shaped.message).toContain("Upgrade this SDK");
  });

  it("keeps blocking every call after the first, not just the ones racing the probe", async () => {
    // A confirmed incompatibility must keep blocking every call after
    // the probe settles, not just the ones that raced the original
    // probe — `contractChecked` alone can't tell "confirmed fine"
    // from "confirmed broken," so the client must remember which one
    // it was.
    const handler: StubHandler = (req) => {
      if (req.path === "/version") {
        return versionBody(incompatibleNewer());
      }
      throw new Error(`the real request must not run: ${req.path}`);
    };
    const client = stubClient(handler, { checkContract: true });

    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(
      IncompatibleServerError,
    );
    // The probe already settled here — this second call sees
    // `contractChecked === true` and must still raise, not silently
    // proceed.
    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(
      IncompatibleServerError,
    );
  });

  // The remaining directions need the SDK's own range to be something
  // other than the real `SUPPORTED_HTTP_CONTRACTS = [1]` — a `const`
  // module binding, immutable even from within its own module, unlike
  // Python's `monkeypatch.setattr`. `incompatibility`'s third
  // parameter exists for exactly this: it defaults to the real
  // constant everywhere production code calls it (`client.ts` never
  // passes a third argument), and lets these two cases override it.

  it("an older server raises with an upgrade-the-server remedy", () => {
    const error = incompatibility(
      { server: "0.5.0", supported: [1] },
      "http://127.0.0.1:8248",
      [2],
    );
    expect(error).toBeInstanceOf(IncompatibleServerError);
    expect(error?.supported_contracts).toEqual([2]);
    expect(error?.server_contracts).toEqual([1]);
    expect(error?.message).toContain("Upgrade the server");
    // The server is already at 0.5.0 — telling the user to "upgrade
    // to 0.5.0" would be nonsensical, since that's the version it's
    // already running.
    expect(error?.message).not.toContain("to 0.5.0");
  });

  it("disjoint interleaved ranges name no direction, just SUPPORTED_HTTP_CONTRACTS", () => {
    // Neither side is simply "newer" — SDK speaks {1, 3}, server
    // speaks {2} — so the remedy must not claim a direction (ADR 0005
    // §6's dual-serving window is not decided, but the message must
    // still make sense if it ever exists).
    const error = incompatibility(
      { server: "0.7.0", supported: [2] },
      "http://127.0.0.1:8248",
      [1, 3],
    );
    expect(error).toBeInstanceOf(IncompatibleServerError);
    expect(error?.message).toContain("SUPPORTED_HTTP_CONTRACTS");
  });

  it("returns null (compatible) when the ranges intersect", () => {
    expect(incompatibility({ server: "0.6.0", supported: [1] }, "http://test")).toBeNull();
  });

  it("returns null (fail-open) for an empty supported array", () => {
    expect(incompatibility({ server: "0.6.0", supported: [] }, "http://test")).toBeNull();
  });
});

describe("fail-open on uninformative /version responses", () => {
  const cases: Array<[string, StubResult]> = [
    ["404 (pre-0.6 server)", { status: 404, body: "" }],
    ["bare ok text", { status: 200, body: "ok" }],
    ["json array body", { status: 200, body: "[1,2,3]" }],
    ["missing http_contract", versionBody({ server: "0.6.0" })],
    ["empty supported", versionBody({ http_contract: { current: 1, supported: [] } })],
    [
      "non-numeric supported",
      versionBody({ http_contract: { current: 1, supported: ["one"] } }),
    ],
    ["401 behind a proxy", { status: 401, body: JSON.stringify({ status: "error", error: "nope" }) }],
  ];

  it.each(cases)("%s does not raise", async (_label, versionResult) => {
    const handler: StubHandler = (req) =>
      req.path === "/version" ? versionResult : okBody(EMPTY_MATCHES);
    const client = stubClient(handler, { checkContract: true });
    await expect(client.context("sake").recall("cue")).resolves.toBeDefined();
  });
});

describe("transport failure", () => {
  it("leaves the contract unchecked for a retry on the next call", async () => {
    const paths: string[] = [];
    const handler: StubHandler = (req) => {
      paths.push(req.path);
      return new Error("connection refused");
    };
    const client = stubClient(handler, { checkContract: true, retries: 0 });

    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(TransportError);
    expect(paths).toEqual(["/version", "/contexts/sake/recall"]);

    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(TransportError);
    expect(paths).toEqual([
      "/version",
      "/contexts/sake/recall",
      "/version",
      "/contexts/sake/recall",
    ]);
  });
});

describe("export paths", () => {
  it("exportStream raises on incompatibility", async () => {
    const handler: StubHandler = (req) => {
      if (req.path === "/version") {
        return versionBody(incompatibleNewer());
      }
      throw new Error(`the export must not run: ${req.path}`);
    };
    const client = stubClient(handler, { checkContract: true });
    const stream = client.context("sake").exportStream();
    await expect(stream.next()).rejects.toBeInstanceOf(IncompatibleServerError);
  });
});

describe("waitUntilReady", () => {
  it("raises immediately instead of stalling out the full timeout", async () => {
    const paths: string[] = [];
    const handler: StubHandler = (req) => {
      paths.push(req.path);
      if (req.path === "/version") {
        return versionBody(incompatibleNewer());
      }
      throw new Error(`live/health must not run: ${req.path}`);
    };
    const client = stubClient(handler, { checkContract: true });

    const started = performance.now();
    await expect(
      client.waitUntilReady({ timeout: 5, interval: 0.5 }),
    ).rejects.toBeInstanceOf(IncompatibleServerError);
    expect(performance.now() - started).toBeLessThan(1000);
    expect(paths).toEqual(["/version"]);
  });
});
