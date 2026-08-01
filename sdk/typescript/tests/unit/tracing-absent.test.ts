/**
 * `@opentelemetry/api` genuinely absent (the common case: it is an
 * OPTIONAL peer dependency — see package.json). `tracing.test.ts` covers
 * the package present; this file simulates it missing via `vi.mock`,
 * since it is always a devDependency here and can't be literally
 * uninstalled the way the Python SDK's CI does for its own equivalent
 * two-phase run.
 *
 * `vi.mock`ing the specifier throws from the mock factory, so any
 * `import("@opentelemetry/api")` — static or dynamic — rejects the same
 * way it would if `node`'s module resolution genuinely couldn't find the
 * package. `vi.resetModules` before each test forces `tracing.ts` to be
 * re-evaluated fresh, so its own module-level `otelModule` cache
 * (`undefined` = not yet attempted) starts clean every time.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

describe("tracing with @opentelemetry/api unavailable", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("span() runs the body against a no-op handle instead of throwing", async () => {
    vi.doMock("@opentelemetry/api", () => {
      throw new Error("simulated: @opentelemetry/api is not installed");
    });
    const tracing = await import("../../src/tracing.js");

    const seen: string[] = [];
    const result = await tracing.span("taguru.retrieve", async (span) => {
      span.count("taguru.origin.count", 1);
      span.flag("taguru.fallback.ran", false);
      span.skip("no_anchors");
      span.citationMissing(1);
      seen.push("ran");
      return 42;
    });

    expect(result).toBe(42);
    expect(seen).toEqual(["ran"]);
  });

  it("injectHeaders() leaves the headers untouched", async () => {
    vi.doMock("@opentelemetry/api", () => {
      throw new Error("simulated: @opentelemetry/api is not installed");
    });
    const tracing = await import("../../src/tracing.js");

    const headers: Record<string, string> = { authorization: "Bearer x" };
    await tracing.injectHeaders(headers);
    expect(headers).toEqual({ authorization: "Bearer x" });
  });

  it("attempts the dynamic import only once, even across many calls", async () => {
    let attempts = 0;
    vi.doMock("@opentelemetry/api", () => {
      attempts += 1;
      throw new Error("simulated: @opentelemetry/api is not installed");
    });
    const tracing = await import("../../src/tracing.js");

    await tracing.span("a", async (span) => span.count("x", 1));
    await tracing.span("b", async (span) => span.count("y", 1));
    await tracing.injectHeaders({});
    await tracing.injectHeaders({});

    // Not vitest's own module cache (a fresh `resetModules` + `doMock`
    // pair happens exactly once per `it`, above) — this is `tracing.ts`'s
    // own `otelModule` variable staying `null` after the first failed
    // attempt, the same caching the Python SDK gets for free from
    // `try/except` running once at module-import time. A version of this
    // module that re-attempted the import on every call would also pass
    // the two tests above, but would fail this one.
    expect(attempts).toBe(1);
  });
});
