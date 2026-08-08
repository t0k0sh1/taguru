/**
 * Regression tests for the hardening fixes ported from the Python SDK
 * (its commit 1622335): delete's retry class and exportStream's transport
 * error wrapping. Kept apart from retry.test.ts — these pin CLIENT-method
 * behavior, not the retry-policy predicates.
 */

import { describe, expect, it, vi } from "vitest";

import { Taguru } from "../../src/client.js";
import { ServerError, TransportError } from "../../src/errors.js";
import { errBody, okBody, stubClient } from "./stub.js";

// "still retries a pre-connect failure" below exercises a real retry, which
// sleeps for `backoffDelay()` between attempts — mocked to 0 so the test
// (and the mutation-testing gate driving it, which mutates that very
// function) stays fast and deterministic instead of racing a real clock.
// Same pattern as retry.test.ts.
vi.mock("../../src/retry.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("../../src/retry.js")>();
  return { ...original, backoffDelay: () => 0 };
});

describe("delete is unsafe_on_ambiguous", () => {
  it.each([
    ["contexts", (client: Taguru) => client.contexts.delete("sake")],
    ["groups", (client: Taguru) => client.groups.delete("bundle")],
  ])(
    "%s.delete does not retry after an ambiguous transport failure",
    async (_kind, call) => {
      // A repeat of an APPLIED delete answers 404 — an automatic retry
      // here would turn a success into NotFoundError, so the ambiguous
      // failure must surface instead of retrying.
      let calls = 0;
      const client = stubClient(
        () => {
          calls += 1;
          return new Error("socket hang up mid-response");
        },
        { retries: 3 },
      );
      await expect(call(client)).rejects.toThrow(TransportError);
      expect(calls).toBe(1);
    },
  );

  it("still retries a pre-connect failure, which never reached the server", async () => {
    let calls = 0;
    const client = stubClient(
      () => {
        calls += 1;
        if (calls === 1) {
          return Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" });
        }
        return okBody(true);
      },
      { retries: 3 },
    );
    expect(await client.contexts.delete("sake")).toBe(true);
    expect(calls).toBe(2);
  });
});

describe("exportStream wraps connection failures in TransportError", () => {
  /** A client whose contract preflight is pre-seeded, on a raw fetch impl
   * (stubFetch can't produce a streaming body). */
  const rawClient = (fetchImpl: typeof fetch): Taguru => {
    const client = new Taguru({ base_url: "http://test", api_key: "k", fetch: fetchImpl });
    (client as unknown as { contractChecked: boolean }).contractChecked = true;
    return client;
  };

  const drain = async (client: Taguru): Promise<Uint8Array[]> => {
    const chunks: Uint8Array[] = [];
    for await (const chunk of client.context("sake").exportStream()) {
      chunks.push(chunk);
    }
    return chunks;
  };

  it("wraps a failed connect", async () => {
    const client = rawClient(async () => {
      throw new TypeError("fetch failed");
    });
    await expect(drain(client)).rejects.toThrow(TransportError);
  });

  it("wraps a mid-stream connection drop", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("first chunk"));
        controller.error(new TypeError("terminated"));
      },
    });
    const client = rawClient(async () => new Response(body, { status: 200 }));
    await expect(drain(client)).rejects.toThrow(TransportError);
  });

  it("does not double-wrap an HTTP error status", async () => {
    const client = stubClient(() => errBody(500, "boom"));
    await expect(drain(client)).rejects.toThrow(ServerError);
  });
});
