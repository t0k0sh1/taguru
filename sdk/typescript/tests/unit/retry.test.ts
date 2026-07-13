import { describe, expect, it, vi } from "vitest";

import { RateLimitError, ServerError, TransportError } from "../../src/errors.js";
import { errBody, okBody, stubClient, type StubResult } from "./stub.js";

vi.mock("../../src/retry.js", async (importOriginal) => {
  const original = await importOriginal<typeof import("../../src/retry.js")>();
  return { ...original, backoffDelay: () => 0 };
});

const EMPTY_MATCHES = { total: 0, matches: [] };
const OP = { subject: "s", label: "l", object: "o", weight: 1.0 };

function flaky(failures: number, failure: () => StubResult, success: unknown = 0) {
  let calls = 0;
  const handler = () => {
    calls += 1;
    return calls <= failures ? failure() : okBody(success);
  };
  return { handler, calls: () => calls };
}

function connectRefused(): Error {
  return new TypeError("fetch failed", {
    cause: Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" }),
  });
}

function midFlight(): Error {
  return new TypeError("fetch failed", {
    cause: Object.assign(new Error("socket hang up"), { code: "UND_ERR_SOCKET" }),
  });
}

describe("retry policy", () => {
  it("retries 429 even on the unsafe write route (shed before executing)", async () => {
    const { handler, calls } = flaky(1, () => errBody(429, "budget", { "retry-after": "0" }));
    const client = stubClient(handler);
    await expect(client.context("sake").addAssociations([OP])).resolves.toBe(0);
    expect(calls()).toBe(2);
  });

  it("retries 503 and honors retry-after", async () => {
    const { handler, calls } = flaky(
      1,
      () => errBody(503, "shed", { "retry-after": "0" }),
      EMPTY_MATCHES,
    );
    const client = stubClient(handler);
    await client.context("sake").recall("cue");
    expect(calls()).toBe(2);
  });

  it("never retries 500", async () => {
    const { handler, calls } = flaky(5, () => errBody(500, "io"));
    const client = stubClient(handler);
    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(ServerError);
    expect(calls()).toBe(1);
  });

  it("retries 502 on a safe read", async () => {
    const { handler, calls } = flaky(1, () => errBody(502, "provider"), EMPTY_MATCHES);
    const client = stubClient(handler);
    await client.context("sake").recall("cue");
    expect(calls()).toBe(2);
  });

  it("retries a pre-connect failure even on the unsafe route", async () => {
    const { handler, calls } = flaky(1, connectRefused);
    const client = stubClient(handler);
    await client.context("sake").addAssociations([OP]);
    expect(calls()).toBe(2);
  });

  it("retries an ambiguous failure on a safe route", async () => {
    const { handler, calls } = flaky(1, midFlight, EMPTY_MATCHES);
    const client = stubClient(handler);
    await client.context("sake").recall("cue");
    expect(calls()).toBe(2);
  });

  it("never retries addAssociations after an ambiguous failure", async () => {
    const { handler, calls } = flaky(1, midFlight);
    const client = stubClient(handler);
    await expect(client.context("sake").addAssociations([OP])).rejects.toBeInstanceOf(
      TransportError,
    );
    expect(calls()).toBe(1);
  });

  it("never retries rename after an ambiguous failure", async () => {
    const { handler, calls } = flaky(1, midFlight);
    const client = stubClient(handler);
    await expect(client.contexts.rename("sake", "shochu")).rejects.toBeInstanceOf(TransportError);
    expect(calls()).toBe(1);

    const group = flaky(1, midFlight);
    const groupClient = stubClient(group.handler);
    await expect(groupClient.groups.rename("kura", "gura")).rejects.toBeInstanceOf(
      TransportError,
    );
    expect(group.calls()).toBe(1);
  });

  it("retries: 0 disables retrying entirely", async () => {
    const { handler, calls } = flaky(1, () => errBody(429, "budget", { "retry-after": "0" }));
    const client = stubClient(handler, { retries: 0 });
    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(RateLimitError);
    expect(calls()).toBe(1);
  });

  it("exhausts the budget and raises the last error", async () => {
    const { handler, calls } = flaky(99, () => errBody(429, "budget", { "retry-after": "0" }));
    const client = stubClient(handler, { retries: 2 });
    await expect(client.context("sake").recall("cue")).rejects.toBeInstanceOf(RateLimitError);
    expect(calls()).toBe(3); // initial + 2 retries
  });
});
