import { describe, expect, it, vi } from "vitest";

import {
  PermissionDeniedError,
  RateLimitError,
  ServerError,
  TransportError,
} from "../../src/errors.js";
import { parseRetryAfter } from "../../src/retry.js";
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
    await expect(client.context("sake").addAssociations([OP])).resolves.toEqual({
      applied: 0,
      issues: [],
      schema_violations: 0,
    });
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

  it("never retries 502 on the unsafe write route", async () => {
    const { handler, calls } = flaky(1, () => errBody(502, "provider"));
    const client = stubClient(handler);
    await expect(client.context("sake").addAssociations([OP])).rejects.toBeInstanceOf(ServerError);
    expect(calls()).toBe(1);
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

  it("surfaces a replica's write refusal first try, writer named (no retry loop)", async () => {
    // A read replica answers every mutating verb 403 `read_only_replica`
    // (server issue #129) — a deliberate refusal retrying cannot change,
    // so it must raise on the FIRST attempt with the writer's address and
    // machine-readable code intact for rerouting.
    const { handler, calls } = flaky(99, () =>
      errBody(
        403,
        "this instance is a read replica: it serves every retrieval verb, " +
          "but writes go to the writer at http://writer.internal:8248",
        undefined,
        "read_only_replica",
      ),
    );
    const client = stubClient(handler);
    const refusal = client.context("sake").addAssociations([OP]);
    await expect(refusal).rejects.toBeInstanceOf(PermissionDeniedError);
    await refusal.catch((error: PermissionDeniedError) => {
      expect(error.code).toBe("read_only_replica");
      expect(error.message).toContain("http://writer.internal:8248");
    });
    expect(calls()).toBe(1);
  });
});

describe("parseRetryAfter", () => {
  it("takes a bare non-negative delay", () => {
    expect(parseRetryAfter("5")).toBe(5);
    expect(parseRetryAfter("  0.5  ")).toBe(0.5);
    expect(parseRetryAfter("1e3")).toBe(1000);
    expect(parseRetryAfter("0")).toBe(0);
  });

  it("refuses a trailing tail, non-finite, or negative value (falls back to backoff)", () => {
    // `Number.parseFloat` would have taken the leading number; the strict
    // parse refuses anything that is not a whole bare delay.
    expect(parseRetryAfter("5 seconds")).toBeNull();
    expect(parseRetryAfter("5xyz")).toBeNull();
    expect(parseRetryAfter("0x10")).toBeNull();
    expect(parseRetryAfter("Infinity")).toBeNull();
    expect(parseRetryAfter("1e400")).toBeNull(); // overflows to Infinity
    expect(parseRetryAfter("-1")).toBeNull();
    expect(parseRetryAfter("")).toBeNull();
    expect(parseRetryAfter(null)).toBeNull();
  });

  it("takes every numeric shape the strict regex must accept, not just single digits", () => {
    // Each of these pins one specific character class in the regex —
    // a leading `+` sign, more than one digit before the decimal point,
    // more than one digit after a bare leading dot, a signed exponent,
    // and a multi-digit exponent — so a narrowed class (e.g. `\d` instead
    // of `\d+`) still matches the single-digit cases above but fails here.
    expect(parseRetryAfter("+5")).toBe(5);
    expect(parseRetryAfter("12.5")).toBe(12.5);
    expect(parseRetryAfter(".75")).toBe(0.75);
    expect(parseRetryAfter("1e+5")).toBe(100000);
    expect(parseRetryAfter("1e12")).toBe(1e12);
  });
});
