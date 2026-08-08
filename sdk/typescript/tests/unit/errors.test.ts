import { describe, expect, it } from "vitest";

import {
  AuthenticationError,
  ConflictError,
  EmbeddingUnavailableError,
  IncompatibleServerError,
  NotFoundError,
  PayloadTooLargeError,
  PermissionDeniedError,
  RateLimitError,
  RequestTimeoutError,
  ServerError,
  ServiceUnavailableError,
  StorageFullError,
  TaguruError,
  UnexpectedStatusError,
  ValidationError,
  errorForStatus,
} from "../../src/errors.js";
import { errBody, stubClient } from "./stub.js";

const TABLE: Array<[number, new (...args: never[]) => Error]> = [
  [400, ValidationError],
  [401, AuthenticationError],
  [403, PermissionDeniedError],
  [404, NotFoundError],
  [405, UnexpectedStatusError],
  [408, RequestTimeoutError],
  [409, ConflictError],
  [413, PayloadTooLargeError],
  [415, ValidationError],
  [422, ValidationError],
  [429, RateLimitError],
  [500, ServerError],
  [501, EmbeddingUnavailableError],
  [502, EmbeddingUnavailableError],
  [503, ServiceUnavailableError],
  [507, StorageFullError],
  [599, ServerError],
];

describe("status → error class table", () => {
  it.each(TABLE)("maps %i", async (status, expected) => {
    const client = stubClient(() => errBody(status, "boom"), { retries: 0 });
    const error = await client
      .context("sake")
      .recall("cue")
      .then(() => null)
      .catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(expected);
    const shaped = error as InstanceType<typeof ValidationError>;
    expect(shaped.status).toBe(status);
    expect(shaped.message).toBe("boom");
    expect(shaped.time).toBe(0.001);
  });

  it("surfaces the machine-readable code; old servers yield null", async () => {
    const coded = stubClient(() => errBody(404, "context 'x' not found", undefined, "no_context"), {
      retries: 0,
    });
    const notFound = await coded
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    expect((notFound as NotFoundError).code).toBe("no_context");

    const limited = stubClient(
      () => errBody(429, "budget", { "retry-after": "7" }, "rate_limited"),
      { retries: 0 },
    );
    const rate = await limited
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    expect((rate as RateLimitError).code).toBe("rate_limited");
    expect((rate as RateLimitError).retry_after).toBe(7);

    // A body without the field (a server predating it) decodes to null.
    const legacy = stubClient(() => errBody(404, "gone"), { retries: 0 });
    const missing = await legacy
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    expect((missing as NotFoundError).code).toBeNull();
  });

  it("maps a plain-text 413 body (axum's own rejection shape)", async () => {
    const client = stubClient(() => ({ status: 413, body: "length limit exceeded" }), {
      retries: 0,
    });
    const error = await client
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(PayloadTooLargeError);
    expect((error as PayloadTooLargeError).message).toBe("length limit exceeded");
    expect((error as PayloadTooLargeError).body).toBe("length limit exceeded");
  });

  it("carries retry_after on 429 and 503", async () => {
    const rate = stubClient(() => errBody(429, "budget", { "retry-after": "7" }), { retries: 0 });
    const rateError = await rate
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    expect((rateError as RateLimitError).retry_after).toBe(7);

    const shed = stubClient(() => errBody(503, "shed", { "retry-after": "2" }), { retries: 0 });
    const shedError = await shed
      .context("sake")
      .recall("cue")
      .catch((caught: unknown) => caught);
    expect(shedError).toBeInstanceOf(ServerError);
    expect((shedError as ServiceUnavailableError).retry_after).toBe(2);
  });

  it("distinguishes 501 from 502 via reason", async () => {
    const unconfigured = stubClient(() => errBody(501, "no provider"), { retries: 0 });
    const notConfigured = await unconfigured
      .context("sake")
      .refreshEmbeddings()
      .catch((caught: unknown) => caught);
    expect((notConfigured as EmbeddingUnavailableError).reason).toBe("not_configured");

    const failing = stubClient(() => errBody(502, "provider died"), { retries: 0 });
    const providerError = await failing
      .context("sake")
      .refreshEmbeddings()
      .catch((caught: unknown) => caught);
    expect((providerError as EmbeddingUnavailableError).reason).toBe("provider_error");
  });

  it("errorForStatus never returns IncompatibleServerError for any mapped status", () => {
    for (const [status] of TABLE) {
      expect(errorForStatus(status, "x")).not.toBeInstanceOf(IncompatibleServerError);
    }
  });
});

describe("IncompatibleServerError", () => {
  it("sets .name, is a TaguruError, and carries a null status", () => {
    const error = new IncompatibleServerError("boom", {
      sdk_version: "0.5.0",
      server_version: "0.7.0",
      supported_contracts: [1],
      server_contracts: [2],
    });
    expect(error.name).toBe("IncompatibleServerError");
    expect(error).toBeInstanceOf(TaguruError);
    expect(error.status).toBeNull();
    expect(error.sdk_version).toBe("0.5.0");
    expect(error.server_version).toBe("0.7.0");
    expect(error.supported_contracts).toEqual([1]);
    expect(error.server_contracts).toEqual([2]);
  });
});

describe("TaguruError cause plumbing", () => {
  it("carries a given cause through to Error.cause", () => {
    const cause = new Error("root failure");
    const error = new TaguruError("boom", { cause });
    expect(error.cause).toBe(cause);
    expect("cause" in error).toBe(true);
  });

  it("never sets .cause at all when none is given", () => {
    const error = new TaguruError("boom");
    expect(error.cause).toBeUndefined();
    // Distinguishes "no cause option was passed" from "cause was passed
    // as undefined": the constructor must not forward an empty `{cause:
    // undefined}` bag to Error() when `options.cause` is absent.
    expect("cause" in error).toBe(false);
  });
});

describe("errorForStatus 5xx boundary", () => {
  it("maps every 5xx status not otherwise mapped to ServerError", () => {
    expect(errorForStatus(500, "x")).toBeInstanceOf(ServerError);
    expect(errorForStatus(599, "x")).toBeInstanceOf(ServerError);
  });

  it("maps 499 and 600 (just outside the 5xx band) to UnexpectedStatusError, not ServerError", () => {
    const below = errorForStatus(499, "x");
    expect(below).toBeInstanceOf(UnexpectedStatusError);
    expect(below).not.toBeInstanceOf(ServerError);

    const above = errorForStatus(600, "x");
    expect(above).toBeInstanceOf(UnexpectedStatusError);
    expect(above).not.toBeInstanceOf(ServerError);
  });
});
