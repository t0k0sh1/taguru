import { describe, expect, it } from "vitest";

import {
  AuthenticationError,
  ConflictError,
  EmbeddingUnavailableError,
  NotFoundError,
  PayloadTooLargeError,
  PermissionDeniedError,
  RateLimitError,
  RequestTimeoutError,
  ServerError,
  ServiceUnavailableError,
  StorageFullError,
  UnexpectedStatusError,
  ValidationError,
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
});
