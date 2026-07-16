import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { Taguru } from "../../src/client.js";

/** A fetch stub whose response body streams a few chunks, then errors. */
function fetchThatFailsMidStream(errorMessage: string): typeof fetch {
  return (async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('{"taguru_batch": 1, "context": "sake"}\n'));
        controller.error(new Error(errorMessage));
      },
    });
    return new Response(stream, { status: 200 });
  }) as unknown as typeof fetch;
}

describe("exportToFile", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "taguru-export-"));
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("leaves no file at the target when the stream fails mid-write", async () => {
    const target = join(dir, "export.ndjson");
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      fetch: fetchThatFailsMidStream("connection reset"),
    });
    await expect(client.context("sake").exportToFile(target)).rejects.toThrow(/connection reset/);
    expect(await readdir(dir)).toEqual([]);
  });

  it("writes the full export atomically on success", async () => {
    const target = join(dir, "export.ndjson");
    const body = '{"taguru_batch": 1, "context": "sake"}\n{"subject": "s", "label": "l", "object": "o"}\n';
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      fetch: (async () => new Response(body, { status: 200 })) as unknown as typeof fetch,
    });
    await client.context("sake").exportToFile(target);
    expect(await readFile(target, "utf8")).toBe(body);
    expect(await readdir(dir)).toEqual(["export.ndjson"]);
  });
});
