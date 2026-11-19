/**
 * Regression tests for client-method hardening: the fixes ported from the
 * Python SDK (its commit 1622335 — delete's retry class and exportStream's
 * transport error wrapping), the retract retry classes, body-read failures
 * on the ordinary request path, exportStream's early exit, and the wire
 * fields the response models carry (compact's publish verdict, a router's
 * `unreached`, the directory's `schema_mode`). Kept apart from
 * retry.test.ts — these pin CLIENT-method behavior, not the retry-policy
 * predicates.
 */

import { describe, expect, it, vi } from "vitest";

import { Taguru } from "../../src/client.js";
import { ServerError, TransportError } from "../../src/errors.js";
import type { DirectoryEntry } from "../../src/models.js";
import { errBody, okBody, stubClient } from "./stub.js";

/** A client whose contract preflight is pre-seeded, on a raw fetch impl
 * (stubFetch can't produce a streaming body). */
const rawClient = (fetchImpl: typeof fetch, options: { retries?: number } = {}): Taguru => {
  const client = new Taguru({
    base_url: "http://test",
    api_key: "k",
    fetch: fetchImpl,
    retries: options.retries,
  });
  (client as unknown as { contractChecked: boolean }).contractChecked = true;
  return client;
};

/** A response whose headers arrive but whose body drops mid-read. */
const droppedBody = (status = 200): Response =>
  new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"result":'));
        controller.error(new TypeError("terminated"));
      },
    }),
    { status },
  );

const envelope = (result: unknown, extra: Record<string, unknown> = {}): Response =>
  new Response(JSON.stringify({ result, status: "ok", time: 0.001, ...extra }), { status: 200 });

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

describe("exportStream early exit", () => {
  it("cancels the body when the consumer stops reading", async () => {
    // An abandoned body left unread holds its connection open; breaking out
    // of the loop must cancel it, not just release the reader's lock.
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(new TextEncoder().encode("chunk"));
      },
      cancel() {
        cancelled = true;
      },
    });
    const client = rawClient(async () => new Response(body, { status: 200 }));
    for await (const chunk of client.context("sake").exportStream()) {
      expect(chunk.byteLength).toBeGreaterThan(0);
      break;
    }
    expect(cancelled).toBe(true);
  });
});

describe("body-read failures on the ordinary request path", () => {
  it("wraps a mid-body drop in TransportError and retries it on a safe route", async () => {
    let calls = 0;
    const client = rawClient(
      async () => {
        calls += 1;
        return calls === 1 ? droppedBody() : envelope({ total: 0, matches: [] });
      },
      { retries: 3 },
    );
    const page = await client.context("sake").recall("cue");
    expect(page.matches).toEqual([]);
    expect(calls).toBe(2);
  });

  it("does not retry a mid-body drop on an unsafe_on_ambiguous route", async () => {
    // The headers arrived, so the request reached the server: an
    // accumulating write must not be re-sent.
    let calls = 0;
    const client = rawClient(
      async () => {
        calls += 1;
        return droppedBody();
      },
      { retries: 3 },
    );
    await expect(
      client.context("sake").addAssociations([{ subject: "a", label: "b", object: "c", weight: 1 }]),
    ).rejects.toThrow(TransportError);
    expect(calls).toBe(1);
  });

  it("wraps a drop while reading an error status's body", async () => {
    const client = rawClient(async () => droppedBody(500), { retries: 0 });
    await expect(client.context("sake").recall("cue")).rejects.toThrow(TransportError);
  });

  it("wraps the request timeout firing after the headers arrived", async () => {
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      timeout: 0.05,
      retries: 0,
      fetch: async (_url, init) =>
        new Response(
          new ReadableStream<Uint8Array>({
            pull: () =>
              new Promise<void>((_resolve, reject) => {
                init!.signal!.addEventListener("abort", () => reject(init!.signal!.reason));
              }),
          }),
          { status: 200 },
        ),
    });
    (client as unknown as { contractChecked: boolean }).contractChecked = true;
    await expect(client.context("sake").recall("cue")).rejects.toThrow(TransportError);
  });

  it("leaves the contract probe unchecked when its body drops", async () => {
    // Nothing was learned from a body that never arrived: the next call
    // must probe again instead of skipping the check for the process's life.
    let probes = 0;
    const client = new Taguru({
      base_url: "http://test",
      api_key: "",
      retries: 0,
      fetch: async (url) => {
        if (new URL(String(url)).pathname === "/version") {
          probes += 1;
          return droppedBody();
        }
        return envelope({ total: 0, matches: [] });
      },
    });
    await client.context("sake").recall("cue");
    await client.context("sake").recall("cue");
    expect(probes).toBe(2);
  });
});

describe("retract routes are unsafe_on_ambiguous", () => {
  it.each([
    ["retractAssociation", (client: Taguru) => client.context("sake").retractAssociation("a", "b", "c")],
    ["retractSource", (client: Taguru) => client.context("sake").retractSource("docs/a.md")],
  ])("%s does not retry after an ambiguous transport failure", async (_kind, call) => {
    // A repeat of an APPLIED retract answers "nothing changed"
    // (`retracted: false` / zero touched) — an automatic retry would
    // misreport a call that did change the graph.
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
  });

  it("still retries a dry-run retractSource, which writes nothing", async () => {
    let calls = 0;
    const client = stubClient(
      () => {
        calls += 1;
        return calls === 1
          ? new Error("socket hang up mid-response")
          : okBody({ associations_touched: 2, passage_removed: true });
      },
      { retries: 3 },
    );
    const preview = await client.context("sake").retractSource("docs/a.md", { dry_run: true });
    expect(preview.associations_touched).toBe(2);
    expect(calls).toBe(2);
  });
});

describe("compact's publish verdict", () => {
  const counts = { bytes_before: 100, bytes_after: 40, dead_edges: 3, aliases_dropped: 1 };

  it("carries image_persisted and passages_compacted as sent", async () => {
    const client = stubClient(() =>
      okBody({ ...counts, passages_compacted: true, image_persisted: false }),
    );
    expect(await client.context("sake").compact()).toEqual({
      ...counts,
      passages_compacted: true,
      image_persisted: false,
    });
  });

  it("reads an older server's silence as persisted, passages untouched", async () => {
    const client = stubClient(() => okBody(counts));
    expect(await client.context("sake").compact()).toEqual({
      ...counts,
      passages_compacted: false,
      image_persisted: true,
    });
  });
});

describe("a router's unreached report", () => {
  const unreached = [{ shard: "http://shard-b:8248", context_ids: ["id-b"], error: "refused" }];

  it.each([
    ["contexts.list", (client: Taguru) => client.contexts.list(), { total: 0, contexts: [] }],
    ["recall", (client: Taguru) => client.recall("cue", { context_ids: ["id-a"] }), { total: 0, matches: [] }],
    [
      "query",
      (client: Taguru) => client.query({ context_ids: ["id-a"], subject: "s" }),
      { total: 0, matches: [] },
    ],
    [
      "searchPassages",
      (client: Taguru) => client.searchPassages("q", { context_ids: ["id-a"] }),
      { plan: { contexts: [] }, hits: [] },
    ],
  ] as const)("%s surfaces it beside the result", async (_kind, call, result) => {
    const partial = rawClient(async () => envelope(result, { unreached }));
    expect(await call(partial)).toEqual({ ...result, unreached });
    const complete = rawClient(async () => envelope(result));
    expect(await call(complete)).toEqual({ ...result, unreached: [] });
  });
});

describe("the directory's schema_mode", () => {
  it("passes the installed schema's mode, and null for none, through", async () => {
    const row = { id: "id-a", name: "sake", schema_mode: "strict" };
    const client = stubClient(() =>
      okBody({ total: 2, contexts: [row, { ...row, id: "id-b", schema_mode: null }] }),
    );
    const page = await client.contexts.list();
    const modes: Array<DirectoryEntry["schema_mode"]> = page.contexts.map((entry) => entry.schema_mode);
    expect(modes).toEqual(["strict", null]);
  });
});
