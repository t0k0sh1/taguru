/**
 * `watchDirectory` (issue #414): the polling directory watcher — each pass
 * is a verbatim `syncObjectStorage` run over a `file://` tree, so these
 * tests cover only what the watcher itself adds: the pass/wait loop,
 * cooperative stop at every stage, and the checkpoint idempotence that
 * makes a quiet pass cheap. Mirrors the Python suite's
 * test_watch_directory.py case-for-case (TypeScript parity: issue #415).
 *
 * Real timers throughout (no fake-timer mocking), matching this port's
 * other tests — intervals are kept at 0 (or, for the "an AbortSignal ends
 * the wait promptly" case, a deliberately long interval that only the
 * signal itself — never the timer — ends).
 */

import {
  mkdtempSync,
  rmSync,
  utimesSync,
  writeFileSync,
  unlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { Taguru } from "taguru";
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { TaguruIngester, type TaguruIngesterFields } from "../../src/ingest.js";
import {
  RunReport,
  watchDirectory,
} from "../../src/ingest-connectors/watch.js";
import { RecordingCheckpointStore } from "./checkpoints.test.js";

const EMPTY_ANSWER = JSON.stringify({
  associations: [],
  aliases: [],
  questions: [],
});

// ---------------------------------------------------------------------------
// A local FakeServer — same shape as s3-connector.test.ts's own (not
// tests/unit/stub.ts's FakeServer; see that file's module doc comment for
// why). Only the routes watchDirectory's own passes actually exercise are
// implemented here.
// ---------------------------------------------------------------------------

class FakeServer {
  imported: string[] = [];
  retracted: string[] = [];
  sources: string[] = [];

  fetch: typeof fetch = async (input, init) => {
    const url = new URL(String(input));
    const path = url.pathname;
    const method = init?.method ?? "GET";
    const ok = (result: unknown): Response =>
      new Response(JSON.stringify({ result, status: "ok", time: 0.001 }), {
        status: 200,
      });

    if (path === "/version") {
      return new Response(
        JSON.stringify({
          server: "0.6.0",
          http_contract: { current: 1, supported: [1] },
        }),
        { status: 200 },
      );
    }
    let body: unknown = null;
    if (typeof init?.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    }

    if (path.endsWith("/labels")) {
      return ok({ total: 0, labels: [] });
    }
    if (path.endsWith("/schema") && method === "GET") {
      return new Response(
        JSON.stringify({
          status: "error",
          code: "no_schema",
          error: "context has no schema document",
          time: 0.001,
        }),
        { status: 404 },
      );
    }
    if (path.endsWith("/sources/retract") && method === "POST") {
      const source = (body as { source: string }).source;
      this.sources = this.sources.filter((s) => s !== source);
      this.retracted.push(source);
      return ok({ associations_touched: 0, passage_removed: true });
    }
    if (path.endsWith("/sources") && method === "GET") {
      const prefix = url.searchParams.get("prefix");
      const candidates = this.sources
        .filter((s) => prefix === null || s.startsWith(prefix))
        .sort();
      return ok({
        total: candidates.length,
        sources: candidates,
        entries: candidates.map((name) => ({ name })),
      });
    }
    if (path === "/import") {
      this.imported.push(typeof init?.body === "string" ? init.body : "");
      return ok({
        batches: [
          {
            context: "sake",
            source: "file:///a.md",
            created: false,
            retracted: 0,
            associations: 0,
            aliases: 0,
            passage_stored: true,
            passage_dropped: false,
            questions_stored: 0,
            questions_dropped: 0,
            sections_stored: 0,
            sections_dropped: 0,
            locators_stored: 0,
            locators_dropped: 0,
            association_paragraphs_dropped: 0,
          },
        ],
      });
    }
    if (path.endsWith("/embeddings/refresh")) {
      return new Response(
        JSON.stringify({ status: "error", error: "no provider", time: 0.001 }),
        { status: 501 },
      );
    }
    throw new Error(`unrouted path: ${path}`);
  };

  client(): Taguru {
    return new Taguru({
      base_url: "http://test",
      api_key: "",
      fetch: this.fetch,
    });
  }
}

function ingester(
  server: FakeServer,
  overrides: Partial<TaguruIngesterFields> = {},
): TaguruIngester {
  return new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({
      responses: Array(20).fill(EMPTY_ANSWER) as string[],
    }),
    client: server.client(),
    ...overrides,
  });
}

function watch(
  directory: string,
  ing: TaguruIngester,
  overrides: Partial<Parameters<typeof watchDirectory>[1]> = {},
): Promise<AsyncGenerator<RunReport, void, unknown>> {
  return watchDirectory(directory, {
    ingester: ing,
    checkpoints: new RecordingCheckpointStore(),
    intervalSecs: 0.0,
    ...overrides,
  });
}

// ---------------------------------------------------------------------------

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "taguru-watch-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe("watchDirectory", () => {
  test("a missing directory fails at the call site, not the first next", async () => {
    await expect(
      watch(join(dir, "does-not-exist"), ingester(new FakeServer())),
    ).rejects.toThrow();
  });

  test("a negative interval is refused", async () => {
    await expect(
      watch(dir, ingester(new FakeServer()), { intervalSecs: -1.0 }),
    ).rejects.toThrow(/intervalSecs/);
  });

  test("each pass yields a report and a quiet pass is all unchanged", async () => {
    writeFileSync(join(dir, "a.md"), "alpha paragraph.");
    const server = new FakeServer();
    const watcher = await watch(dir, ingester(server));

    const first = await watcher.next();
    expect(first.done).toBe(false);
    expect(first.value!.imported).toBe(1);
    expect(server.imported.length).toBe(1);

    // Nothing changed: the second pass skips on the (size, mtime) listing
    // fingerprint — no re-import, and the same generator keeps going.
    const second = await watcher.next();
    expect(second.value!.unchanged).toBe(1);
    expect(second.value!.imported).toBe(0);
    expect(server.imported.length).toBe(1);
  });

  test("a changed file is picked up on the next pass", async () => {
    const target = join(dir, "a.md");
    writeFileSync(target, "alpha paragraph.");
    const server = new FakeServer();
    const watcher = await watch(dir, ingester(server));
    await watcher.next();

    writeFileSync(target, "alpha paragraph, revised for the second pass.");
    // Belt and braces against coarse filesystem mtime granularity: the size
    // already differs, but pin a distinct mtime too.
    const revised = new Date(1_700_000_000 * 1000);
    utimesSync(target, revised, revised);

    const report = await watcher.next();
    expect(report.value!.imported).toBe(1);
    expect(server.imported.length).toBe(2);
  });

  test("a deleted file is detected but never retracted by default", async () => {
    writeFileSync(join(dir, "a.md"), "alpha.");
    writeFileSync(join(dir, "b.md"), "beta.");
    const server = new FakeServer();
    const watcher = await watch(dir, ingester(server));
    await watcher.next();

    unlinkSync(join(dir, "b.md"));
    const report = await watcher.next();
    expect(report.value!.deletedDetected).toBe(1);
    expect(report.value!.retracted).toBe(0);
    expect(server.retracted).toEqual([]);
  });

  test("sources carry the file uri identity", async () => {
    const { mkdirSync } = await import("node:fs");
    mkdirSync(join(dir, "docs"));
    writeFileSync(join(dir, "docs", "a.md"), "alpha.");
    const server = new FakeServer();
    const watcher = await watch(dir, ingester(server));
    const report = await watcher.next();

    expect(report.value!.imported).toBe(1);
    const imported = report.value!.events.filter((e) => e.phase === "imported");
    expect(imported).toHaveLength(1);
    expect(imported[0]!.source.startsWith("file://")).toBe(true);
    expect(imported[0]!.source.endsWith("/docs/a.md")).toBe(true);
  });

  test("a stop already set yields no pass at all", async () => {
    writeFileSync(join(dir, "a.md"), "alpha.");
    const server = new FakeServer();
    const controller = new AbortController();
    controller.abort();
    const watcher = await watch(dir, ingester(server), {
      shouldStop: controller.signal,
    });

    const reports: RunReport[] = [];
    for await (const report of watcher) {
      reports.push(report);
    }
    expect(reports).toEqual([]);
    expect(server.imported).toEqual([]);
  });

  test("an abort during the wait ends the watch promptly", async () => {
    writeFileSync(join(dir, "a.md"), "alpha.");
    const server = new FakeServer();
    const controller = new AbortController();
    // A wait long enough that only the abort ending it lets this test
    // finish — the signal wakes the wait the moment it fires, no polling.
    const watcher = await watch(dir, ingester(server), {
      intervalSecs: 600.0,
      shouldStop: controller.signal,
    });

    const first = await watcher.next();
    expect(first.value!.imported).toBe(1);
    controller.abort();
    const second = await watcher.next();
    expect(second.done).toBe(true);
  });

  test("a callable stop is honored between passes", async () => {
    writeFileSync(join(dir, "a.md"), "alpha.");
    const server = new FakeServer();
    let passes = 0;
    const stopAfterOne = (): boolean => passes >= 1;

    const watcher = await watch(dir, ingester(server), {
      shouldStop: stopAfterOne,
    });
    for await (const _report of watcher) {
      passes += 1;
    }
    expect(passes).toBe(1);
  });

  test("dry run passes touch nothing", async () => {
    writeFileSync(join(dir, "a.md"), "alpha.");
    const server = new FakeServer();
    const watcher = await watch(dir, ingester(server), { dryRun: true });

    const report = await watcher.next();
    expect(report.value!.parsed).toBe(1);
    expect(server.imported).toEqual([]);
  });
});
