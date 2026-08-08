/**
 * S3Connector/syncObjectStorage (ADR 0007 §9, issue #351): dispatch by
 * extension/content-type, S3 identity re-stamping, tag mapping, the
 * two-layer checkpoint (fetch-skip + ingest-skip), deletion policy, and
 * `dryRun` — mirrors the Python suite's test_s3_connector.py case-for-case
 * (TypeScript parity: issue #415), exercised against `FakeObjectStore`
 * (tests/s3.ts, no filesystem, no network) and a local `FakeServer` (below).
 *
 * `FakeServer` here is NOT `tests/unit/stub.ts`'s `FakeServer`: this suite
 * needs two routes (`POST .../sources/retract`, `GET .../sources`) plus an
 * `importResultOverride` knob that stub.ts's shared fake does not carry —
 * mirrors the Python suite's own `conftest.py::FakeServer`, which carries
 * exactly these extra fields for the same reason (`sources`/`retracted`
 * back the retract/mirror deletion-policy tests; `import_result_override`
 * backs the locators/sections-dropped test). Duplicated rather than
 * extending stub.ts's `FakeServer`, which this port's scope deliberately
 * excludes from edits.
 *
 * Two deliberate deviations from the Python original, both forced by this
 * SDK's own, already-established shape (see s3.ts's own module doc comment
 * for the first):
 *
 * - No test mirrors
 *   `test_retract_without_a_sync_client_raises_instead_of_guessing`: this
 *   SDK's `TaguruIngester` always has exactly one client (no
 *   `async_client`-only state to construct), so there is nothing to prove
 *   here.
 * - `eventsOut is closed even when the pass raises` (below) triggers the
 *   exception a different way than the Python original's own
 *   `test_events_out_is_closed_even_when_the_pass_raises` (which relies on
 *   the now-inapplicable missing-sync-client `ValueError`): a store whose
 *   `get` throws a plain, unclassified `Error` — the one exception shape
 *   `syncObjectStorage`'s own catch blocks deliberately do not swallow —
 *   proves the same `try`/`finally` guarantee.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { Taguru } from "taguru";
import { describe, expect, test, vi } from "vitest";

import { docxBytes, para } from "../docx.js";
import { FakeObjectStore } from "../s3.js";
import { TaguruIngester, type TaguruIngesterFields } from "../../src/ingest.js";
import { MAX_NAME_BYTES, MAX_TAG_BYTES } from "../../src/ingest-connectors/document.js";
import {
  ObjectMeta,
  ObjectNotFoundError,
  type ObjectStore,
  PermanentStoreError,
  TransientStoreError,
} from "../../src/ingest-connectors/objectstore.js";
import { S3ObjectStore, type S3ClientLike, type S3ObjectLike } from "../../src/ingest-connectors/objectstore-s3.js";
import * as observability from "../../src/ingest-connectors/observability.js";
import { RunRecorder } from "../../src/ingest-connectors/observability.js";
import {
  ReadResult,
  S3Connector,
  S3ObjectCheckpoint,
  S3SyncReport,
  SourceEvent,
  syncObjectStorage,
} from "../../src/ingest-connectors/s3.js";
import { textPdf } from "../pdfs.js";
import { RecordingCheckpointStore } from "./checkpoints.test.js";

const EMPTY_ANSWER = JSON.stringify({ associations: [], aliases: [], questions: [] });

function enc(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

async function firstMeta(store: ObjectStore, prefix: string): Promise<ObjectMeta> {
  for await (const meta of store.list(prefix)) {
    return meta;
  }
  throw new Error("store.list yielded no objects");
}

// ---------------------------------------------------------------------------
// A local FakeServer — see this file's own module doc comment for why it
// is not tests/unit/stub.ts's FakeServer.
// ---------------------------------------------------------------------------

class FakeServer {
  imported: string[] = [];
  failImport = false;
  importResultOverride: Record<string, unknown> | null = null;
  embeddingsRefreshStatus: 200 | 501 | 502 = 501;
  schemaDocument: Record<string, unknown> | null = null;
  /** The context's current source list, and every source `retractSource`
   * has been asked to withdraw (in call order) — backs `syncObjectStorage`'s
   * own `retract`/`mirror` deletion-policy tests: seed `sources` to
   * simulate what the server already holds, then assert `retracted`. */
  sources: string[] = [];
  retracted: string[] = [];

  fetch: typeof fetch = async (input, init) => {
    const url = new URL(String(input));
    const path = url.pathname;
    const method = init?.method ?? "GET";
    const ok = (result: unknown): Response =>
      new Response(JSON.stringify({ result, status: "ok", time: 0.001 }), { status: 200 });

    if (path === "/version") {
      return new Response(
        JSON.stringify({ server: "0.6.0", http_contract: { current: 1, supported: [1] } }),
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
      if (this.schemaDocument === null) {
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
      return ok(this.schemaDocument);
    }
    if (path.endsWith("/sources/retract") && method === "POST") {
      const source = (body as { source: string }).source;
      this.sources = this.sources.filter((s) => s !== source);
      this.retracted.push(source);
      return ok({ associations_touched: 0, passage_removed: true });
    }
    if (path.endsWith("/sources") && method === "GET") {
      const prefix = url.searchParams.get("prefix");
      const after = url.searchParams.get("after");
      const limitRaw = url.searchParams.get("limit");
      const limit = limitRaw !== null ? Number(limitRaw) : null;
      let candidates = this.sources
        .filter((s) => prefix === null || s.startsWith(prefix))
        .sort();
      const total = candidates.length;
      if (after !== null) {
        candidates = candidates.filter((s) => s > after);
      }
      if (limit !== null) {
        candidates = candidates.slice(0, limit);
      }
      return ok({ total, sources: candidates, entries: candidates.map((name) => ({ name })) });
    }
    if (path === "/import") {
      if (this.failImport) {
        return new Response(
          JSON.stringify({ status: "error", code: "internal", error: "simulated failure", time: 0.001 }),
          { status: 500 },
        );
      }
      this.imported.push(typeof init?.body === "string" ? init.body : "");
      const batchResult =
        this.importResultOverride ?? {
          context: "sake",
          source: "docs/aomine.md",
          created: false,
          retracted: 0,
          associations: 2,
          aliases: 1,
          passage_stored: true,
          passage_dropped: false,
          questions_stored: 1,
          questions_dropped: 0,
          sections_stored: 0,
          sections_dropped: 0,
          locators_stored: 0,
          locators_dropped: 0,
          association_paragraphs_dropped: 0,
        };
      return ok({ batches: [batchResult] });
    }
    if (path.endsWith("/embeddings/refresh")) {
      if (this.embeddingsRefreshStatus === 200) {
        return ok({ embedded: 0, total: 0 });
      }
      const error = this.embeddingsRefreshStatus === 502 ? "provider failure" : "no provider";
      return new Response(
        JSON.stringify({ status: "error", error, time: 0.001 }),
        { status: this.embeddingsRefreshStatus },
      );
    }
    throw new Error(`unrouted path: ${path}`);
  };

  client(): Taguru {
    return new Taguru({ base_url: "http://test", api_key: "", fetch: this.fetch });
  }
}

function ingester(server: FakeServer, overrides: Partial<TaguruIngesterFields> = {}): TaguruIngester {
  return new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({ responses: Array(20).fill(EMPTY_ANSWER) as string[] }),
    client: server.client(),
    ...overrides,
  });
}

// ---------------------------------------------------------------------------

test("Phase and SourceEvent are still importable from the s3 submodule", () => {
  expect(S3SyncReport).toBe(observability.RunReport);
  const event = new SourceEvent({ source: "s3://b/k", phase: "discovered", elapsedMs: 0 });
  expect(event).toBeInstanceOf(observability.SourceEvent);
});

// ---------------------------------------------------------------------------
// S3Connector — dispatch, identity re-stamping, size/source-id/tag handling
// ---------------------------------------------------------------------------

describe("S3Connector dispatch", () => {
  test("dispatches a text object and restamps its identity", async () => {
    const store = new FakeObjectStore("reports");
    store.put("notes.md", enc("paragraph one.\n\nparagraph two."));
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.source).toBe("s3://reports/notes.md");
    expect(document.metadata.originUri).toBe("s3://reports/notes.md");
    expect(document.metadata.displayName).toBe("notes.md");
    expect(document.fingerprintInputs.parser).toBe("taguru-text-connector");
    expect(document.diagnostics).toEqual([]);
  });

  test("dispatches an html object", async () => {
    const store = new FakeObjectStore("reports");
    store.put("page.html", enc("<html><body><p>hello world</p></body></html>"));
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.text).toBe("hello world");
    expect(document.fingerprintInputs.parser).toBe("taguru-html-connector");
  });

  test("dispatches a pdf object", async () => {
    const store = new FakeObjectStore("reports");
    store.put("q1.pdf", textPdf(["quarterly results"]));
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.text).toBe("quarterly results");
    expect(document.fingerprintInputs.parser).toBe("taguru-pdf-connector");
    expect(document.locators[0]!.locator.kind).toBe("page");
  });

  test("dispatches a docx object", async () => {
    const store = new FakeObjectStore("reports");
    store.put("q1.docx", docxBytes(para("quarterly results")));
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.text).toBe("quarterly results");
    expect(document.fingerprintInputs.parser).toBe("taguru-docx-connector");
  });

  test("unsupported extension still gets one content-type check", async () => {
    const store = new FakeObjectStore();
    store.put("archive.zip", enc("PK\x03\x04..."));
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.text).toBe("");
    expect(document.diagnostics[0]!.code).toBe("unsupported_format");
    expect([...store.getCalls.entries()]).toEqual([["archive.zip", 1]]);
  });

  test("missing extension falls back to content type", async () => {
    const store = new FakeObjectStore();
    store.put("data", enc("plain body"), { contentType: "text/plain" });
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.text).toBe("plain body");
    expect(document.fingerprintInputs.parser).toBe("taguru-text-connector");
    expect([...store.getCalls.entries()]).toEqual([["data", 1]]);
  });

  test("missing extension and unresolvable content type is unsupported", async () => {
    const store = new FakeObjectStore();
    store.put("data", enc("\x00\x01binary"), { contentType: "application/octet-stream" });
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.diagnostics[0]!.code).toBe("unsupported_format");
    expect([...store.getCalls.entries()]).toEqual([["data", 1]]); // fetched once, never again
  });

  test("size cap still applies on the content-type fallback fetch", async () => {
    const store = new FakeObjectStore("reports");
    store.put("data", new Uint8Array(100).fill(0x78), { contentType: "application/octet-stream" });
    const connector = new S3Connector(store, { maxFileBytes: 10 });

    const document = await connector.read("s3://reports/data");

    expect(document.diagnostics[0]!.code).toBe("content_too_large");
  });

  test("content too large is reported without any fetch", async () => {
    const store = new FakeObjectStore();
    store.put("big.txt", new Uint8Array(100).fill(0x78));
    const connector = new S3Connector(store, { maxFileBytes: 10 });

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.diagnostics[0]!.code).toBe("content_too_large");
    expect(store.getCalls.size).toBe(0);
  });

  test("oversized source id is reported without any fetch", async () => {
    const store = new FakeObjectStore("b");
    const longKey = "x".repeat(MAX_NAME_BYTES + 10) + ".txt";
    store.put(longKey, enc("hello"));
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.diagnostics[0]!.code).toBe("source_id_too_long");
    expect(store.getCalls.size).toBe(0);
  });

  test("object tags are mapped onto metadata sorted", async () => {
    const store = new FakeObjectStore();
    store.put("a.md", enc("hello"), { tags: { team: "", project: "quarterly" } });
    const connector = new S3Connector(store);

    const document = await connector.readObject(await firstMeta(store, ""));

    expect(document.metadata.tags).toEqual(["project=quarterly", "team"]);
  });

  test("excess tags are dropped and counted", async () => {
    const store = new FakeObjectStore();
    const tags: Record<string, string> = {};
    for (let i = 0; i < 35; i += 1) {
      tags[`tag${String(i).padStart(3, "0")}`] = "v";
    }
    store.put("a.md", enc("hello"), { tags });
    const connector = new S3Connector(store);

    const result: ReadResult = await connector.readObjectResult(await firstMeta(store, ""));

    expect(result.document.metadata.tags.length).toBe(32);
    expect(result.tagsDropped).toBe(3);
  });

  test("oversized tag value is dropped and counted", async () => {
    const store = new FakeObjectStore();
    store.put("a.md", enc("hello"), { tags: { ok: "fine", huge: "v".repeat(MAX_TAG_BYTES) } });
    const connector = new S3Connector(store);

    const result = await connector.readObjectResult(await firstMeta(store, ""));

    expect(result.document.metadata.tags).toEqual(["ok=fine"]);
    expect(result.tagsDropped).toBe(1);
  });

  test("read via the plain connector protocol entry point", async () => {
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("hello"));
    const connector = new S3Connector(store);

    expect(connector.supports("s3://reports/a.md")).toBe(true);
    expect(connector.supports("s3://other-bucket/a.md")).toBe(false);
    const document = await connector.read("s3://reports/a.md");
    expect(document.source).toBe("s3://reports/a.md");
  });

  test("read enforces max file bytes even without a listing", async () => {
    const store = new FakeObjectStore("reports");
    store.put("big.txt", new Uint8Array(100).fill(0x78));
    const connector = new S3Connector(store, { maxFileBytes: 10 });

    const document = await connector.read("s3://reports/big.txt");

    expect(document.diagnostics[0]!.code).toBe("content_too_large");
  });

  test("parse options digest is stable and reflects config", async () => {
    const store = new FakeObjectStore();
    const a = new S3Connector(store, { maxFileBytes: 1024 });
    const b = new S3Connector(store, { maxFileBytes: 1024 });
    const c = new S3Connector(store, { maxFileBytes: 2048 });
    expect(await a.parseOptionsDigest()).toBe(await b.parseOptionsDigest());
    expect(await a.parseOptionsDigest()).not.toBe(await c.parseOptionsDigest());
  });
});

// ---------------------------------------------------------------------------
// syncObjectStorage — orchestration: checkpoint layers, deletion, dryRun
// ---------------------------------------------------------------------------

describe("syncObjectStorage", () => {
  test("a full pass imports every discovered object", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"));
    store.put("b.md", enc("beta content"));
    const checkpoints = new RecordingCheckpointStore();

    const report = await syncObjectStorage(store, "", { ingester: ingester(server), checkpoints });

    // Counts tally each source's LAST phase only (ADR 0007 §11) — a source
    // that reached "imported" never counts under "discovered" too.
    expect(report.imported).toBe(2);
    expect(report.discovered).toBe(0);
    expect(report.failed).toBe(0);
    expect(server.imported.length).toBe(2);
  });

  test("a second pass with no change skips the fetch entirely", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"), { etag: "e1" });
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    expect([...store.getCalls.entries()]).toEqual([["a.md", 1]]);

    const report = await syncObjectStorage(store, "", { ingester: ing, checkpoints });

    expect(report.unchanged).toBe(1);
    expect(report.imported).toBe(0);
    expect([...store.getCalls.entries()]).toEqual([["a.md", 1]]); // no second fetch at all
    expect(server.imported.length).toBe(1); // no second import either
  });

  test("a metadata-only change skips the re-import but not the fetch", async () => {
    // A listing-metadata bump (e.g. a tag/touch) with byte-identical
    // content re-fetches (Layer 1 misses) but does not re-ingest (Layer 2,
    // ConnectorCheckpoint, still hits) — ADR 0007 §6.3.
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"), { etag: "e1" });
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    store.put("a.md", enc("alpha content"), { etag: "e2" }); // same bytes, new etag
    const report = await syncObjectStorage(store, "", { ingester: ing, checkpoints });

    expect(report.unchanged).toBe(1);
    expect(store.getCalls.get("a.md")).toBe(2); // re-fetched
    expect(server.imported.length).toBe(1); // but not re-imported
  });

  test("genuinely changed content is re-imported", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"), { etag: "e1" });
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    store.put("a.md", enc("alpha content, revised"), { etag: "e2" });
    const report = await syncObjectStorage(store, "", { ingester: ing, checkpoints });

    expect(report.imported).toBe(1);
    expect(server.imported.length).toBe(2);
  });

  test("deletion default policy only reports, never retracts", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    store.put("b.md", enc("beta"));
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    store.delete("b.md");
    const report = await syncObjectStorage(store, "", { ingester: ing, checkpoints });

    expect(report.deletedDetected).toBe(1);
    expect(report.retracted).toBe(0);
    expect(server.retracted).toEqual([]);
  });

  test("deletion retract policy withdraws only the detected deletion", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    store.put("b.md", enc("beta"));
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    store.delete("b.md");
    const report = await syncObjectStorage(store, "", {
      ingester: ing,
      checkpoints,
      deletionPolicy: "retract",
    });

    expect(report.deletedDetected).toBe(1);
    expect(report.retracted).toBe(1);
    expect(server.retracted).toEqual(["s3://reports/b.md"]);
  });

  test("retract clears both checkpoints so a recreated object is reimported", async () => {
    // Without clearing S3ObjectCheckpoint/ConnectorCheckpoint on retract, a
    // non-versioned bucket's object deleted then recreated elsewhere with
    // the SAME (size, lastModified) — the fingerprint tier that
    // combination falls back to — would read as "unchanged" on the very
    // next pass and never be re-ingested, even though the server no longer
    // has its passage at all.
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    const sameWhen = new Date("2026-01-01T00:00:00Z");
    store.put("a.md", enc("alpha"), { lastModified: sameWhen });
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    store.delete("a.md");
    await syncObjectStorage(store, "", { ingester: ing, checkpoints, deletionPolicy: "retract" });
    expect(server.retracted).toEqual(["s3://reports/a.md"]);

    // Recreated with the exact same size and lastModified a non-versioned
    // bucket's fingerprint falls back to — the false-"unchanged" scenario.
    store.put("a.md", enc("alpha"), { lastModified: sameWhen });
    const report = await syncObjectStorage(store, "", { ingester: ing, checkpoints });

    expect(report.imported).toBe(1);
    expect(server.imported.length).toBe(2);
  });

  test("deletion mirror policy self-heals even without a prior inventory", async () => {
    // "mirror" reconciles against the server's own source list under this
    // prefix even on the FIRST pass, when no inventory has ever been
    // written yet — a phantom source the connector's own listing never
    // even mentions still gets retracted (ADR 0007 §9).
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    server.sources = ["s3://reports/a.md", "s3://reports/phantom.md"];
    const checkpoints = new RecordingCheckpointStore();

    const report = await syncObjectStorage(store, "", {
      ingester: ingester(server),
      checkpoints,
      deletionPolicy: "mirror",
    });

    expect(report.retracted).toBe(1);
    expect(server.retracted).toEqual(["s3://reports/phantom.md"]);
  });

  // No test mirrors the Python original's
  // test_retract_without_a_sync_client_raises_instead_of_guessing — see
  // this file's own module doc comment.

  test("eventsOut is closed even when the pass raises", async () => {
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    const checkpoints = new RecordingCheckpointStore();
    store.get = async () => {
      throw new Error("simulated unexpected failure");
    };
    const closeSpy = vi.spyOn(RunRecorder.prototype, "close");
    const dir = mkdtempSync(join(tmpdir(), "taguru-s3-events-raise-"));
    try {
      await expect(
        syncObjectStorage(store, "", {
          ingester: ingester(new FakeServer()),
          checkpoints,
          eventsOut: join(dir, "events.jsonl"),
        }),
      ).rejects.toThrow("simulated unexpected failure");
      expect(closeSpy).toHaveBeenCalled();
    } finally {
      closeSpy.mockRestore();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("dry run touches nothing", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    const checkpoints = new RecordingCheckpointStore();

    const report = await syncObjectStorage(store, "", {
      ingester: ingester(server),
      checkpoints,
      dryRun: true,
    });

    expect(report.parsed).toBe(1); // "would fetch/parse," ADR 0007 §11
    expect(store.getCalls.size).toBe(0);
    expect(server.imported).toEqual([]);
    expect(checkpoints.state.size).toBe(0); // no checkpoint and no inventory written
  });

  test("dry run reports unchanged when the listing metadata already matches", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"), { etag: "e1" });
    const checkpoints = new RecordingCheckpointStore();
    const ing = ingester(server);

    await syncObjectStorage(store, "", { ingester: ing, checkpoints });
    const report = await syncObjectStorage(store, "", { ingester: ing, checkpoints, dryRun: true });

    expect(report.unchanged).toBe(1);
    expect(report.parsed).toBe(0);
  });

  test("a permanent fetch failure fails only that object", async () => {
    // Only the FIRST object listed ("a.md", alphabetically first) should
    // fail — the real (unpatched) FakeObjectStore.get behavior backs every
    // later call, proving one object's permanent failure doesn't abort the
    // rest of the enumeration.
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    store.put("b.md", enc("beta"));
    const checkpoints = new RecordingCheckpointStore();
    const realGet = store.get.bind(store);
    let calls = 0;
    store.get = async (key, options) => {
      calls += 1;
      if (calls === 1) {
        throw new PermanentStoreError("access denied");
      }
      return realGet(key, options);
    };

    const report = await syncObjectStorage(store, "", { ingester: ingester(server), checkpoints });

    expect(report.failed).toBe(1);
    expect(report.imported).toBe(1);
  });

  test("a vanished object is skipped, not failed", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    const checkpoints = new RecordingCheckpointStore();
    store.get = async () => {
      throw new ObjectNotFoundError("gone");
    };

    const report = await syncObjectStorage(store, "", { ingester: ingester(server), checkpoints });

    expect(report.skipped).toBe(1);
    expect(report.failed).toBe(0);
  });

  test("a listing failure fails the pass and never writes the inventory", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    store.failListWith = new TransientStoreError("timeout");
    const checkpoints = new RecordingCheckpointStore();

    const report = await syncObjectStorage(store, "", { ingester: ingester(server), checkpoints });

    expect(report.failed).toBe(1);
    expect([...checkpoints.state.keys()].some((key) => key.startsWith("s3-inventory:"))).toBe(false);
  });

  test("should_stop before any fetch still discovers everything and writes the inventory", async () => {
    const server = new FakeServer();
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha"));
    store.put("b.md", enc("beta"));
    const checkpoints = new RecordingCheckpointStore();

    const report = await syncObjectStorage(store, "", {
      ingester: ingester(server),
      checkpoints,
      shouldStop: () => true,
    });

    expect(report.interrupted).toBe(true);
    expect(report.discovered).toBe(2);
    expect(report.imported).toBe(0);
    expect(store.getCalls.size).toBe(0); // every discover landed before the first fetch
    expect([...checkpoints.state.keys()].some((key) => key.startsWith("s3-inventory:"))).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // Cross-connector observability (ADR 0007 §11, issue #353) — the pieces
  // S3's own report shares with syncReferences's.
  // ---------------------------------------------------------------------------

  test("eventsOut writes the same shape as the returned events", async () => {
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"));
    const dir = mkdtempSync(join(tmpdir(), "taguru-s3-events-"));
    try {
      const eventsPath = join(dir, "events.jsonl");
      const report = await syncObjectStorage(store, "", {
        ingester: ingester(new FakeServer()),
        checkpoints: new RecordingCheckpointStore(),
        eventsOut: eventsPath,
      });

      const onDisk = readFileSync(eventsPath, "utf-8")
        .split("\n")
        .filter((line) => line.length > 0)
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      const fromReport = report.events.map((event) => event.toDict());
      expect(onDisk).toEqual(fromReport);
      expect(report.eventsPath).toBe(eventsPath);
      const phases = onDisk
        .filter((line) => line["source"] === "s3://reports/a.md")
        .map((line) => line["phase"]);
      expect(phases).toEqual(["discovered", "parsed", "extracted", "imported"]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("duration_ms is non-negative on a real run", async () => {
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"));
    const report = await syncObjectStorage(store, "", {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
    });
    expect(report.durationMs).toBeGreaterThanOrEqual(0.0);
  });

  test("parsed bytes is the object's own size, not the parsed text's", async () => {
    const store = new FakeObjectStore("reports");
    const body = enc("alpha content, deliberately longer than the parsed text will be");
    store.put("a.md", body);
    const report = await syncObjectStorage(store, "", {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
    });
    const parsedEvent = report.events.find((event) => event.phase === "parsed")!;
    expect(parsedEvent.bytes).toBe(body.length);
  });

  test("duplicate listing key does not disturb the winning occurrence's tally", async () => {
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"));

    // Wraps a real FakeObjectStore, yielding its one object TWICE from
    // list() — simulating a store bug/quirk (ADR 0007 §11.1); a real
    // S3-compatible listing should never do this, but the driver must not
    // corrupt its own tally if one does.
    class DuplicateListingStore implements ObjectStore {
      constructor(private readonly inner: typeof store) {}
      get baseUri(): string {
        return this.inner.baseUri;
      }
      async *list(prefix: string): AsyncGenerator<ObjectMeta> {
        const metas: ObjectMeta[] = [];
        for await (const meta of this.inner.list(prefix)) {
          metas.push(meta);
        }
        for (const meta of [...metas, ...metas]) {
          yield meta;
        }
      }
      get(key: string, options?: { versionId?: string }) {
        return this.inner.get(key, options);
      }
      objectTags(key: string) {
        return this.inner.objectTags(key);
      }
    }

    const report = await syncObjectStorage(new DuplicateListingStore(store), "", {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
    });

    expect(report.imported).toBe(1);
    expect(store.getCalls.get("a.md")).toBe(1);
    expect(
      report.events.some((event) => event.diagnostic !== null && event.diagnostic.code === "duplicate_source"),
    ).toBe(true);
  });

  test("locators and sections dropped are surfaced on the report", async () => {
    const server = new FakeServer();
    server.importResultOverride = {
      context: "sake",
      source: "s3://reports/a.md",
      created: true,
      retracted: 0,
      associations: 0,
      aliases: 0,
      passage_stored: true,
      passage_dropped: false,
      questions_stored: 0,
      questions_dropped: 0,
      sections_stored: 0,
      sections_dropped: 2,
      locators_stored: 0,
      locators_dropped: 4,
      association_paragraphs_dropped: 0,
    };
    const store = new FakeObjectStore("reports");
    store.put("a.md", enc("alpha content"));
    const report = await syncObjectStorage(store, "", {
      ingester: ingester(server),
      checkpoints: new RecordingCheckpointStore(),
    });
    expect(report.imported).toBe(1);
    expect(report.locatorsDropped).toBe(4);
    expect(report.sectionsDropped).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// S3ObjectCheckpoint — the cheap "skip the fetch" layer, on its own
// ---------------------------------------------------------------------------

describe("S3ObjectCheckpoint", () => {
  test("round-trips", async () => {
    const checkpoint = new S3ObjectCheckpoint(new RecordingCheckpointStore());
    await checkpoint.save("s3://b/k", ["etag", "abc"]);
    expect(await checkpoint.load("s3://b/k")).toEqual(["etag", "abc"]);
  });

  test("is null when never saved", async () => {
    const checkpoint = new S3ObjectCheckpoint(new RecordingCheckpointStore());
    expect(await checkpoint.load("s3://b/never-seen")).toBeNull();
  });

  test("is null on corrupt bytes", async () => {
    const store = new RecordingCheckpointStore(
      new Map([["s3-object:s3://b/k", new TextEncoder().encode("not json {")]]),
    );
    const checkpoint = new S3ObjectCheckpoint(store);
    expect(await checkpoint.load("s3://b/k")).toBeNull();
  });

  test("delete removes the entry", async () => {
    const checkpoint = new S3ObjectCheckpoint(new RecordingCheckpointStore());
    await checkpoint.save("s3://b/k", ["etag", "abc"]);
    await checkpoint.delete("s3://b/k");
    expect(await checkpoint.load("s3://b/k")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Credential boundary (ADR 0007 §9): a real credential never appears in
// the checkpoint, report, or document bytes. Deliberate TS deviation: the
// Python original builds a real boto3 client (stubbed at the HTTP layer
// with botocore.stub.Stubber) so fake-but-distinctive env-chain
// credentials flow through the SAME code path a real deployment's env vars
// would. `@aws-sdk/client-s3` is an optional peer dependency not installed
// in this workspace (objectstore-s3.ts's own module doc comment), so —
// mirroring gcs-object-store.test.ts/azure-object-store.test.ts's own
// established convention — this proves the property structurally instead:
// `S3ObjectStore`'s `client` escape hatch has no parameter through which a
// credential could flow at all, so a store built from an injected fake
// client (below) cannot leak one regardless of what the environment holds.
// ---------------------------------------------------------------------------

test("credentials never appear in checkpoint, report, or document bytes", async () => {
  const fakeAccessKey = "AKIAFAKEFAKEFAKEFAKE";
  const fakeSecret = "Fake/Secret+Access+Key+Value+Never+Persisted";
  vi.stubEnv("AWS_ACCESS_KEY_ID", fakeAccessKey);
  vi.stubEnv("AWS_SECRET_ACCESS_KEY", fakeSecret);

  try {
    class FakeS3Client implements S3ClientLike {
      async bucketVersioningEnabled(): Promise<boolean> {
        return false;
      }
      async *listObjectsV2(_bucket: string, prefix: string): AsyncGenerator<S3ObjectLike> {
        if ("a.md".startsWith(prefix)) {
          yield { key: "a.md", size: 11, lastModified: null, etag: '"abc"' };
        }
      }
      async *listObjectVersions(): AsyncGenerator<S3ObjectLike> {
        return;
      }
      async getObject(_bucket: string, key: string): Promise<{ body: Uint8Array; contentType: string | null }> {
        if (key !== "a.md") {
          throw new ObjectNotFoundError(`${key}: no such object`);
        }
        return { body: enc("hello world"), contentType: "text/markdown" };
      }
      async getObjectTagging(): Promise<Record<string, string>> {
        return {};
      }
    }

    const store = new S3ObjectStore("reports", { client: new FakeS3Client() });
    const checkpoints = new RecordingCheckpointStore();
    const server = new FakeServer();

    const report = await syncObjectStorage(store, "", { ingester: ingester(server), checkpoints });

    expect(report.imported).toBe(1);
    const haystacks: string[] = [
      JSON.stringify([...checkpoints.state.entries()].map(([key, value]) => [key, [...value]])),
      JSON.stringify(report.events.map((event) => event.toDict())),
      ...server.imported,
    ];
    for (const secret of [fakeAccessKey, fakeSecret]) {
      for (const haystack of haystacks) {
        expect(haystack).not.toContain(secret);
      }
    }
  } finally {
    vi.unstubAllEnvs();
  }
});
