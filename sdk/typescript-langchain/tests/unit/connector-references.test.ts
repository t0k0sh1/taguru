/**
 * syncReferences (ADR 0007 §11, issue #353): the cross-connector local-
 * file/URL driver — dispatch that never lets a URL fall into a local-path
 * connector's file read, the per-kind `dryRun` table, the enumerate-
 * before-fetch ordering, `duplicate_source`, redirect `retarget`, and
 * interruption/failure reporting — exercised against real
 * `TextFileConnector` and `HtmlConnector` instances, a real
 * `tests/httpd.ts` server for URL fetches, and the same in-memory
 * `FakeServer` every other ingest test in this suite uses. Mirrors the
 * Python suite's test_connector_references.py case-for-case where this
 * SDK's already-ported sibling modules support the same behavior; see
 * references.ts's own module doc for the deliberate deviations (no
 * ImportError-catching `defaultConnectors()`, a single `string` reference
 * type, `RunRecorder.addDropped()` called with no arguments).
 * TypeScript parity: issue #415.
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import * as fsPromises from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { IngestEvent, IngestEventCallback } from "../../src/events.js";
import { TaguruIngester, type IngestOutcome, type TaguruIngesterFields } from "../../src/ingest.js";
import {
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  FingerprintInputs,
} from "../../src/ingest-connectors/document.js";
import { HtmlConnector } from "../../src/ingest-connectors/html.js";
import { RunRecorder } from "../../src/ingest-connectors/observability.js";
import type { Connector } from "../../src/ingest-connectors/protocol.js";
import { defaultConnectors, planReferences, syncReferences } from "../../src/ingest-connectors/references.js";
import { TextFileConnector } from "../../src/ingest-connectors/text.js";
import { type Route, type RouteServer, serve } from "../httpd.js";
import { RecordingCheckpointStore } from "./checkpoints.test.js";
import { FakeServer } from "./stub.js";

// Only `stat` is wrapped (default: forward straight to the real
// implementation) — `test dry-run file vanishing between plan and probe
// reports parsed` below is the one test that reconfigures it, and restores
// the pass-through implementation in its own `finally`. Every other test in
// this file is unaffected.
vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return { ...actual, stat: vi.fn(actual.stat) };
});

const EMPTY_ANSWER = JSON.stringify({ associations: [], aliases: [], questions: [] });

function ingester(
  server: FakeServer,
  overrides: Partial<TaguruIngesterFields> = {},
  responseCount = 50,
): TaguruIngester {
  return new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({ responses: Array(responseCount).fill(EMPTY_ANSWER) as string[] }),
    client: server.client(),
    ...overrides,
  });
}

function write(dirPath: string, name: string, data: string): string {
  const path = join(dirPath, name);
  writeFileSync(path, data, "utf-8");
  return path;
}

async function withServer(
  routes: Record<string, Route>,
  fn: (server: RouteServer) => Promise<void>,
): Promise<void> {
  const server = await serve(routes);
  try {
    await fn(server);
  } finally {
    await server.close();
  }
}

/** Wraps a real connector, recording every `read()` call — used to assert
 * dry-run never calls `read()` at all. */
class SpyConnector implements Connector {
  readonly readCalls: string[] = [];

  constructor(private readonly delegate: Connector) {}

  get parser(): string {
    return this.delegate.parser;
  }

  get parserVersion(): string {
    return this.delegate.parserVersion;
  }

  async parseOptionsDigest(): Promise<string> {
    return this.delegate.parseOptionsDigest();
  }

  supports(reference: string): boolean {
    return this.delegate.supports(reference);
  }

  async read(reference: string): Promise<ConnectorDocument> {
    this.readCalls.push(reference);
    return this.delegate.read(reference);
  }
}

function spyConnectors(): { connectors: readonly Connector[]; spy: SpyConnector } {
  const spy = new SpyConnector(new TextFileConnector());
  return { connectors: [spy, new HtmlConnector()], spy };
}

function makeOutcome(overrides: Partial<IngestOutcome> & { source: string }): IngestOutcome {
  return {
    ok: false,
    ndjson: null,
    created: false,
    retracted: 0,
    associations: 0,
    aliases: 0,
    passage_stored: false,
    questions_stored: 0,
    sections_stored: 0,
    sections_dropped: 0,
    locators_stored: 0,
    locators_dropped: 0,
    duplicates_dropped: 0,
    invalid_dropped: 0,
    llm_calls: 0,
    chunks: 0,
    correction_attempts: 0,
    lossless_repairs: [],
    error: null,
    embeddings_refresh_warning: null,
    interrupted: false,
    chunks_reused: 0,
    ...overrides,
  };
}

/** A minimal stand-in for `TaguruIngester` — `syncReferences` only ever
 * calls `.on_event` (via `RunRecorder.attached`, get/set) and `.ingestText`
 * (via `ingestConnectorDocument`) on it, so a full ingester (LLM, client,
 * checkpoint store) is unnecessary overhead for a test whose only interest
 * is what `syncReferences` does with a controlled `IngestOutcome`. Mirrors
 * the Python suite's `_StubIngester`. */
class StubIngester {
  on_event: IngestEventCallback | undefined;
  raise_on_error = false;
  calls = 0;

  constructor(private readonly outcomeValue: IngestOutcome) {}

  async ingestText(_text: string, _options: unknown): Promise<IngestOutcome> {
    this.calls += 1;
    return this.outcomeValue;
  }
}

/** An ingester whose `ingestText` throws — unlike a connector's own
 * `read()`, `syncReferences` does not wrap the bridge call in a try/catch,
 * so this exercises `RunRecorder` being managed by a `try`/`finally`:
 * `eventsOut`'s file handle must still close. Mirrors the Python suite's
 * `_RaisingIngester`. */
class RaisingIngester {
  on_event: IngestEventCallback | undefined;
  raise_on_error = false;

  async ingestText(_text: string, _options: unknown): Promise<IngestOutcome> {
    throw new Error("ingester exploded");
  }
}

function asIngester(stub: { on_event: IngestEventCallback | undefined }): TaguruIngester {
  return stub as unknown as TaguruIngester;
}

class RaisingConnector implements Connector {
  readonly parser = "raising-connector";
  readonly parserVersion = "1.0.0";

  async parseOptionsDigest(): Promise<string> {
    return "x";
  }

  supports(reference: string): boolean {
    return reference.endsWith(".boom");
  }

  read(_reference: string): Promise<ConnectorDocument> {
    throw new Error("connector bug");
  }
}

class EmptyTextConnector implements Connector {
  readonly parser = "empty";
  readonly parserVersion = "1.0.0";

  async parseOptionsDigest(): Promise<string> {
    return "x";
  }

  supports(_reference: string): boolean {
    return true;
  }

  async read(reference: string): Promise<ConnectorDocument> {
    return new ConnectorDocument({
      source: reference,
      text: "",
      metadata: new ConnectorMetadata({ originUri: reference, displayName: reference }),
      fingerprintInputs: new FingerprintInputs({
        rawContentSha256: "x",
        parser: this.parser,
        parserVersion: this.parserVersion,
        parseOptionsDigest: "x",
      }),
      diagnostics: [new Diagnostic({ code: "ocr_required", message: "no text", source: reference })],
    });
  }
}

describe("syncReferences / planReferences (ADR 0007 §11, issue #353)", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "taguru-references-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  // -------------------------------------------------------------------------
  // planReferences — classification and dispatch, before any fetch
  // -------------------------------------------------------------------------

  describe("planReferences", () => {
    it("local extension dispatches to the text connector", async () => {
      const reference = write(dir, "a.md", "paragraph one.");
      const plans = await planReferences([reference], { connectors: defaultConnectors() });
      expect(plans).toHaveLength(1);
      expect(plans[0]!.connector).toBeInstanceOf(TextFileConnector);
      expect(plans[0]!.kind).toBe("path");
      expect(plans[0]!.diagnostic).toBeNull();
    });

    it(
      "a markdown-shaped url is never claimed by the text connector (the bug this module " +
        "exists to prevent: TextFileConnector.supports checks only the suffix)",
      async () => {
        const plans = await planReferences(["https://example.com/readme.md"], {
          connectors: defaultConnectors(),
        });
        expect(plans).toHaveLength(1);
        expect(plans[0]!.kind).toBe("url");
        expect(plans[0]!.connector).toBeInstanceOf(HtmlConnector);
        expect(plans[0]!.connector).not.toBeInstanceOf(TextFileConnector);
      },
    );

    it("object storage scheme is reported unsupported naming the other driver", async () => {
      const plans = await planReferences(["s3://bucket/key.pdf", "file:///tmp/x.pdf"], {
        connectors: defaultConnectors(),
      });
      expect(plans).toHaveLength(2);
      for (const plan of plans) {
        expect(plan.connector).toBeNull();
        expect(plan.diagnostic?.code).toBe("unsupported_format");
        expect(plan.diagnostic?.message).toContain("syncObjectStorage");
      }
    });

    it("unrecognized local extension is unsupported_format", async () => {
      const reference = write(dir, "a.exe", "binary-ish");
      const plans = await planReferences([reference], { connectors: defaultConnectors() });
      expect(plans[0]!.connector).toBeNull();
      expect(plans[0]!.diagnostic?.code).toBe("unsupported_format");
    });

    it("missing file is unreadable", async () => {
      const plans = await planReferences([join(dir, "never-existed.md")], {
        connectors: defaultConnectors(),
      });
      expect(plans[0]!.connector).toBeNull();
      expect(plans[0]!.diagnostic?.code).toBe("unreadable");
    });

    it("a directory reference is unreadable", async () => {
      const plans = await planReferences([dir], { connectors: defaultConnectors() });
      expect(plans[0]!.connector).toBeNull();
      expect(plans[0]!.diagnostic?.code).toBe("unreadable");
    });

    it("duplicate reference is flagged and the first keeps its connector", async () => {
      const reference = write(dir, "a.md", "paragraph one.");
      const plans = await planReferences([reference, reference], { connectors: defaultConnectors() });
      expect(plans).toHaveLength(2);
      expect(plans[0]!.connector).not.toBeNull();
      expect(plans[0]!.diagnostic).toBeNull();
      expect(plans[1]!.connector).toBeNull();
      expect(plans[1]!.diagnostic?.code).toBe("duplicate_source");
    });

    it("two urls differing only by a denylisted query param are duplicates", async () => {
      const plans = await planReferences(
        [
          "https://example.com/report.html?token=abc",
          "https://example.com/report.html?token=xyz",
        ],
        { connectors: defaultConnectors() },
      );
      expect(plans[1]!.diagnostic?.code).toBe("duplicate_source");
    });

    it("source_bytes is the file's own size", async () => {
      const reference = write(dir, "a.md", "twelve bytes");
      const plans = await planReferences([reference], { connectors: defaultConnectors() });
      expect(plans[0]!.sourceBytes).toBe(Buffer.byteLength("twelve bytes"));
    });

    it("source_bytes is zero for a url", async () => {
      const plans = await planReferences(["https://example.com/a.html"], {
        connectors: defaultConnectors(),
      });
      expect(plans[0]!.sourceBytes).toBe(0);
    });
  });

  // -------------------------------------------------------------------------
  // dry_run — per-kind rules (ADR 0007 §11)
  // -------------------------------------------------------------------------

  describe("dry run — per-kind rules", () => {
    it("local file with no probe reports parsed and never reads", async () => {
      const reference = write(dir, "a.md", "paragraph one.");
      const { connectors, spy } = spyConnectors();
      const checkpoints = new RecordingCheckpointStore();
      const report = await syncReferences([reference], {
        ingester: ingester(new FakeServer()),
        checkpoints,
        connectors,
        dryRun: true,
      });
      expect(report.parsed).toBe(1);
      expect(report.unchanged).toBe(0);
      expect(spy.readCalls).toEqual([]);
      // A dry-run probe LOOKUP is expected (that's how "unchanged" gets
      // decided at all) — the "touches nothing" contract is about never
      // WRITING to the checkpoint store under dry-run.
      expect(checkpoints.log).not.toContainEqual(["save", `file-probe:${reference}`]);
      expect(checkpoints.log).not.toContainEqual(["save", `connector:${reference}`]);
    });

    it("local file matching probe reports unchanged", async () => {
      const reference = write(dir, "a.md", "paragraph one.");
      const { connectors, spy } = spyConnectors();
      const checkpoints = new RecordingCheckpointStore();

      // A real (non-dry) run first, to populate the file-probe checkpoint.
      await syncReferences([reference], { ingester: ingester(new FakeServer()), checkpoints, connectors });
      spy.readCalls.length = 0;

      const report = await syncReferences([reference], {
        ingester: ingester(new FakeServer()),
        checkpoints,
        connectors,
        dryRun: true,
      });
      expect(report.unchanged).toBe(1);
      expect(report.parsed).toBe(0);
      expect(spy.readCalls).toEqual([]);
    });

    it("local file with stale mtime reports parsed", async () => {
      const reference = write(dir, "a.md", "paragraph one.");
      const { connectors, spy } = spyConnectors();
      const checkpoints = new RecordingCheckpointStore();
      await syncReferences([reference], { ingester: ingester(new FakeServer()), checkpoints, connectors });
      spy.readCalls.length = 0;

      // Touch the file with new content — a different size, so the probe
      // can never match by accident.
      writeFileSync(reference, "paragraph one, edited.", "utf-8");

      const report = await syncReferences([reference], {
        ingester: ingester(new FakeServer()),
        checkpoints,
        connectors,
        dryRun: true,
      });
      expect(report.parsed).toBe(1);
      expect(report.unchanged).toBe(0);
      expect(spy.readCalls).toEqual([]);
    });

    it("local file with changed options digest reports parsed", async () => {
      const reference = write(dir, "a.md", "paragraph one.");
      const checkpoints = new RecordingCheckpointStore();
      await syncReferences([reference], {
        ingester: ingester(new FakeServer()),
        checkpoints,
        connectors: [new TextFileConnector({ extractHeadings: true }), new HtmlConnector()],
      });
      const report = await syncReferences([reference], {
        ingester: ingester(new FakeServer()),
        checkpoints,
        connectors: [new TextFileConnector({ extractHeadings: false }), new HtmlConnector()],
        dryRun: true,
      });
      expect(report.parsed).toBe(1);
      expect(report.unchanged).toBe(0);
    });

    it(
      "file vanishing between plan and probe reports parsed (the narrow TOCTOU " +
        "dryRunPlan's own catch guards: the file existed when planReferences stat'd it, but " +
        "is gone by the time the dry-run probe stats it again a moment later)",
      async () => {
        const reference = write(dir, "a.md", "paragraph one.");
        const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
        let calls = 0;
        const statMock = fsPromises.stat as unknown as {
          mockImplementation(fn: (...args: unknown[]) => unknown): void;
        };
        statMock.mockImplementation(async (...args: unknown[]) => {
          calls += 1;
          if (calls > 1) {
            throw new Error("ENOENT: simulated vanish");
          }
          return actual.stat(args[0] as string, args[1] as { bigint: true });
        });
        try {
          const report = await syncReferences([reference], {
            ingester: ingester(new FakeServer()),
            checkpoints: new RecordingCheckpointStore(),
            dryRun: true,
          });
          expect(report.parsed).toBe(1);
        } finally {
          statMock.mockImplementation(async (...args: unknown[]) =>
            actual.stat(args[0] as string, args[1] as { bigint: true }),
          );
        }
      },
    );

    it("url reports parsed and makes no network request", async () => {
      // Port 1 refuses connections immediately on any sane host — if
      // syncReferences ever attempted a fetch under dryRun, this would
      // reject instead of quietly reporting `parsed`.
      const report = await syncReferences(["http://127.0.0.1:1/unreachable.html"], {
        ingester: ingester(new FakeServer()),
        checkpoints: new RecordingCheckpointStore(),
        dryRun: true,
      });
      expect(report.parsed).toBe(1);
      expect(report.unchanged).toBe(0);
      expect(report.failed).toBe(0);
    });

    it(
      "vanished file is skipped unreadable, not a crash (planReferences itself stats every " +
        "path reference, dry-run or not — a file that vanished before the plan is built is " +
        "caught there, never reaching dryRunPlan's own narrower race at all)",
      async () => {
        const reference = write(dir, "a.md", "paragraph one.");
        const checkpoints = new RecordingCheckpointStore();
        await syncReferences([reference], { ingester: ingester(new FakeServer()), checkpoints });
        rmSync(reference);
        const report = await syncReferences([reference], {
          ingester: ingester(new FakeServer()),
          checkpoints,
          dryRun: true,
        });
        expect(report.skipped).toBe(1);
        const skipped = report.events.find((event) => event.phase === "skipped")!;
        expect(skipped.diagnostic?.code).toBe("unreadable");
      },
    );
  });

  // -------------------------------------------------------------------------
  // Enumeration before the first fetch (ADR 0007 §11)
  // -------------------------------------------------------------------------

  it("every discovered event lands before the first read", async () => {
    const references = [0, 1, 2].map((i) => write(dir, `${i}.md`, `paragraph ${i}.`));
    const { connectors } = spyConnectors();
    const report = await syncReferences(references, {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
      connectors,
    });
    expect(report.imported).toBe(3);
    const discoveredEvents = report.events.filter((event) => event.phase === "discovered");
    expect(discoveredEvents).toHaveLength(3);
    // All three `discovered` events precede all three `read()` calls — true
    // by construction (planReferences + the discover loop run to
    // completion before the process loop starts), asserted here via the
    // event ordering: every `discovered` line for EVERY source appears
    // before the first non-`discovered` line for ANY of them.
    const firstNonDiscoveredIndex = report.events.findIndex((event) => event.phase !== "discovered");
    expect(firstNonDiscoveredIndex).toBe(3);
  });

  // -------------------------------------------------------------------------
  // Real run, checkpoint reuse, duplicate handling, redirects
  // -------------------------------------------------------------------------

  it("real run then rerun is unchanged then dry run is also unchanged", async () => {
    const reference = write(dir, "a.md", "paragraph one.");
    const checkpoints = new RecordingCheckpointStore();

    const first = await syncReferences([reference], { ingester: ingester(new FakeServer()), checkpoints });
    expect(first.imported).toBe(1);

    const second = await syncReferences([reference], { ingester: ingester(new FakeServer()), checkpoints });
    expect(second.unchanged).toBe(1);
    expect(second.imported).toBe(0);

    const third = await syncReferences([reference], {
      ingester: ingester(new FakeServer()),
      checkpoints,
      dryRun: true,
    });
    expect(third.unchanged).toBe(1);
  });

  it("duplicate reference is read exactly once", async () => {
    const reference = write(dir, "a.md", "paragraph one.");
    const { connectors, spy } = spyConnectors();
    const report = await syncReferences([reference, reference], {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
      connectors,
    });
    expect(spy.readCalls).toEqual([reference]);
    // The winning occurrence's own tally is undisturbed by the duplicate.
    expect(report.imported).toBe(1);
    expect(report.events.some((event) => event.diagnostic?.code === "duplicate_source")).toBe(true);
  });

  it("redirected url retargets without double counting", async () => {
    await withServer(
      {
        "/old.html": { location: "/new.html" },
        "/new.html": { body: Buffer.from("<html><body><p>paragraph one.</p></body></html>") },
      },
      async (server) => {
        const report = await syncReferences([`${server.baseUrl}/old.html`], {
          ingester: ingester(new FakeServer()),
          checkpoints: new RecordingCheckpointStore(),
          // The test server is on 127.0.0.1 — HtmlConnector's default SSRF
          // guard blocks private/internal destinations, so this test
          // explicitly opts back in, same as html-connector-fetch.test.ts.
          connectors: [new TextFileConnector(), new HtmlConnector({ allowPrivateNetworks: true })],
        });
        expect(report.imported).toBe(1);
        expect(report.discovered).toBe(0);
        expect(report.failed).toBe(0);
        // `discovered` legitimately kept the PRE-redirect URL (that's the
        // honest history retarget() preserves in the JSONL trail) — but
        // every later phase, and the tally itself, moved to the
        // POST-redirect one.
        const byPhase = new Map(report.events.map((event) => [event.phase, event.source]));
        expect(byPhase.get("discovered")).toBe(`${server.baseUrl}/old.html`);
        expect(byPhase.get("imported")).toBe(`${server.baseUrl}/new.html`);
      },
    );
  });

  it(
    "two urls redirecting to the same final url do not corrupt the tally (the post-fetch " +
      "twin of the duplicate-reference test above — two DIFFERENT pre-fetch references, so " +
      "planReferences's own pre-fetch dedup cannot catch this)",
    async () => {
      await withServer(
        {
          "/old-a.html": { location: "/new.html" },
          "/old-b.html": { location: "/new.html" },
          "/new.html": { body: Buffer.from("<html><body><p>paragraph one.</p></body></html>") },
        },
        async (server) => {
          const report = await syncReferences(
            [`${server.baseUrl}/old-a.html`, `${server.baseUrl}/old-b.html`],
            {
              ingester: ingester(new FakeServer()),
              checkpoints: new RecordingCheckpointStore(),
              connectors: [new TextFileConnector(), new HtmlConnector({ allowPrivateNetworks: true })],
            },
          );
          const newUrl = `${server.baseUrl}/new.html`;

          // The real outcome survives: exactly one import, under the shared
          // final URL — never silently downgraded to unchanged/discovered
          // by the second reference's own retarget.
          expect(report.imported).toBe(1);
          expect(report.discovered).toBe(0);
          expect(report.unchanged).toBe(0);
          expect(report.failed).toBe(0);

          const eventsBySource = new Map<string, string[]>();
          for (const event of report.events) {
            const phases = eventsBySource.get(event.source) ?? [];
            phases.push(event.phase);
            eventsBySource.set(event.source, phases);
          }
          expect(eventsBySource.get(newUrl)?.at(-1)).toBe("imported");
          // The second (losing) reference is visible as a duplicate of the
          // winning final URL, not as a second competing history under it.
          const duplicateEvents = report.events.filter(
            (event) => event.diagnostic?.code === "duplicate_source",
          );
          expect(duplicateEvents).toHaveLength(1);
          expect(duplicateEvents[0]!.diagnostic!.message).toContain(newUrl);
        },
      );
    },
  );

  // -------------------------------------------------------------------------
  // should_stop, ingest failure, connector-raised exceptions
  // -------------------------------------------------------------------------

  it(
    "should_stop interrupts and leaves the tail discovered (the stop signal must fire " +
      "strictly AFTER the first document's own ingest completes)",
    async () => {
      const references = [0, 1, 2].map((i) => write(dir, `${i}.md`, `paragraph ${i}.`));
      const controller = new AbortController();
      const ing = ingester(new FakeServer(), {
        on_event: (event: IngestEvent) => {
          if (event.kind === "import_completed") {
            controller.abort();
          }
        },
      });
      const report = await syncReferences(references, {
        ingester: ing,
        checkpoints: new RecordingCheckpointStore(),
        shouldStop: controller.signal,
      });
      expect(report.interrupted).toBe(true);
      expect(report.imported).toBe(1);
      expect(report.discovered).toBe(2);
    },
  );

  it("ingest failure is reported failed and the run continues", async () => {
    const a = write(dir, "a.md", "paragraph one.");
    const b = write(dir, "b.md", "paragraph two.");
    const stub = new StubIngester(
      makeOutcome({ source: "doesn't matter", ok: false, error: "model exploded" }),
    );
    const report = await syncReferences([a, b], {
      ingester: asIngester(stub),
      checkpoints: new RecordingCheckpointStore(),
    });
    expect(report.failed).toBe(2);
    expect(report.imported).toBe(0);
    expect(stub.calls).toBe(2);
    const failedEvents = report.events.filter((event) => event.phase === "failed");
    expect(failedEvents.every((event) => event.diagnostic?.message.includes("model exploded"))).toBe(
      true,
    );
  });

  it("events_out is closed even when ingest raises", async () => {
    const reference = write(dir, "a.md", "paragraph one.");
    const eventsPath = join(dir, "events.jsonl");
    const closeSpy = vi.spyOn(RunRecorder.prototype, "close");
    try {
      await expect(
        syncReferences([reference], {
          ingester: asIngester(new RaisingIngester()),
          checkpoints: new RecordingCheckpointStore(),
          eventsOut: eventsPath,
        }),
      ).rejects.toThrow("ingester exploded");
      expect(closeSpy).toHaveBeenCalled();
    } finally {
      closeSpy.mockRestore();
    }
  });

  it("a connector that raises is reported failed not propagated", async () => {
    const reference = join(dir, "a.boom");
    writeFileSync(reference, "irrelevant", "utf-8");
    const report = await syncReferences([reference], {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
      connectors: [new RaisingConnector()],
    });
    expect(report.failed).toBe(1);
    const failedEvents = report.events.filter((event) => event.phase === "failed");
    expect(failedEvents[0]!.diagnostic?.message).toContain("connector bug");
  });

  it(
    "a diagnostics-only document is reported failed (never reaches the network — bridge.ts's " +
      "own posture — and this driver reports it failed, not skipped, since the object WAS " +
      "read; it simply had nothing usable in it)",
    async () => {
      const reference = join(dir, "scan.pdf");
      writeFileSync(reference, "irrelevant", "utf-8");
      const report = await syncReferences([reference], {
        ingester: asIngester(new StubIngester(makeOutcome({ source: reference, ok: true }))),
        checkpoints: new RecordingCheckpointStore(),
        connectors: [new EmptyTextConnector()],
      });
      expect(report.failed).toBe(1);
      expect(report.imported).toBe(0);
    },
  );

  // -------------------------------------------------------------------------
  // `extracted` — exactly once per imported source
  // -------------------------------------------------------------------------

  it("extracted is recorded once per imported source", async () => {
    const reference = write(dir, "a.md", "paragraph one.");
    const report = await syncReferences([reference], {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
    });
    expect(report.imported).toBe(1);
    const extractedEvents = report.events.filter((event) => event.phase === "extracted");
    expect(extractedEvents).toHaveLength(1);
    expect(extractedEvents[0]!.source).toBe(reference);
  });

  it("a preexisting on_event callback still fires during sync", async () => {
    const reference = write(dir, "a.md", "paragraph one.");
    const seen: string[] = [];
    const ing = ingester(new FakeServer(), { on_event: (event: IngestEvent) => seen.push(event.kind) });
    await syncReferences([reference], { ingester: ing, checkpoints: new RecordingCheckpointStore() });
    expect(seen).toContain("import_started");
  });

  // -------------------------------------------------------------------------
  // events_out — the JSONL sidecar end to end
  // -------------------------------------------------------------------------

  it("events_out writes every phase transition", async () => {
    const reference = write(dir, "a.md", "paragraph one.");
    const eventsPath = join(dir, "events.jsonl");
    const report = await syncReferences([reference], {
      ingester: ingester(new FakeServer()),
      checkpoints: new RecordingCheckpointStore(),
      eventsOut: eventsPath,
    });
    const lines = readFileSync(eventsPath, "utf-8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    const phases = lines.filter((line) => line["source"] === reference).map((line) => line["phase"]);
    expect(phases).toEqual(["discovered", "parsed", "extracted", "imported"]);
    expect(report.eventsPath).toBe(eventsPath);
  });
});
