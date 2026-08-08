/**
 * Machine-readable summary fixture tests (issue #353's own explicit
 * acceptance criterion: "machine-readable summary の fixture テストがある"),
 * ported from the Python twin's test_connector_observability_fixtures.py
 * (TypeScript parity: issue #415).
 *
 * The Python suite pins three golden scenarios — `s3_sync`,
 * `references_dry_run`, `references_run` — produced by
 * `sync_object_storage`/`sync_references`. This file now carries all
 * three (`syncObjectStorage`'s `s3.ts` port and `syncReferences`'s
 * `references.ts` port have both landed in this SDK), alongside the
 * driver-independent machinery this file always carried:
 *
 * - The stable summary key set every golden must share
 *   (`test_summary_key_set_matches_every_golden`).
 * - The normalization/compare machinery
 *   (`_normalize_events_jsonl`/`_normalize_summary`/`_compare_or_write`),
 *   proven against a synthetic run below and reused verbatim by every
 *   golden.
 */

import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { FakeListChatModel } from "@langchain/core/utils/testing";
import { expect, test } from "vitest";

import { TaguruIngester } from "../../src/ingest.js";
import { Diagnostic } from "../../src/ingest-connectors/document.js";
import { HtmlConnector } from "../../src/ingest-connectors/html.js";
import {
  RUN_SUMMARY_KIND,
  RUN_SUMMARY_VERSION,
  RunReport,
  SourceEvent,
} from "../../src/ingest-connectors/observability.js";
import {
  ObjectMeta,
  ObjectNotFoundError,
  type ObjectStore,
  objectFingerprint,
} from "../../src/ingest-connectors/objectstore.js";
import { syncReferences } from "../../src/ingest-connectors/references.js";
import { S3ObjectCheckpoint, syncObjectStorage } from "../../src/ingest-connectors/s3.js";
import { TextFileConnector } from "../../src/ingest-connectors/text.js";
import { type Route, serve } from "../httpd.js";
import { FakeObjectStore } from "../s3.js";
import { RecordingCheckpointStore } from "./checkpoints.test.js";
import { FakeServer } from "./stub.js";

const FIXTURES_ROOT = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "fixtures",
  "ingest-observability",
);
const UPDATE_ENV_VAR = "TAGURU_UPDATE_INGEST_OBSERVABILITY_FIXTURES";

/**
 * Replaces every volatile substring (an absolute temp-directory path, an
 * ephemeral test-server host:port) with a fixed placeholder before writing
 * or comparing a golden — mirrors the Python suite's `_normalize`.
 */
function normalize(text: string, replacements: ReadonlyArray<readonly [string, string]>): string {
  let result = text;
  for (const [from, to] of replacements) {
    result = result.split(from).join(to);
  }
  return result;
}

function sortKeys<T extends Record<string, unknown>>(value: T): T {
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = value[key];
  }
  return sorted as T;
}

/**
 * The TypeScript twin of the Python suite's `_normalize_events_jsonl`:
 * `elapsed_ms` is blanked (different every run), `source` and — inside a
 * present `diagnostic` — both `source` and `message` are run through
 * `replacements` (an OS error or this driver's own `duplicate()` wording
 * can embed a volatile path inside `message` text, not only in a
 * dedicated field), and every line's keys are sorted for a stable diff.
 */
function normalizeEventsJsonl(
  jsonl: string,
  replacements: ReadonlyArray<readonly [string, string]>,
): string {
  const lines = jsonl
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => {
      const event = JSON.parse(line) as Record<string, unknown>;
      event["elapsed_ms"] = 0.0;
      event["source"] = normalize(String(event["source"]), replacements);
      const diagnostic = event["diagnostic"] as Record<string, unknown> | null;
      if (diagnostic !== null) {
        diagnostic["source"] = normalize(String(diagnostic["source"]), replacements);
        diagnostic["message"] = normalize(String(diagnostic["message"]), replacements);
      }
      return JSON.stringify(sortKeys(event));
    });
  return lines.join("\n") + "\n";
}

/** The TypeScript twin of the Python suite's `_normalize_summary`. */
function normalizeSummary(summary: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = { ...summary, duration_ms: 0.0 };
  if (normalized["events_path"] !== null) {
    normalized["events_path"] = "<events>";
  }
  return normalized;
}

/**
 * The TypeScript twin of the Python suite's `_compare_or_write`: under
 * `TAGURU_UPDATE_INGEST_OBSERVABILITY_FIXTURES=1`, (re)writes the golden
 * pair; otherwise asserts the given jsonl/summary match the checked-in
 * golden byte for byte.
 */
function compareOrWrite(name: string, fields: { jsonl: string; summary: Record<string, unknown> }): void {
  const jsonlPath = join(FIXTURES_ROOT, `${name}.jsonl`);
  const summaryPath = join(FIXTURES_ROOT, `${name}.summary.json`);
  const summaryText = JSON.stringify(sortKeys(fields.summary), null, 2) + "\n";

  if (process.env[UPDATE_ENV_VAR]) {
    mkdirSync(FIXTURES_ROOT, { recursive: true });
    writeFileSync(jsonlPath, fields.jsonl, "utf-8");
    writeFileSync(summaryPath, summaryText, "utf-8");
    return;
  }

  expect(existsSync(jsonlPath), `missing golden ${jsonlPath} — regenerate with ${UPDATE_ENV_VAR}=1`).toBe(
    true,
  );
  expect(
    existsSync(summaryPath),
    `missing golden ${summaryPath} — regenerate with ${UPDATE_ENV_VAR}=1`,
  ).toBe(true);
  expect(fields.jsonl).toBe(readFileSync(jsonlPath, "utf-8"));
  expect(summaryText).toBe(readFileSync(summaryPath, "utf-8"));
}

// ---------------------------------------------------------------------------
// The stable summary key set — a literal list, not a derived one, so an
// added/removed key is a deliberate two-place edit (here and the golden).
// ---------------------------------------------------------------------------

const SUMMARY_KEYS = [
  "kind",
  "connector_run",
  "connector",
  "duration_ms",
  "interrupted",
  "discovered",
  "unchanged",
  "parsed",
  "extracted",
  "imported",
  "skipped",
  "failed",
  "locators_dropped",
  "sections_dropped",
  "tags_dropped",
  "deleted_detected",
  "retracted",
  "events",
  "events_path",
].sort();

test("summary key set matches every golden", () => {
  if (!existsSync(FIXTURES_ROOT)) {
    // No TypeScript connector driver has landed a golden yet — see the
    // module doc comment above. Nothing to check until one does.
    return;
  }
  for (const entry of readdirSync(FIXTURES_ROOT)) {
    if (!entry.endsWith(".summary.json")) {
      continue;
    }
    const summary = JSON.parse(readFileSync(join(FIXTURES_ROOT, entry), "utf-8")) as Record<
      string,
      unknown
    >;
    expect(Object.keys(summary).sort(), entry).toEqual(SUMMARY_KEYS);
  }
});

test("RunReport.toDict's own key set matches the pinned list", () => {
  const report = new RunReport();
  expect(Object.keys(report.toDict()).sort()).toEqual(SUMMARY_KEYS);
  expect(report.toDict()["kind"]).toBe(RUN_SUMMARY_KIND);
  expect(report.toDict()["connector_run"]).toBe(RUN_SUMMARY_VERSION);
});

// ---------------------------------------------------------------------------
// Fixture-handling machinery round trip — proves normalizeEventsJsonl /
// normalizeSummary / compareOrWrite work against a synthetic run, so a
// later S3/references port can call compareOrWrite("s3_sync", ...) with
// confidence the harness itself is correct, without needing a real
// connector driver to prove it. The golden pair this writes is a scratch
// fixture, not a checked-in one — removed again at the end of the test.
// ---------------------------------------------------------------------------

test("compareOrWrite regenerates a golden and then matches it byte for byte", () => {
  const events = [
    new SourceEvent({ source: "/tmp/xyz/a.md", phase: "discovered", elapsedMs: 12.3 }),
    new SourceEvent({
      source: "/tmp/xyz/dup.md",
      phase: "skipped",
      elapsedMs: 0.0,
      diagnostic: new Diagnostic({
        code: "duplicate_source",
        message: 'already claimed as "/tmp/xyz/a.md" earlier in this run',
        source: "/tmp/xyz/a.md",
      }),
    }),
    new SourceEvent({ source: "/tmp/xyz/a.md", phase: "imported", elapsedMs: 45.6 }),
  ];
  const report = new RunReport({
    connector: "synthetic-test",
    imported: 1,
    skipped: 0,
    events,
    eventsPath: "/tmp/xyz/events.jsonl",
  });
  const replacements: Array<[string, string]> = [["/tmp/xyz", "<tmp>"]];
  const jsonl = normalizeEventsJsonl(report.eventsJsonl(), replacements);
  const summary = normalizeSummary(report.toDict());

  const previousUpdate = process.env[UPDATE_ENV_VAR];
  const jsonlPath = join(FIXTURES_ROOT, "synthetic.jsonl");
  const summaryPath = join(FIXTURES_ROOT, "synthetic.summary.json");
  try {
    process.env[UPDATE_ENV_VAR] = "1";
    compareOrWrite("synthetic", { jsonl, summary });
    delete process.env[UPDATE_ENV_VAR];
    // Re-running unchanged inputs against the golden just written must
    // match byte for byte — no volatile field slipped through
    // normalization.
    compareOrWrite("synthetic", { jsonl, summary });
    expect(readFileSync(jsonlPath, "utf-8")).not.toContain("/tmp/xyz");
    expect(readFileSync(summaryPath, "utf-8")).not.toContain("/tmp/xyz");
  } finally {
    if (previousUpdate === undefined) {
      delete process.env[UPDATE_ENV_VAR];
    } else {
      process.env[UPDATE_ENV_VAR] = previousUpdate;
    }
    rmSync(jsonlPath, { force: true });
    rmSync(summaryPath, { force: true });
  }
});

const EMPTY_ANSWER = JSON.stringify({ associations: [], aliases: [], questions: [] });

function goldenIngester(server: FakeServer): TaguruIngester {
  return new TaguruIngester({
    context: "sake",
    llm: new FakeListChatModel({ responses: Array(20).fill(EMPTY_ANSWER) as string[] }),
    client: server.client(),
  });
}

// ---------------------------------------------------------------------------
// Golden 1: syncObjectStorage — imported, unchanged (pre-seeded
// fingerprint), failed (unsupported extension), skipped (vanished object),
// a duplicate listing key, and tagsDropped.
// ---------------------------------------------------------------------------

/** Wraps a real `FakeObjectStore`, yielding its `"a.md"` object TWICE from
 * `list()` — simulating a store bug/quirk (ADR 0007 §11.1); a real
 * S3-compatible listing should never do this, but the driver must not
 * corrupt its own tally if one does. Mirrors the Python golden's own
 * `list_with_one_duplicate` monkeypatch, expressed as a wrapper class here
 * since `FakeObjectStore.list` (tests/s3.ts) is an async-generator method,
 * not a reassignable property the way `.get` is. */
class DuplicateListingStore implements ObjectStore {
  constructor(private readonly inner: FakeObjectStore) {}

  get baseUri(): string {
    return this.inner.baseUri;
  }

  async *list(prefix: string): AsyncGenerator<ObjectMeta> {
    const metas: ObjectMeta[] = [];
    for await (const meta of this.inner.list(prefix)) {
      metas.push(meta);
    }
    const duplicate = metas.find((meta) => meta.key === "a.md")!;
    yield duplicate;
    for (const meta of metas) {
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

test("s3_sync golden", async () => {
  const store = new FakeObjectStore("golden-bucket");
  const sameWhen = new Date("2026-01-01T00:00:00Z");

  store.put("a.md", new TextEncoder().encode("alpha content"));
  store.put("b.md", new TextEncoder().encode("beta content"), { lastModified: sameWhen }); // pre-seeded -> unchanged
  store.put("c.zip", new TextEncoder().encode("PK\x03\x04...")); // no installed connector -> failed
  store.put("vanished.md", new TextEncoder().encode("gone by fetch time"));
  const manyTags: Record<string, string> = {};
  for (let i = 0; i < 40; i += 1) {
    manyTags[`t${String(i).padStart(2, "0")}`] = "v";
  }
  store.put("many-tags.md", new TextEncoder().encode("tagged content"), { tags: manyTags });

  const checkpoints = new RecordingCheckpointStore();
  let metaB: ObjectMeta | null = null;
  for await (const meta of store.list("")) {
    if (meta.key === "b.md") {
      metaB = meta;
    }
  }
  const fingerprintB = objectFingerprint(metaB!);
  await new S3ObjectCheckpoint(checkpoints).save(`${store.baseUri}/b.md`, fingerprintB!);

  const realGet = store.get.bind(store);
  store.get = async (key, options) => {
    if (key === "vanished.md") {
      throw new ObjectNotFoundError("gone by fetch time");
    }
    return realGet(key, options);
  };

  const report = await syncObjectStorage(new DuplicateListingStore(store), "", {
    ingester: goldenIngester(new FakeServer()),
    checkpoints,
  });

  const replacements: Array<[string, string]> = [[store.baseUri, "s3://<bucket>"]];
  const jsonl = normalizeEventsJsonl(report.eventsJsonl(), replacements);
  const summary = normalizeSummary(report.toDict());
  compareOrWrite("s3_sync", { jsonl, summary });
});

// ---------------------------------------------------------------------------
// Golden 2: syncReferences({ dryRun: true }) — the per-kind dry-run table:
// local unchanged (matching probe), local parsed (no probe), unsupported
// extension (skipped), missing file (skipped), URL (always parsed, no
// network).
// ---------------------------------------------------------------------------

test("references_dry_run golden", async () => {
  const tmpPath = mkdtempSync(join(tmpdir(), "taguru-references-dry-run-golden-"));
  try {
    const probed = join(tmpPath, "probed.md");
    writeFileSync(probed, "paragraph one.", "utf-8");
    const unprobed = join(tmpPath, "unprobed.md");
    writeFileSync(unprobed, "paragraph two.", "utf-8");
    const unsupported = join(tmpPath, "unsupported.exe");
    writeFileSync(unsupported, "binary-ish", "utf-8");
    const missing = join(tmpPath, "missing.md");

    const checkpoints = new RecordingCheckpointStore();
    // A real run over `probed` alone, to populate its file-probe checkpoint
    // — `unprobed` deliberately never runs for real, so its dry-run stays
    // `parsed`.
    await syncReferences([probed], { ingester: goldenIngester(new FakeServer()), checkpoints });

    const report = await syncReferences(
      [probed, unprobed, unsupported, missing, "http://127.0.0.1:1/unreachable.html"],
      { ingester: goldenIngester(new FakeServer()), checkpoints, dryRun: true },
    );

    const replacements: Array<[string, string]> = [[tmpPath, "<tmp>"]];
    const jsonl = normalizeEventsJsonl(report.eventsJsonl(), replacements);
    const summary = normalizeSummary(report.toDict());
    compareOrWrite("references_dry_run", { jsonl, summary });
  } finally {
    rmSync(tmpPath, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// Golden 3: syncReferences — a real run: an imported local file, a
// duplicate reference of that same file, and a redirected URL (retarget).
// ---------------------------------------------------------------------------

test("references_run golden", async () => {
  const tmpPath = mkdtempSync(join(tmpdir(), "taguru-references-run-golden-"));
  try {
    const reference = join(tmpPath, "manual.md");
    writeFileSync(reference, "paragraph one.", "utf-8");

    const routes: Record<string, Route> = {
      "/old.html": { location: "/new.html" },
      "/new.html": { body: Buffer.from("<html><body><p>paragraph two.</p></body></html>") },
    };
    const server = await serve(routes);
    let report: RunReport;
    let replacements: Array<[string, string]>;
    try {
      report = await syncReferences([reference, reference, `${server.baseUrl}/old.html`], {
        ingester: goldenIngester(new FakeServer()),
        checkpoints: new RecordingCheckpointStore(),
        connectors: [new TextFileConnector(), new HtmlConnector({ allowPrivateNetworks: true })],
      });
      replacements = [
        [tmpPath, "<tmp>"],
        [server.baseUrl, "http://<server>"],
      ];
    } finally {
      await server.close();
    }

    const jsonl = normalizeEventsJsonl(report.eventsJsonl(), replacements);
    const summary = normalizeSummary(report.toDict());
    compareOrWrite("references_run", { jsonl, summary });
  } finally {
    rmSync(tmpPath, { recursive: true, force: true });
  }
});
