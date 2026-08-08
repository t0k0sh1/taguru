/**
 * Machine-readable summary fixture tests (issue #353's own explicit
 * acceptance criterion: "machine-readable summary の fixture テストがある"),
 * ported from the Python twin's test_connector_observability_fixtures.py
 * (TypeScript parity: issue #415).
 *
 * The Python suite pins three golden scenarios — `s3_sync`,
 * `references_dry_run`, `references_run` — produced by
 * `sync_object_storage`/`sync_references`, connector DRIVERS this SDK has
 * not yet ported to TypeScript (issue #415 tracks this `observability`
 * module alone; the S3/references drivers are separate, later ports that
 * depend on `RunRecorder.attach`/`attached`). This file therefore ports the
 * driver-independent half of the Python suite:
 *
 * - The stable summary key set every golden must share
 *   (`test_summary_key_set_matches_every_golden`).
 * - The normalization/compare machinery
 *   (`_normalize_events_jsonl`/`_normalize_summary`/`_compare_or_write`) a
 *   later S3/references port can reuse verbatim to add its own goldens
 *   under `tests/fixtures/ingest-observability/`, proven here against a
 *   synthetic run rather than a real connector driver.
 *
 * Once `syncObjectStorage`/`syncReferences` land in TypeScript, their own
 * golden-scenario tests belong in this file, alongside these.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "vitest";

import { Diagnostic } from "../../src/ingest-connectors/document.js";
import {
  RUN_SUMMARY_KIND,
  RUN_SUMMARY_VERSION,
  RunReport,
  SourceEvent,
} from "../../src/ingest-connectors/observability.js";

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
