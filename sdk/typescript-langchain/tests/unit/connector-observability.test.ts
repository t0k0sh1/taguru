/**
 * Cross-connector observability (ADR 0007 §11, issue #353): `SourceEvent`/
 * `RunReport` serialization, `RunRecorder`'s last-phase-only tally and
 * `retarget`, `SourceEventSink`'s append-only JSONL posture (mirroring
 * `taguru extract`'s `DiagnosticsSink`, src/extract.rs:2372), and the
 * `onIngestEvent`/`attach`/`attached` bridge that lets a driver report the
 * `extracted` phase from `TaguruIngester`'s own `ImportStarted` event.
 * Mirrors the Python twin's test_connector_observability.py case for case
 * (TypeScript parity: issue #415).
 */

import {
  closeSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  rmSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test, vi } from "vitest";

import { Diagnostic } from "../../src/ingest-connectors/document.js";
import {
  PHASES,
  RUN_SUMMARY_KIND,
  RUN_SUMMARY_VERSION,
  RunRecorder,
  RunReport,
  S3SyncReport,
  SOURCE_EVENT_KIND,
  SourceEvent,
  SourceEventSink,
} from "../../src/ingest-connectors/observability.js";
import type { IngestEvent, IngestEventCallback } from "../../src/events.js";
import type { TaguruIngester } from "../../src/ingest.js";

// ---------------------------------------------------------------------------
// SourceEvent / RunReport — serialization shape
// ---------------------------------------------------------------------------

test("source event toDict carries every field, null when absent", () => {
  const event = new SourceEvent({ source: "docs/manual.md", phase: "discovered", elapsedMs: 0.0 });
  expect(event.toDict()).toEqual({
    kind: SOURCE_EVENT_KIND,
    source: "docs/manual.md",
    phase: "discovered",
    elapsed_ms: 0.0,
    bytes: 0,
    parser: null,
    diagnostic: null,
  });
});

test("source event toDict nests a diagnostic as exactly three fields", () => {
  const diagnostic = new Diagnostic({
    code: "ocr_required",
    message: "no usable text layer",
    source: "a.pdf",
  });
  const event = new SourceEvent({
    source: "a.pdf",
    phase: "skipped",
    elapsedMs: 31.4,
    bytes: 2048,
    diagnostic,
  });
  expect(event.toDict()["diagnostic"]).toEqual({
    code: "ocr_required",
    message: "no usable text layer",
    source: "a.pdf",
  });
});

test("run report toDict key set is pinned", () => {
  // A literal key list in the test source, not a derived one — an added or
  // removed key must edit both this list and the ADR/CHANGELOG in the same
  // PR, never drift unnoticed (ADR 0007 §11's stable summary shape).
  const report = new RunReport();
  expect(Object.keys(report.toDict()).sort()).toEqual(
    [
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
    ].sort(),
  );
  expect(report.toDict()["kind"]).toBe(RUN_SUMMARY_KIND);
  expect(report.toDict()["connector_run"]).toBe(RUN_SUMMARY_VERSION);
});

test("run report events field is a count, not the events themselves", () => {
  const events = [
    new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }),
    new SourceEvent({ source: "a.md", phase: "parsed", elapsedMs: 5.0 }),
  ];
  const report = new RunReport({ events });
  expect(report.toDict()["events"]).toBe(2);
});

test("run report eventsJsonl renders one line per event", () => {
  const events = [
    new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }),
    new SourceEvent({ source: "a.md", phase: "parsed", elapsedMs: 5.0 }),
  ];
  const report = new RunReport({ events });
  const rendered = report.eventsJsonl();
  const lines = rendered.split("\n").filter((line) => line.length > 0);
  expect(lines).toHaveLength(2);
  expect(JSON.parse(lines[0]!).phase).toBe("discovered");
  expect(JSON.parse(lines[1]!).phase).toBe("parsed");
  // Trailing-newline-terminated, matching what SourceEventSink.write()
  // produces incrementally on disk for the same run.
  expect(rendered.endsWith("\n")).toBe(true);
});

test("run report eventsJsonl is empty string with no events", () => {
  expect(new RunReport({ events: [] }).eventsJsonl()).toBe("");
});

test("S3SyncReport is the RunReport alias", () => {
  expect(S3SyncReport).toBe(RunReport);
});

test("PHASES matches the seven-state vocabulary", () => {
  expect(PHASES).toEqual([
    "discovered",
    "unchanged",
    "parsed",
    "extracted",
    "imported",
    "skipped",
    "failed",
  ]);
});

test("source event shares the Rust diagnostics key set", () => {
  // Anchors ADR 0007 §8's promise that a connector diagnostic uses "the
  // same three fields taguru extract's own diagnostics sidecar" already
  // uses — the TypeScript-side parity check for that Rust doc comment.
  const diagnostic = new Diagnostic({ code: "corrupt", message: "truncated", source: "a.pdf" });
  expect(Object.keys(diagnostic.toDict()).sort()).toEqual(["code", "message", "source"]);
});

// ---------------------------------------------------------------------------
// RunRecorder — last-phase-only tally, retarget, dropped/deletion counters
// ---------------------------------------------------------------------------

test("last-phase-only tally counts a source once under its final phase", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.record("a.md", "parsed");
  recorder.record("a.md", "extracted");
  recorder.record("a.md", "imported");
  const report = recorder.finish();
  expect(report.discovered).toBe(0);
  expect(report.parsed).toBe(0);
  expect(report.extracted).toBe(0);
  expect(report.imported).toBe(1);
  expect(report.events).toHaveLength(4);
});

test("two sources tally independently", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.discovered("b.md");
  recorder.record("a.md", "imported");
  recorder.record("b.md", "failed");
  const report = recorder.finish();
  expect(report.imported).toBe(1);
  expect(report.failed).toBe(1);
});

test("retarget keeps one reference counted once", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("https://example.com/a");
  recorder.retarget("https://example.com/a", "https://example.com/a/");
  recorder.record("https://example.com/a/", "imported");
  const report = recorder.finish();
  expect(report.imported).toBe(1);
  expect(report.discovered).toBe(0);
  expect(report.failed).toBe(0);
});

test("retarget is a no-op for an unknown old source", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.retarget("never-discovered", "also-never");
  recorder.record("a.md", "imported");
  expect(recorder.finish().imported).toBe(1);
});

test("retarget is a no-op when old equals new", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.retarget("a.md", "a.md");
  recorder.record("a.md", "imported");
  expect(recorder.finish().imported).toBe(1);
});

test("retarget returns true on a normal rename", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("old");
  expect(recorder.retarget("old", "new")).toBe(true);
});

test("retarget onto an existing different source does not clobber it", () => {
  // The bug this return-value contract exists to prevent: two distinct
  // references (a and b) both retargeting onto the same already-imported
  // `new` must never let the second one reset `new`'s tally.
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a");
  expect(recorder.retarget("a", "new")).toBe(true);
  recorder.record("new", "imported");

  recorder.discovered("b");
  expect(recorder.retarget("b", "new")).toBe(false);

  const report = recorder.finish();
  expect(report.imported).toBe(1);
  expect(report.discovered).toBe(0);
  expect(report.unchanged).toBe(0);
  const duplicateEvents = report.events.filter(
    (e) => e.diagnostic !== null && e.diagnostic.code === "duplicate_source",
  );
  expect(duplicateEvents).toHaveLength(1);
  expect(duplicateEvents[0]!.source).toBe("b");
});

test("duplicate does not disturb the claimed source's tally", () => {
  // The bug this method exists to prevent: a duplicate input sharing an
  // already-imported source's id must never turn that source's tally back
  // into `skipped`.
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.record("a.md", "imported");
  recorder.duplicate("a.md", { of: "a.md" });
  const report = recorder.finish();
  expect(report.imported).toBe(1);
  expect(report.skipped).toBe(0);
});

test("duplicate is visible in the events log but not the tally", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("https://example.com/a");
  recorder.record("https://example.com/a", "imported");
  recorder.duplicate("https://example.com/a?utm_source=x", { of: "https://example.com/a" });
  const report = recorder.finish();
  expect(report.imported).toBe(1);
  expect(report.discovered).toBe(0);
  const duplicateEvents = report.events.filter(
    (e) => e.source === "https://example.com/a?utm_source=x",
  );
  expect(duplicateEvents).toHaveLength(1);
  expect(duplicateEvents[0]!.phase).toBe("skipped");
  expect(duplicateEvents[0]!.diagnostic).not.toBeNull();
  expect(duplicateEvents[0]!.diagnostic!.code).toBe("duplicate_source");
});

test("addDropped accumulates across calls", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.addDropped({ locators: 1, sections: 2 });
  recorder.addDropped({ locators: 3, tags: 4 });
  const report = recorder.finish();
  expect(report.locatorsDropped).toBe(4);
  expect(report.sectionsDropped).toBe(2);
  expect(report.tagsDropped).toBe(4);
});

test("addDeletions accumulates across calls", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.addDeletions({ detected: 2, retracted: 1 });
  recorder.addDeletions({ detected: 1, retracted: 0 });
  const report = recorder.finish();
  expect(report.deletedDetected).toBe(3);
  expect(report.retracted).toBe(1);
});

test("finish interrupted is reported verbatim", () => {
  const recorder = new RunRecorder({ connector: "test" });
  expect(recorder.finish({ interrupted: true }).interrupted).toBe(true);
});

test("keepEvents false drops events but keeps the tally", () => {
  const recorder = new RunRecorder({ connector: "test", keepEvents: false });
  recorder.discovered("a.md");
  recorder.record("a.md", "imported");
  const report = recorder.finish();
  expect(report.events).toEqual([]);
  expect(report.imported).toBe(1);
});

test("connector name is reported on the summary", () => {
  const recorder = new RunRecorder({ connector: "taguru-pdf-connector" });
  expect(recorder.finish().connector).toBe("taguru-pdf-connector");
});

// ---------------------------------------------------------------------------
// onIngestEvent / attach / attached — the `extracted` phase
// ---------------------------------------------------------------------------

/**
 * The only thing `attach`/`attached` touches on a `TaguruIngester` is its
 * `on_event` field — a bare stand-in avoids constructing a real ingester
 * (LLM, client, checkpoint store) for a test that never ingests anything.
 * Mirrors the Python suite's `_FakeIngester`.
 */
class FakeIngester {
  on_event: IngestEventCallback | undefined;
}

function asIngester(fake: FakeIngester): TaguruIngester {
  return fake as unknown as TaguruIngester;
}

test("onIngestEvent maps import_started to extracted", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.onIngestEvent({ kind: "import_started", source: "a.md" });
  expect(recorder.finish().extracted).toBe(1);
});

test("onIngestEvent ignores every other event kind", () => {
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  recorder.onIngestEvent({ kind: "chunk_started", source: "a.md", index: 0, total: 1 });
  recorder.onIngestEvent({ kind: "import_completed", source: "a.md", elapsed_seconds: 1.0 });
  const report = recorder.finish();
  expect(report.discovered).toBe(1);
  expect(report.extracted).toBe(0);
});

test("attach() returns a detach function that restores the previous callback", () => {
  const fake = new FakeIngester();
  const original: IngestEventCallback = () => {};
  fake.on_event = original;
  const recorder = new RunRecorder({ connector: "test" });
  const detach = recorder.attach(asIngester(fake));
  expect(fake.on_event).not.toBe(original);
  detach();
  expect(fake.on_event).toBe(original);
});

test("attached records extracted for the duration of the scope", async () => {
  const fake = new FakeIngester();
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  await recorder.attached(asIngester(fake), () => {
    fake.on_event?.({ kind: "import_started", source: "a.md" });
  });
  expect(recorder.finish().extracted).toBe(1);
});

test("attached chains a pre-existing callback, never replacing it", async () => {
  const seen: IngestEvent[] = [];
  const fake = new FakeIngester();
  fake.on_event = (event) => {
    seen.push(event);
  };
  const recorder = new RunRecorder({ connector: "test" });
  recorder.discovered("a.md");
  const event: IngestEvent = { kind: "import_started", source: "a.md" };
  await recorder.attached(asIngester(fake), () => {
    fake.on_event?.(event);
  });
  expect(seen).toEqual([event]);
  expect(recorder.finish().extracted).toBe(1);
});

test("attached restores the previous callback after the scope", async () => {
  const fake = new FakeIngester();
  const original: IngestEventCallback = () => {};
  fake.on_event = original;
  const recorder = new RunRecorder({ connector: "test" });
  await recorder.attached(asIngester(fake), () => {
    expect(fake.on_event).not.toBe(original);
  });
  expect(fake.on_event).toBe(original);
});

test("attached restores the previous callback even on exception", async () => {
  const fake = new FakeIngester();
  const original: IngestEventCallback = () => {};
  fake.on_event = original;
  const recorder = new RunRecorder({ connector: "test" });
  await expect(
    recorder.attached(asIngester(fake), () => {
      throw new Error("boom");
    }),
  ).rejects.toThrow("boom");
  expect(fake.on_event).toBe(original);
});

// ---------------------------------------------------------------------------
// SourceEventSink — append-only JSONL, DiagnosticsSink's degrade posture
// ---------------------------------------------------------------------------

function withTmpDir(fn: (directory: string) => void): void {
  const directory = mkdtempSync(join(tmpdir(), "connector-observability-"));
  try {
    fn(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

test("sink writes one flushed line per event", () => {
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const sink = new SourceEventSink(path);
    sink.write(new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }));
    // Flushed, not just buffered — readable before close().
    let lines = readFileSync(path, "utf-8").split("\n").filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0]!).phase).toBe("discovered");
    sink.write(new SourceEvent({ source: "a.md", phase: "parsed", elapsedMs: 5.0 }));
    lines = readFileSync(path, "utf-8").split("\n").filter((l) => l.length > 0);
    expect(lines).toHaveLength(2);
    sink.close();
  });
});

test("sink truncates an existing file on open", () => {
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const fd = openSync(path, "w");
    closeSync(fd);
    const stale = "stale content from a prior run\n";
    writeFileSync(path, stale, "utf-8");
    const sink = new SourceEventSink(path);
    sink.write(new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }));
    sink.close();
    const lines = readFileSync(path, "utf-8").split("\n").filter((l) => l.length > 0);
    expect(lines).toHaveLength(1);
    expect(lines[0]).not.toContain("stale content");
  });
});

test("sink open failure warns once and the run continues", () => {
  withTmpDir((directory) => {
    const directoryAsPath = join(directory, "not-a-file");
    mkdirSync(directoryAsPath);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const sink = new SourceEventSink(directoryAsPath);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toContain("opening");
    // No further warnings — write() silently no-ops rather than throwing.
    sink.write(new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }));
    expect(warn).toHaveBeenCalledTimes(1);
    sink.close();
    warn.mockRestore();
  });
});

test("caller-supplied file descriptor is never closed by the sink", () => {
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const fd = openSync(path, "w");
    const sink = new SourceEventSink({ fd, path });
    sink.write(new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }));
    sink.close();
    // Still open: a direct write against the caller's own fd must not throw.
    expect(() => writeSync(fd, "")).not.toThrow();
    closeSync(fd);
  });
});

test("write to a caller-closed file descriptor degrades instead of raising", () => {
  // A caller-supplied file descriptor is never owned by the sink (the test
  // above) — if the caller closes it first, a later write() must degrade
  // like any other write failure (one warning, then silent), not throw.
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const fd = openSync(path, "w");
    closeSync(fd);
    const sink = new SourceEventSink({ fd, path });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    sink.write(new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }));
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]![0]).toContain("writing to");
    // No further warnings — later writes silently no-op rather than raising.
    sink.write(new SourceEvent({ source: "b.md", phase: "discovered", elapsedMs: 0.0 }));
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
  });
});

test("sink path property reports a caller descriptor's name", () => {
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const fd = openSync(path, "w");
    const sink = new SourceEventSink({ fd, path });
    expect(sink.path).toBe(path);
    closeSync(fd);
  });
});

test("sink close() flushes and closes an owned file", () => {
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const sink = new SourceEventSink(path);
    sink.write(new SourceEvent({ source: "a.md", phase: "discovered", elapsedMs: 0.0 }));
    sink.close();
    // Closed: reading the file after close() sees the flushed content.
    expect(readFileSync(path, "utf-8").split("\n").filter((l) => l.length > 0)).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// RunRecorder + eventsOut end to end
// ---------------------------------------------------------------------------

test("recorder eventsOut writes the same shape as finish", () => {
  withTmpDir((directory) => {
    const path = join(directory, "events.jsonl");
    const recorder = new RunRecorder({ connector: "test", eventsOut: path });
    recorder.discovered("a.md");
    recorder.record("a.md", "imported", { parser: "taguru-text-connector" });
    const report = recorder.finish();
    recorder.close();

    const onDisk = readFileSync(path, "utf-8")
      .split("\n")
      .filter((l) => l.length > 0)
      .map((line) => JSON.parse(line));
    const fromReport = report.events.map((event) => event.toDict());
    expect(onDisk).toEqual(fromReport);
    expect(report.eventsPath).toBe(path);
    // Byte-for-byte, not just parsed-equal: eventsJsonl()'s own trailing
    // newline must match what the sink wrote incrementally to `path`.
    expect(report.eventsJsonl()).toBe(readFileSync(path, "utf-8"));
  });
});
