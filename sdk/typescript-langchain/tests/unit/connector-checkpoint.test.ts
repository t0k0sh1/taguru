/**
 * ConnectorCheckpoint (ADR 0007 §6.3, issue #347): "did I already fetch
 * and parse this object," composing with — never replacing —
 * `CheckpointStore`. Also `FileProbeCheckpoint` (ADR 0007 §11, issue
 * #353): the local-file twin of `S3ObjectCheckpoint` — "does this file's
 * cheap metadata match what we saw last time," answerable with no read at
 * all, consulted only under `--dry-run`. TypeScript parity: issue #415.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, expect, test, vi } from "vitest";

import { FilesystemCheckpointStore } from "../../src/checkpoints.js";
import {
  ConnectorCheckpoint,
  FileProbe,
  FileProbeCheckpoint,
} from "../../src/ingest-connectors/checkpoint.js";
import {
  ConnectorDocument,
  ConnectorMetadata,
  FingerprintInputs,
  LocatorEntry,
} from "../../src/ingest-connectors/document.js";
import {
  FailingDeleteStore,
  FailingLoadStore,
  FailingSaveStore,
  RecordingCheckpointStore,
} from "./checkpoints.test.js";

afterEach(() => {
  vi.restoreAllMocks();
});

function fingerprint(
  overrides: Partial<ConstructorParameters<typeof FingerprintInputs>[0]> = {},
): FingerprintInputs {
  return new FingerprintInputs({
    rawContentSha256: "deadbeef",
    parser: "taguru-text-connector",
    parserVersion: "1.0.0",
    parseOptionsDigest: "cafef00d",
    ...overrides,
  });
}

function document(options: { source?: string; fingerprint?: FingerprintInputs } = {}) {
  const source = options.source ?? "doc.md";
  return new ConnectorDocument({
    source,
    text: "paragraph one.\n\nparagraph two.",
    locators: [new LocatorEntry({ paragraph: 1, locator: { kind: "page", value: "1" } })],
    metadata: new ConnectorMetadata({ originUri: source, displayName: source }),
    fingerprintInputs: options.fingerprint ?? fingerprint(),
  });
}

function seeded(entries: Record<string, string | Uint8Array>): RecordingCheckpointStore {
  const seed = new Map<string, Uint8Array>();
  for (const [key, value] of Object.entries(entries)) {
    seed.set(key, typeof value === "string" ? new TextEncoder().encode(value) : value);
  }
  return new RecordingCheckpointStore(seed);
}

test("round-trips a document when the fingerprint matches", async () => {
  const checkpoint = new ConnectorCheckpoint(new RecordingCheckpointStore());
  const doc = document();
  await checkpoint.save(doc);
  const loaded = await checkpoint.load(doc.source, doc.fingerprintInputs);
  expect(loaded).toEqual(doc);
});

test("load is null on a fingerprint mismatch", async () => {
  const checkpoint = new ConnectorCheckpoint(new RecordingCheckpointStore());
  const doc = document();
  await checkpoint.save(doc);
  const mismatched = fingerprint({ rawContentSha256: "different" });
  expect(await checkpoint.load(doc.source, mismatched)).toBeNull();
});

test("load is null when the stored document's source does not match", async () => {
  // A CheckpointStore is only obligated to key by the string it was given
  // (`CheckpointStore`'s own contract) — it is not guaranteed to never
  // return another key's bytes under a different key (a buggy or lossy
  // custom implementation). Two documents whose fetched bytes are
  // byte-identical share a fingerprint, so the fingerprint check alone
  // cannot catch this; the explicit `document.source` check does.
  const documentA = document({ source: "a.md" });
  const payload = JSON.stringify(documentA.toDict());
  // Seed the store so a lookup under "b.md"'s own key returns "a.md"'s
  // checkpoint bytes verbatim — simulating a store bug/collision rather
  // than reproducing one through ConnectorCheckpoint's own key derivation.
  const checkpoint = new ConnectorCheckpoint(seeded({ "connector:b.md": payload }));
  expect(await checkpoint.load("b.md", documentA.fingerprintInputs)).toBeNull();
});

test("load is null when nothing was ever saved", async () => {
  const checkpoint = new ConnectorCheckpoint(new RecordingCheckpointStore());
  expect(await checkpoint.load("never-seen.md", fingerprint())).toBeNull();
});

test("load is null on corrupt bytes", async () => {
  const checkpoint = new ConnectorCheckpoint(seeded({ "connector:doc.md": "not json {" }));
  expect(await checkpoint.load("doc.md", fingerprint())).toBeNull();
});

test("delete removes a saved document", async () => {
  const checkpoint = new ConnectorCheckpoint(new RecordingCheckpointStore());
  const doc = document();
  await checkpoint.save(doc);
  await checkpoint.delete(doc.source);
  expect(await checkpoint.load(doc.source, doc.fingerprintInputs)).toBeNull();
});

test("namespace prevents collision with TaguruIngester's own checkpoint", async () => {
  // A ConnectorCheckpoint and TaguruIngester's own chunk checkpoint can
  // safely share one FilesystemCheckpointStore directory: the namespace
  // prefix changes the full key string checkpointFileName hashes, so the
  // two never collide on the same file even for the identical bare source
  // id.
  const directory = mkdtempSync(join(tmpdir(), "connector-checkpoint-"));
  try {
    const store = new FilesystemCheckpointStore(directory);
    const connectorCheckpoint = new ConnectorCheckpoint(store);
    const doc = document({ source: "shared-source.md" });
    await connectorCheckpoint.save(doc);

    // TaguruIngester's own checkpoint machinery would call store.save
    // directly with the bare source id — simulate that here without
    // constructing a full ingester.
    await store.save(
      "shared-source.md",
      new TextEncoder().encode('{"fingerprint": {}, "units": {}}'),
    );

    // The connector's own entry is untouched by the ingester writing under
    // the bare (unprefixed) key.
    const loaded = await connectorCheckpoint.load("shared-source.md", doc.fingerprintInputs);
    expect(loaded).toEqual(doc);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("load failure warns and returns null", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const checkpoint = new ConnectorCheckpoint(new FailingLoadStore());
  expect(await checkpoint.load("doc.md", fingerprint())).toBeNull();
  expect(warn).toHaveBeenCalledWith(expect.stringContaining("load"));
});

test("save failure warns but does not reject", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const checkpoint = new ConnectorCheckpoint(new FailingSaveStore());
  await checkpoint.save(document());
  expect(warn).toHaveBeenCalledWith(expect.stringContaining("save"));
});

test("delete failure is silently ignored", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const checkpoint = new ConnectorCheckpoint(new FailingDeleteStore());
  await checkpoint.delete("doc.md"); // must not reject or warn
  expect(warn).not.toHaveBeenCalled();
});

// ---------------------------------------------------------------------------
// FileProbeCheckpoint — the cheap "skip the dry-run reparse" layer
// ---------------------------------------------------------------------------

function probe(overrides: Partial<ConstructorParameters<typeof FileProbe>[0]> = {}): FileProbe {
  return new FileProbe({
    size: 4096,
    mtimeNs: 1_700_000_000_000_000_000n,
    parser: "taguru-text-connector",
    parserVersion: "1.0.0",
    parseOptionsDigest: "cafef00d",
    ...overrides,
  });
}

test("FileProbeCheckpoint round-trips", async () => {
  const checkpoint = new FileProbeCheckpoint(new RecordingCheckpointStore());
  await checkpoint.save("docs/manual.md", probe());
  expect(await checkpoint.load("docs/manual.md")).toEqual(probe());
});

test("FileProbeCheckpoint is null when never saved", async () => {
  const checkpoint = new FileProbeCheckpoint(new RecordingCheckpointStore());
  expect(await checkpoint.load("never-seen.md")).toBeNull();
});

test("FileProbeCheckpoint is null on corrupt bytes", async () => {
  const checkpoint = new FileProbeCheckpoint(
    seeded({ "file-probe:docs/manual.md": "not json {" }),
  );
  expect(await checkpoint.load("docs/manual.md")).toBeNull();
});

test("FileProbeCheckpoint is null when the source does not match", async () => {
  // Same store-contract caveat ConnectorCheckpoint guards against: a
  // lookup under a different source than the one the payload was recorded
  // for must never return that payload.
  const payload = JSON.stringify({
    version: 1,
    source: "a.md",
    size: 1,
    mtime_ns: "1",
    parser: "taguru-text-connector",
    parser_version: "1.0.0",
    parse_options_digest: "x",
  });
  const checkpoint = new FileProbeCheckpoint(seeded({ "file-probe:b.md": payload }));
  expect(await checkpoint.load("b.md")).toBeNull();
});

test("FileProbeCheckpoint is null on a stale version", async () => {
  // A future version bump must not honor an old-format entry — the same
  // degrade-to-uncached posture every version check in this SDK takes,
  // never a partial/best-effort read of a mismatched version.
  const payload = JSON.stringify({
    version: 0,
    source: "docs/manual.md",
    size: 1,
    mtime_ns: "1",
    parser: "taguru-text-connector",
    parser_version: "1.0.0",
    parse_options_digest: "x",
  });
  const checkpoint = new FileProbeCheckpoint(seeded({ "file-probe:docs/manual.md": payload }));
  expect(await checkpoint.load("docs/manual.md")).toBeNull();
});

test("FileProbeCheckpoint delete removes the entry", async () => {
  const checkpoint = new FileProbeCheckpoint(new RecordingCheckpointStore());
  await checkpoint.save("docs/manual.md", probe());
  await checkpoint.delete("docs/manual.md");
  expect(await checkpoint.load("docs/manual.md")).toBeNull();
});

test("FileProbeCheckpoint load failure is silently uncached", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const checkpoint = new FileProbeCheckpoint(new FailingLoadStore());
  expect(await checkpoint.load("docs/manual.md")).toBeNull();
  expect(warn).not.toHaveBeenCalled();
});

test("FileProbeCheckpoint save failure does not reject", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const checkpoint = new FileProbeCheckpoint(new FailingSaveStore());
  await checkpoint.save("docs/manual.md", probe());
  expect(warn).not.toHaveBeenCalled();
});

test("FileProbeCheckpoint delete failure is silently ignored", async () => {
  const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  const checkpoint = new FileProbeCheckpoint(new FailingDeleteStore());
  await checkpoint.delete("docs/manual.md"); // must not reject or warn
  expect(warn).not.toHaveBeenCalled();
});

test("FileProbeCheckpoint namespace prevents collision with ConnectorCheckpoint", async () => {
  const directory = mkdtempSync(join(tmpdir(), "file-probe-checkpoint-"));
  try {
    const store = new FilesystemCheckpointStore(directory);
    const fileProbeCheckpoint = new FileProbeCheckpoint(store);
    const connectorCheckpoint = new ConnectorCheckpoint(store);
    await fileProbeCheckpoint.save("shared-source.md", probe());
    const doc = document({ source: "shared-source.md" });
    await connectorCheckpoint.save(doc);

    expect(await fileProbeCheckpoint.load("shared-source.md")).toEqual(probe());
    expect(await connectorCheckpoint.load("shared-source.md", doc.fingerprintInputs)).toEqual(doc);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
