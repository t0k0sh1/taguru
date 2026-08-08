/**
 * The object-storage boundary (ADR 0007 §9, issue #351):
 * `objectFingerprint`'s checkpoint-fingerprint priority, `FileObjectStore`
 * (node:fs only — needs no optional peer dependency at all), and
 * `openObjectStore`'s scheme dispatch. TypeScript parity: issue #415.
 *
 * The Python original's full `S3ObjectStore` suite (real `boto3`, stubbed
 * at the HTTP layer with `botocore.stub.Stubber`) is out of this file's
 * scope — `S3ObjectStore` (`objectstore-s3.ts`, the `@aws-sdk/client-s3`
 * backend) is exercised via its own injectable `client` seam in
 * `s3-connector.test.ts`'s "credentials never appear" test, mirroring
 * gcs-object-store.test.ts/azure-object-store.test.ts's own established
 * convention (a network-free fake standing in for the optional, not
 * installed, cloud SDK). The one test below that dispatches a real
 * `s3://` URL end to end needed only `objectstore-s3.ts` to exist — it no
 * longer needs `test.skip` now that it does.
 */

import { chmodSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { expect, test } from "vitest";

import {
  FileObjectStore,
  ObjectMeta,
  ObjectNotFoundError,
  objectFingerprint,
  openObjectStore,
} from "../../src/ingest-connectors/objectstore.js";

async function collect<T>(iterable: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = [];
  for await (const item of iterable) {
    items.push(item);
  }
  return items;
}

function meta(overrides: Partial<ConstructorParameters<typeof ObjectMeta>[0]> = {}): ObjectMeta {
  return new ObjectMeta({ key: "a.txt", size: 5, lastModified: null, ...overrides });
}

function writeBucket(root: string, objects: Record<string, string>): void {
  for (const [key, body] of Object.entries(objects)) {
    const path = join(root, key);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, body);
  }
}

// ---------------------------------------------------------------------------
// objectFingerprint — ADR 0007 §9's priority order
// ---------------------------------------------------------------------------

test("version_id wins over everything else", () => {
  const m = meta({
    versionId: "v1",
    checksum: "c1",
    lastModified: new Date("2026-01-01T00:00:00Z"),
    etag: "e1",
  });
  expect(objectFingerprint(m)).toEqual(["version_id", "v1"]);
});

test("checksum wins over size_last_modified and etag", () => {
  const m = meta({ checksum: "c1", lastModified: new Date("2026-01-01T00:00:00Z"), etag: "e1" });
  expect(objectFingerprint(m)).toEqual(["checksum", "c1"]);
});

test("size and last_modified wins over etag alone", () => {
  const when = new Date("2026-01-01T00:00:00Z");
  const m = meta({ size: 42, lastModified: when, etag: "e1" });
  expect(objectFingerprint(m)).toEqual(["size_last_modified", `42:${when.toISOString()}`]);
});

test("etag is the last resort", () => {
  const m = meta({ etag: "e1" });
  expect(objectFingerprint(m)).toEqual(["etag", "e1"]);
});

test("no usable fingerprint is null", () => {
  expect(objectFingerprint(meta())).toBeNull();
});

// ---------------------------------------------------------------------------
// FileObjectStore — node:fs only, the file:// test/air-gapped backend
// ---------------------------------------------------------------------------

test("FileObjectStore refuses a missing bucket directory", async () => {
  const parent = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    await expect(FileObjectStore.open(join(parent, "does-not-exist"))).rejects.toThrow(
      /does not exist/,
    );
  } finally {
    rmSync(parent, { recursive: true, force: true });
  }
});

test("FileObjectStore lists by string prefix, not directory boundary", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, {
      "2026/q1-report.pdf": "pdf bytes",
      "2026/q1-notes.txt": "notes",
      "2026/q2-report.pdf": "other",
    });
    const store = await FileObjectStore.open(dir);
    const keys = (await collect(store.list("2026/q1"))).map((m) => m.key).sort();
    expect(keys).toEqual(["2026/q1-notes.txt", "2026/q1-report.pdf"]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore list skips an unreadable subdirectory instead of aborting", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  const locked = join(dir, "locked");
  try {
    writeBucket(dir, { "readable/a.txt": "fine", "locked/secret.txt": "hidden" });
    chmodSync(locked, 0o000);
    if (process.getuid?.() === 0) {
      // Root ignores permission bits — the EACCES this test provokes
      // cannot happen, so there is nothing to assert.
      return;
    }
    const store = await FileObjectStore.open(dir);
    const keys = (await collect(store.list(""))).map((m) => m.key);
    expect(keys).toEqual(["readable/a.txt"]);
  } finally {
    chmodSync(locked, 0o755);
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore list still surfaces an unreadable ROOT", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, { "a.txt": "x" });
    const store = await FileObjectStore.open(dir);
    chmodSync(dir, 0o000);
    if (process.getuid?.() === 0) {
      return;
    }
    // A permission denial on the store's root is a misconfigured store,
    // not one unreadable corner — it must not be silently skipped.
    await expect(collect(store.list(""))).rejects.toThrow();
  } finally {
    chmodSync(dir, 0o755);
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore list reports size and mtime", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, { "a.txt": "hello" });
    const store = await FileObjectStore.open(dir);
    const metas = await collect(store.list(""));
    expect(metas).toHaveLength(1);
    expect(metas[0]!.key).toBe("a.txt");
    expect(metas[0]!.size).toBe(5);
    expect(metas[0]!.lastModified).not.toBeNull();
    expect(metas[0]!.etag).toBeNull();
    expect(metas[0]!.versionId).toBeNull();
    expect(metas[0]!.checksum).toBeNull();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore get returns bytes and guessed content type", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, { "doc.pdf": "%PDF-1.4 ..." });
    const store = await FileObjectStore.open(dir);
    const fetched = await store.get("doc.pdf");
    expect(new TextDecoder().decode(fetched.body)).toBe("%PDF-1.4 ...");
    expect(fetched.contentType).toBe("application/pdf");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore get raises not found for a missing key", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    const store = await FileObjectStore.open(dir);
    await expect(store.get("missing.txt")).rejects.toBeInstanceOf(ObjectNotFoundError);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore never carries tags", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, { "a.txt": "x" });
    const store = await FileObjectStore.open(dir);
    expect(await store.objectTags("a.txt")).toEqual({});
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore refuses a key that escapes the bucket", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, { "a.txt": "x" });
    const store = await FileObjectStore.open(dir);
    await expect(store.get("../outside.txt")).rejects.toThrow(/escapes/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("FileObjectStore base_uri", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    const store = await FileObjectStore.open(dir);
    expect(store.baseUri).toBe(`file://${realpathSync(dir)}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// openObjectStore — scheme dispatch, mirroring src/ship.rs's open_store
// ---------------------------------------------------------------------------

test("openObjectStore dispatches the file scheme", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    writeBucket(dir, { "a.txt": "x" });
    const [store, prefix] = await openObjectStore(`file://${dir}`);
    expect(store).toBeInstanceOf(FileObjectStore);
    expect(prefix).toBe("");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("openObjectStore rejects s3 kwargs for the file scheme", async () => {
  const dir = mkdtempSync(join(tmpdir(), "objectstore-"));
  try {
    await expect(
      openObjectStore(`file://${dir}`, { regionName: "us-east-1" }),
    ).rejects.toThrow(/does not accept/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("openObjectStore rejects an s3 url with no bucket", async () => {
  await expect(openObjectStore("s3:///prefix")).rejects.toThrow(/bucket/);
});

test("openObjectStore rejects an unsupported scheme", async () => {
  // gs:// and az:// stopped being this test's example the day #414 made
  // them real schemes; ftp:// is nobody's object store.
  await expect(openObjectStore("ftp://bucket/prefix")).rejects.toThrow(
    /s3:\/\/, gs:\/\/, az:\/\/, or file:\/\//,
  );
});

// The Python original's `test_open_object_store_returns_the_urls_own_path_as_prefix`
// dispatches a real `s3://` URL end to end (skipped there too, via
// `pytest.importorskip("boto3")`, when the optional dependency isn't
// installed). `S3ObjectStore` (objectstore-s3.ts) defers its own
// `@aws-sdk/client-s3` presence check to first `list`/`get`/`objectTags`
// call rather than to construction (mirroring `GCSObjectStore`/
// `AzureBlobObjectStore`'s own documented deviation) — so dispatching this
// URL through `openObjectStore` and reading `baseUri`/the returned prefix
// alone never touches the optional, not-installed package.
test("openObjectStore returns the url's own path as prefix (s3)", async () => {
  const [store, prefix] = await openObjectStore("s3://my-bucket/reports/2026/", {
    regionName: "us-east-1",
  });
  expect(store.baseUri).toBe("s3://my-bucket");
  expect(prefix).toBe("reports/2026/");
});
