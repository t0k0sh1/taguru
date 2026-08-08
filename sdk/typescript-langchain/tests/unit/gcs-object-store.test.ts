/**
 * `GCSObjectStore` (issue #414): the `gs://` backend of the
 * object-storage boundary, exercised through plain fake objects that
 * expose the same STRUCTURAL fields (`code`/`name`) a real
 * `@google-cloud/storage` error would, injected through the class's own
 * documented `client` escape hatch — the TypeScript analogue of the
 * Python original's "real library, no live bucket" posture (which
 * exercises the actual `google.api_core`/`google.auth` exception
 * classes); `@google-cloud/storage` itself is an optional peer dependency
 * not installed in this workspace, so this suite never imports it.
 * TypeScript parity: issue #415.
 */

import { expect, test } from "vitest";

import {
  ObjectNotFoundError,
  objectFingerprint,
  openObjectStore,
  PermanentStoreError,
  TransientStoreError,
} from "../../src/ingest-connectors/objectstore.js";
import {
  GCSObjectStore,
  type GCSBlobLike,
  type GCSBucketLike,
  type GCSClientLike,
} from "../../src/ingest-connectors/objectstore-gcs.js";

const UPDATED = new Date("2026-03-01T00:00:00Z");

async function collect<T>(iterable: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = [];
  for await (const item of iterable) {
    items.push(item);
  }
  return items;
}

function makeBlob(
  overrides: Partial<{
    name: string | null;
    size: number | null;
    updated: Date | null;
    etag: string | null;
    generation: string | number | null;
    crc32c: string | null;
    md5Hash: string | null;
    contentType: string | null;
    metadata: Record<string, string> | null;
    body: Uint8Array;
  }> = {},
): GCSBlobLike {
  const fields = {
    name: "docs/a.txt",
    size: 5,
    updated: UPDATED,
    etag: "etag-1",
    generation: 7 as string | number | null,
    crc32c: "crc-1",
    md5Hash: "md5-1",
    contentType: "text/plain",
    metadata: null as Record<string, string> | null,
    body: new TextEncoder().encode("hello"),
    ...overrides,
  };
  return {
    name: fields.name,
    size: fields.size,
    updated: fields.updated,
    etag: fields.etag,
    generation: fields.generation,
    crc32c: fields.crc32c,
    md5Hash: fields.md5Hash,
    contentType: fields.contentType,
    metadata: fields.metadata,
    download: async () => fields.body,
  };
}

class FakeBucket implements GCSBucketLike {
  blobs = new Map<string, GCSBlobLike>();
  error: unknown = null;
  seenGeneration: (string | null | undefined)[] = [];

  async getBlob(key: string, generation?: string | null): Promise<GCSBlobLike | null> {
    this.seenGeneration.push(generation);
    if (this.error !== null) {
      throw this.error;
    }
    return this.blobs.get(key) ?? null;
  }
}

class FakeClient implements GCSClientLike {
  listed: GCSBlobLike[] = [];
  listError: unknown = null;
  fakeBucket: FakeBucket = new FakeBucket();
  seenPrefix: string[] = [];

  async *listBlobs(_bucketName: string, prefix: string): AsyncGenerator<GCSBlobLike> {
    this.seenPrefix.push(prefix);
    if (this.listError !== null) {
      throw this.listError;
    }
    for (const blob of this.listed) {
      yield blob;
    }
  }

  bucket(_name: string): GCSBucketLike {
    return this.fakeBucket;
  }
}

function store(client: GCSClientLike): GCSObjectStore {
  return new GCSObjectStore("my-bucket", { client });
}

test("client is mutually exclusive with project", () => {
  expect(() => new GCSObjectStore("b", { project: "p", client: new FakeClient() })).toThrow(
    /mutually exclusive/,
  );
});

test("base_uri names the bucket", () => {
  expect(store(new FakeClient()).baseUri).toBe("gs://my-bucket");
});

// ---------------------------------------------------------------------------
// list — field mapping onto ObjectMeta and its fingerprint tiers
// ---------------------------------------------------------------------------

test("list maps generation to version_id and prefers crc32c", async () => {
  const client = new FakeClient();
  client.listed = [makeBlob()];
  const [m] = await collect(store(client).list("docs/"));
  expect(client.seenPrefix).toEqual(["docs/"]);
  expect(m!.key).toBe("docs/a.txt");
  expect(m!.size).toBe(5);
  expect(m!.lastModified).toEqual(UPDATED);
  expect(m!.etag).toBe("etag-1");
  // Generation is stamped on every overwrite whether or not the bucket
  // has versioning on — the top fingerprint tier, always available.
  expect(m!.versionId).toBe("7");
  expect(m!.checksum).toBe("crc32c:crc-1");
  expect(objectFingerprint(m!)).toEqual(["version_id", "7"]);
});

test("list checksum falls back to md5 then null", async () => {
  const client = new FakeClient();
  client.listed = [makeBlob({ crc32c: null }), makeBlob({ crc32c: null, md5Hash: null })];
  const [first, second] = await collect(store(client).list(""));
  expect(first!.checksum).toBe("md5:md5-1");
  expect(second!.checksum).toBeNull();
});

test("list without generation degrades version_id to null", async () => {
  const client = new FakeClient();
  client.listed = [makeBlob({ generation: null })];
  const [m] = await collect(store(client).list(""));
  expect(m!.versionId).toBeNull();
});

test("list on a missing bucket is permanent, not object not found", async () => {
  const client = new FakeClient();
  client.listError = { code: 404, message: "no such bucket" };
  await expect(collect(store(client).list(""))).rejects.toBeInstanceOf(PermanentStoreError);
});

test.each([
  ["Forbidden", { code: 403, message: "denied" }, PermanentStoreError],
  ["DefaultCredentialsError", { name: "DefaultCredentialsError", message: "no ADC" }, PermanentStoreError],
  ["ServiceUnavailable", { code: 503, message: "503" }, TransientStoreError],
  ["TooManyRequests", { code: 429, message: "429" }, TransientStoreError],
  ["ConnectionResetError", { message: "reset" }, TransientStoreError],
] as const)("list error classification: %s", async (_name, error, expected) => {
  const client = new FakeClient();
  client.listError = error;
  await expect(collect(store(client).list(""))).rejects.toBeInstanceOf(expected);
});

// ---------------------------------------------------------------------------
// get — bytes + content type, generation pinning, not-found semantics
// ---------------------------------------------------------------------------

test("get returns bytes, content type, and pins the listed generation", async () => {
  const bucket = new FakeBucket();
  bucket.blobs.set("docs/a.txt", makeBlob());
  const client = new FakeClient();
  client.fakeBucket = bucket;
  const fetched = await store(client).get("docs/a.txt", { versionId: "7" });
  expect(fetched.body).toEqual(new TextEncoder().encode("hello"));
  expect(fetched.contentType).toBe("text/plain");
  expect(bucket.seenGeneration).toEqual(["7"]);
});

test("get without version_id passes no generation", async () => {
  const bucket = new FakeBucket();
  bucket.blobs.set("a", makeBlob());
  const client = new FakeClient();
  client.fakeBucket = bucket;
  await store(client).get("a");
  expect(bucket.seenGeneration).toEqual([null]);
});

test("get missing object is object not found", async () => {
  const client = new FakeClient();
  await expect(store(client).get("gone.txt")).rejects.toBeInstanceOf(ObjectNotFoundError);
});

test("get not found from the service is object not found", async () => {
  // A vanished generation (overwritten object, versioning off) answers
  // 404 — "skip this object this pass," never a store failure.
  const bucket = new FakeBucket();
  bucket.error = { code: 404, message: "stale generation" };
  const client = new FakeClient();
  client.fakeBucket = bucket;
  await expect(store(client).get("a", { versionId: "6" })).rejects.toBeInstanceOf(
    ObjectNotFoundError,
  );
});

test("get permission denial is permanent", async () => {
  const bucket = new FakeBucket();
  bucket.error = { code: 403, message: "denied" };
  const client = new FakeClient();
  client.fakeBucket = bucket;
  await expect(store(client).get("a")).rejects.toBeInstanceOf(PermanentStoreError);
});

// ---------------------------------------------------------------------------
// objectTags — custom metadata, with the tags-only degrade
// ---------------------------------------------------------------------------

test("objectTags maps the blob's custom metadata", async () => {
  const bucket = new FakeBucket();
  bucket.blobs.set("a", makeBlob({ metadata: { team: "ingest", quarter: "q1" } }));
  const client = new FakeClient();
  client.fakeBucket = bucket;
  expect(await store(client).objectTags("a")).toEqual({ team: "ingest", quarter: "q1" });
});

test("objectTags degrades to empty for a missing blob or metadata", async () => {
  const bucket = new FakeBucket();
  bucket.blobs.set("a", makeBlob({ metadata: null }));
  const client = new FakeClient();
  client.fakeBucket = bucket;
  const s = store(client);
  expect(await s.objectTags("a")).toEqual({});
  expect(await s.objectTags("gone")).toEqual({});
});

test("objectTags degrades on permission denial but raises on transient", async () => {
  const deniedBucket = new FakeBucket();
  deniedBucket.error = { code: 403, message: "denied" };
  const deniedClient = new FakeClient();
  deniedClient.fakeBucket = deniedBucket;
  expect(await store(deniedClient).objectTags("a")).toEqual({});

  const busyBucket = new FakeBucket();
  busyBucket.error = { code: 503, message: "503" };
  const busyClient = new FakeClient();
  busyClient.fakeBucket = busyBucket;
  await expect(store(busyClient).objectTags("a")).rejects.toBeInstanceOf(TransientStoreError);
});

// ---------------------------------------------------------------------------
// Construction without a client: package presence is checked on first
// use, not eagerly — the deliberate TS deviation objectstore-gcs.ts
// documents (an `await import()` cannot run inside a synchronous
// constructor). Not part of the Python original's own suite (whose
// `GCSObjectStore.__init__` checks package presence synchronously and is
// never exercised without a client there either).
// ---------------------------------------------------------------------------

test("construction without a client defers the missing-package check to first use", async () => {
  expect(() => new GCSObjectStore("my-bucket")).not.toThrow();
  await expect(collect(new GCSObjectStore("my-bucket").list(""))).rejects.toThrow(
    /@google-cloud\/storage/,
  );
});

// ---------------------------------------------------------------------------
// openObjectStore — the gs:// scheme
// ---------------------------------------------------------------------------

test("openObjectStore dispatches gs urls", async () => {
  const [s, prefix] = await openObjectStore("gs://my-bucket/reports/2026");
  expect(s).toBeInstanceOf(GCSObjectStore);
  expect(s.baseUri).toBe("gs://my-bucket");
  expect(prefix).toBe("reports/2026");
});

test("openObjectStore requires a gs bucket", async () => {
  await expect(openObjectStore("gs:///prefix-only")).rejects.toThrow(/must name a bucket/);
});

test("openObjectStore rejects s3-only knobs for gs", async () => {
  await expect(
    openObjectStore("gs://bucket", { regionName: "asia-northeast1" }),
  ).rejects.toThrow(/does not accept/);
});
