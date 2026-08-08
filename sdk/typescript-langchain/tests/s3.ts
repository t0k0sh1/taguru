/**
 * Test-only `ObjectStore` (objectstore.ts) support for the S3 connector's
 * own tests (issue #351): `FakeObjectStore`, an in-memory store synthesized
 * at test time (the same "no binary/external fixture" convention
 * `tests/pdfs.ts`/`tests/docx.ts` already established), and `writeBucket`,
 * a small helper that lays out a plain directory tree for
 * `FileObjectStore` (objectstore.ts) — the `file://` backend issue #351's
 * own acceptance criteria names explicitly. The mechanical mirror of the
 * Python `tests._s3` module (issue #415).
 *
 * `FakeObjectStore` exists ALONGSIDE `FileObjectStore` rather than instead
 * of it: the two together let a test choose the right level —
 * `FakeObjectStore` for anything about listing/fingerprint/dispatch/
 * checkpoint logic (no filesystem, deterministic, can inject
 * `versionId`/`checksum`/tags/failures `FileObjectStore` can never
 * produce), `FileObjectStore`-over-`writeBucket` for anything proving the
 * real node:fs-only backend and a real end-to-end sync work.
 */

import {
  FetchedObject,
  type ObjectMeta as ObjectMetaType,
  ObjectMeta,
  ObjectNotFoundError,
  type ObjectStore,
} from "../src/ingest-connectors/objectstore.js";

interface StoredObject {
  body: Uint8Array;
  contentType: string | null;
  etag: string | null;
  versionId: string | null;
  checksum: string | null;
  lastModified: Date | null;
  tags: Record<string, string>;
}

/**
 * An in-memory `ObjectStore` (objectstore.ts) — seeded via `put`, never
 * touches a filesystem or network. Tracks `getCalls`/`tagsCalls` (key →
 * count) so a test can assert a checkpoint hit skipped the fetch entirely,
 * and accepts an injected `failListWith`/`failGetWith`/`failTagsWith`
 * error instance to prove transient/permanent-error handling without a
 * real AWS SDK error.
 */
export class FakeObjectStore implements ObjectStore {
  private readonly bucket: string;
  private readonly objects = new Map<string, StoredObject>();
  readonly getCalls = new Map<string, number>();
  readonly tagsCalls = new Map<string, number>();
  failListWith: Error | null = null;
  failGetWith: Error | null = null;
  failTagsWith: Error | null = null;

  /** Overridable so a test can inject a flaky/patched `get` — mirrors the
   * Python original's own `store.get = flaky_get`-style monkeypatch. */
  get: (key: string, options?: { versionId?: string }) => Promise<FetchedObject>;

  constructor(bucket = "test-bucket") {
    this.bucket = bucket;
    this.get = (key, options) => this.defaultGet(key, options);
  }

  get baseUri(): string {
    return `s3://${this.bucket}`;
  }

  /** Seeds (or replaces) one object. A later `put` of the same `key` is
   * how a test simulates "the object changed" between two sync passes. */
  put(
    key: string,
    body: Uint8Array,
    options?: {
      contentType?: string | null;
      etag?: string | null;
      versionId?: string | null;
      checksum?: string | null;
      lastModified?: Date | null;
      tags?: Record<string, string>;
    },
  ): void {
    this.objects.set(key, {
      body,
      contentType: options?.contentType ?? null,
      etag: options?.etag ?? null,
      versionId: options?.versionId ?? null,
      checksum: options?.checksum ?? null,
      lastModified: options?.lastModified ?? null,
      tags: options?.tags ?? {},
    });
  }

  /** Removes `key` — how a test simulates "the object was deleted between
   * two sync passes." */
  delete(key: string): void {
    this.objects.delete(key);
  }

  async *list(prefix: string): AsyncGenerator<ObjectMetaType> {
    if (this.failListWith !== null) {
      throw this.failListWith;
    }
    for (const key of [...this.objects.keys()].sort()) {
      if (!key.startsWith(prefix)) {
        continue;
      }
      const stored = this.objects.get(key)!;
      yield new ObjectMeta({
        key,
        size: stored.body.length,
        lastModified: stored.lastModified,
        etag: stored.etag,
        versionId: stored.versionId,
        checksum: stored.checksum,
      });
    }
  }

  private async defaultGet(key: string, options?: { versionId?: string }): Promise<FetchedObject> {
    this.getCalls.set(key, (this.getCalls.get(key) ?? 0) + 1);
    if (this.failGetWith !== null) {
      throw this.failGetWith;
    }
    const stored = this.objects.get(key);
    const versionId = options?.versionId;
    if (stored === undefined || (versionId !== undefined && stored.versionId !== versionId)) {
      throw new ObjectNotFoundError(`${JSON.stringify(key)} does not exist in this fake bucket`);
    }
    return new FetchedObject({ body: stored.body, contentType: stored.contentType });
  }

  async objectTags(key: string): Promise<Record<string, string>> {
    this.tagsCalls.set(key, (this.tagsCalls.get(key) ?? 0) + 1);
    if (this.failTagsWith !== null) {
      throw this.failTagsWith;
    }
    const stored = this.objects.get(key);
    return stored !== undefined ? { ...stored.tags } : {};
  }
}

/**
 * Lays `objects` (key → bytes) out under `root` as a plain directory tree —
 * what `FileObjectStore` (objectstore.ts) reads. `root` itself must already
 * exist (mirrors `FileObjectStore`'s/`src/ship.rs`'s own "the bucket must
 * exist" contract).
 */
export async function writeBucket(root: string, objects: Record<string, Uint8Array | string>): Promise<void> {
  const fsp = await import("node:fs/promises");
  const pathMod = await import("node:path");
  for (const [key, body] of Object.entries(objects)) {
    const path = pathMod.join(root, key);
    await fsp.mkdir(pathMod.dirname(path), { recursive: true });
    await fsp.writeFile(path, typeof body === "string" ? body : body);
  }
}
