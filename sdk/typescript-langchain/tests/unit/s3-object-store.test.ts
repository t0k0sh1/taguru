/**
 * S3ObjectStore.objectTags — the tags-only permission degrade and the
 * Transient/Permanent/NotFound classification behind it (issue #737):
 * the GCS and Azure backends each carry a dedicated test of this exact
 * split, and S3's had none. Driven through an injected S3ClientLike
 * whose getObjectTagging throws the shapes `classifyS3Error` sorts.
 */

import { expect, test } from "vitest";

import { S3ObjectStore, type S3ClientLike } from "../../src/ingest-connectors/objectstore-s3.js";
import { TransientStoreError } from "../../src/ingest-connectors/objectstore.js";

function storeWhoseTaggingThrows(error: unknown): S3ObjectStore {
  const client = {
    async getObjectTagging(_bucket: string, _key: string): Promise<Record<string, string>> {
      throw error;
    },
  } as unknown as S3ClientLike;
  return new S3ObjectStore("bucket", { client });
}

test("objectTags maps the object's tag set and passes the bucket and key through", async () => {
  const seen: Array<[string, string]> = [];
  const client = {
    async getObjectTagging(bucket: string, key: string): Promise<Record<string, string>> {
      seen.push([bucket, key]);
      return { team: "ingest", quarter: "q1" };
    },
  } as unknown as S3ClientLike;
  const store = new S3ObjectStore("bucket", { client });
  expect(await store.objectTags("a.txt")).toEqual({ team: "ingest", quarter: "q1" });
  expect(seen).toEqual([["bucket", "a.txt"]]);
});

test("objectTags degrades to empty for a vanished object", async () => {
  const store = storeWhoseTaggingThrows({ name: "NoSuchKey", message: "gone" });
  expect(await store.objectTags("gone.txt")).toEqual({});
});

test("objectTags degrades to empty on a tags-only permission gap", async () => {
  // Both spellings of "not allowed": the named S3 error and a bare 403.
  const denied = storeWhoseTaggingThrows({ name: "AccessDenied", message: "denied" });
  expect(await denied.objectTags("a.txt")).toEqual({});
  const forbidden = storeWhoseTaggingThrows({ message: "403", $metadata: { httpStatusCode: 403 } });
  expect(await forbidden.objectTags("a.txt")).toEqual({});
});

test("objectTags raises on transient trouble instead of faking an empty set", async () => {
  const busy = storeWhoseTaggingThrows({ message: "503", $metadata: { httpStatusCode: 503 } });
  await expect(busy.objectTags("a.txt")).rejects.toBeInstanceOf(TransientStoreError);
});
