/**
 * `AzureBlobObjectStore` (issue #414): the `az://` backend of the
 * object-storage boundary, exercised through plain fake objects that
 * expose the same STRUCTURAL fields (`code`/`statusCode`/`name`) a real
 * `@azure/storage-blob`/`@azure/core-rest-pipeline` error would, injected
 * through the class's own documented `client` escape hatch — the
 * TypeScript analogue of the Python original's "real library, no live
 * container" posture (which exercises the actual `azure-core` exception
 * classes); `@azure/storage-blob`/`@azure/identity` are optional peer
 * dependencies not installed in this workspace, so this suite never
 * imports them. TypeScript parity: issue #415.
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
  AzureBlobObjectStore,
  type AzureBlobPropertiesLike,
  type AzureContainerClientLike,
  type AzureDownloadResultLike,
} from "../../src/ingest-connectors/objectstore-azure.js";

const MODIFIED = new Date("2026-03-01T00:00:00Z");

async function collect<T>(iterable: AsyncIterable<T>): Promise<T[]> {
  const items: T[] = [];
  for await (const item of iterable) {
    items.push(item);
  }
  return items;
}

function properties(
  overrides: Partial<AzureBlobPropertiesLike> = {},
): AzureBlobPropertiesLike {
  return {
    name: "docs/a.txt",
    size: 5,
    lastModified: MODIFIED,
    etag: '"0x8D-etag"',
    versionId: null,
    contentMd5: new Uint8Array([0x01, 0xab]),
    ...overrides,
  };
}

class FakeDownloadResult implements AzureDownloadResultLike {
  constructor(
    private readonly bytes: Uint8Array = new TextEncoder().encode("hello"),
    readonly contentType: string | null = "text/plain",
  ) {}

  async readAll(): Promise<Uint8Array> {
    return this.bytes;
  }
}

class FakeBlobClient {
  tags: Record<string, string> | null = null;
  error: unknown = null;

  async getTags(): Promise<Record<string, string> | null> {
    if (this.error !== null) {
      throw this.error;
    }
    return this.tags;
  }
}

class FakeContainerClient implements AzureContainerClientLike {
  listed: AzureBlobPropertiesLike[] = [];
  listError: unknown = null;
  downloadError: unknown = null;
  downloader: AzureDownloadResultLike = new FakeDownloadResult();
  blobClient: FakeBlobClient = new FakeBlobClient();
  seenPrefix: (string | undefined)[] = [];
  seenVersion: (string | undefined)[] = [];

  async *listBlobs(prefix?: string): AsyncGenerator<AzureBlobPropertiesLike> {
    this.seenPrefix.push(prefix);
    if (this.listError !== null) {
      throw this.listError;
    }
    for (const item of this.listed) {
      yield item;
    }
  }

  async downloadBlob(_key: string, versionId?: string): Promise<AzureDownloadResultLike> {
    this.seenVersion.push(versionId);
    if (this.downloadError !== null) {
      throw this.downloadError;
    }
    return this.downloader;
  }

  getBlobClient(_key: string): FakeBlobClient {
    return this.blobClient;
  }
}

function store(client: AzureContainerClientLike): AzureBlobObjectStore {
  return new AzureBlobObjectStore("my-container", { client });
}

function coded(overrides: Record<string, unknown>): unknown {
  return { name: "RestError", ...overrides };
}

test("client is mutually exclusive with account and endpoint", () => {
  expect(
    () => new AzureBlobObjectStore("c", { account: "acct", client: new FakeContainerClient() }),
  ).toThrow(/mutually exclusive/);
});

test("base_uri names the container", () => {
  expect(store(new FakeContainerClient()).baseUri).toBe("az://my-container");
});

test("construction requires an account name", () => {
  const savedName = process.env["AZURE_STORAGE_ACCOUNT_NAME"];
  const savedAccount = process.env["AZURE_STORAGE_ACCOUNT"];
  delete process.env["AZURE_STORAGE_ACCOUNT_NAME"];
  delete process.env["AZURE_STORAGE_ACCOUNT"];
  try {
    expect(() => new AzureBlobObjectStore("c")).toThrow(/AZURE_STORAGE_ACCOUNT_NAME/);
  } finally {
    if (savedName !== undefined) {
      process.env["AZURE_STORAGE_ACCOUNT_NAME"] = savedName;
    }
    if (savedAccount !== undefined) {
      process.env["AZURE_STORAGE_ACCOUNT"] = savedAccount;
    }
  }
});

test("endpoint_url must not embed credentials", () => {
  expect(
    () => new AzureBlobObjectStore("c", { endpointUrl: "https://key:secret@example.com" }),
  ).toThrow(/must not embed credentials/);
});

// ---------------------------------------------------------------------------
// list — field mapping onto ObjectMeta and its fingerprint tiers
// ---------------------------------------------------------------------------

test("list maps properties and strips the quoted etag", async () => {
  const client = new FakeContainerClient();
  client.listed = [properties()];
  const [m] = await collect(store(client).list("docs/"));
  expect(client.seenPrefix).toEqual(["docs/"]);
  expect(m!.key).toBe("docs/a.txt");
  expect(m!.size).toBe(5);
  expect(m!.lastModified).toEqual(MODIFIED);
  expect(m!.etag).toBe("0x8D-etag");
  expect(m!.versionId).toBeNull();
  expect(m!.checksum).toBe("md5:01ab");
  expect(objectFingerprint(m!)).toEqual(["checksum", "md5:01ab"]);
});

test("list version_id rides when the container stamps one", async () => {
  const client = new FakeContainerClient();
  client.listed = [properties({ versionId: "2026-03-01T00:00:00Z" })];
  const [m] = await collect(store(client).list(""));
  expect(objectFingerprint(m!)).toEqual(["version_id", "2026-03-01T00:00:00Z"]);
});

test("list without content_md5 degrades checksum to null", async () => {
  const client = new FakeContainerClient();
  client.listed = [properties({ contentMd5: null })];
  const [m] = await collect(store(client).list(""));
  expect(m!.checksum).toBeNull();
});

test("list on a missing container is permanent, not object not found", async () => {
  const client = new FakeContainerClient();
  client.listError = coded({ code: "ContainerNotFound", message: "no container" });
  await expect(collect(store(client).list(""))).rejects.toBeInstanceOf(PermanentStoreError);
});

test.each([
  [
    "ClientAuthenticationError",
    { name: "ClientAuthenticationError", message: "bad credential" },
    PermanentStoreError,
  ],
  [
    "AuthorizationPermissionMismatch",
    coded({ code: "AuthorizationPermissionMismatch", statusCode: 403, message: "denied" }),
    PermanentStoreError,
  ],
  ["ServiceRequestError", { name: "ServiceRequestError", message: "dns" }, TransientStoreError],
  [
    "ServerBusy",
    coded({ code: "ServerBusy", statusCode: 503, message: "busy" }),
    TransientStoreError,
  ],
] as const)("list error classification: %s", async (_name, error, expected) => {
  const client = new FakeContainerClient();
  client.listError = error;
  await expect(collect(store(client).list(""))).rejects.toBeInstanceOf(expected);
});

// ---------------------------------------------------------------------------
// get — bytes + content type, version pinning, not-found semantics
// ---------------------------------------------------------------------------

test("get returns bytes, content type, and pins the listed version", async () => {
  const client = new FakeContainerClient();
  const fetched = await store(client).get("docs/a.txt", { versionId: "v1" });
  expect(fetched.body).toEqual(new TextEncoder().encode("hello"));
  expect(fetched.contentType).toBe("text/plain");
  expect(client.seenVersion).toEqual(["v1"]);
});

test("get missing blob is object not found", async () => {
  const client = new FakeContainerClient();
  client.downloadError = coded({ code: "BlobNotFound", statusCode: 404, message: "no blob" });
  await expect(store(client).get("gone.txt")).rejects.toBeInstanceOf(ObjectNotFoundError);
});

test("get on a missing container stays permanent", async () => {
  // Azure names the missing thing itself, so even the fetch path can tell
  // "the container is a typo" (permanent) apart from "this one blob
  // vanished" (skip this pass).
  const client = new FakeContainerClient();
  client.downloadError = coded({
    code: "ContainerNotFound",
    statusCode: 404,
    message: "no container",
  });
  await expect(store(client).get("a")).rejects.toBeInstanceOf(PermanentStoreError);
});

// ---------------------------------------------------------------------------
// objectTags — blob index tags, with the tags-only degrade
// ---------------------------------------------------------------------------

test("objectTags reads blob index tags", async () => {
  const client = new FakeContainerClient();
  client.blobClient.tags = { team: "ingest" };
  expect(await store(client).objectTags("a")).toEqual({ team: "ingest" });
});

test("objectTags degrades to empty on null, denial, or missing", async () => {
  const nullClient = new FakeContainerClient();
  nullClient.blobClient.tags = null;
  expect(await store(nullClient).objectTags("a")).toEqual({});

  const deniedClient = new FakeContainerClient();
  deniedClient.blobClient.error = coded({
    code: "AuthorizationPermissionMismatch",
    statusCode: 403,
    message: "tags denied",
  });
  expect(await store(deniedClient).objectTags("a")).toEqual({});

  const goneClient = new FakeContainerClient();
  goneClient.blobClient.error = coded({ statusCode: 404, message: "no blob" });
  expect(await store(goneClient).objectTags("a")).toEqual({});
});

test("objectTags raises on transient failures", async () => {
  const client = new FakeContainerClient();
  client.blobClient.error = { name: "ServiceRequestError", message: "dns" };
  await expect(store(client).objectTags("a")).rejects.toBeInstanceOf(TransientStoreError);
});

// ---------------------------------------------------------------------------
// openObjectStore — the az:// scheme
// ---------------------------------------------------------------------------

test("openObjectStore dispatches az urls", async () => {
  const saved = process.env["AZURE_STORAGE_ACCOUNT_NAME"];
  process.env["AZURE_STORAGE_ACCOUNT_NAME"] = "myaccount";
  try {
    const [s, prefix] = await openObjectStore("az://my-container/reports/2026");
    expect(s).toBeInstanceOf(AzureBlobObjectStore);
    expect(s.baseUri).toBe("az://my-container");
    expect(prefix).toBe("reports/2026");
  } finally {
    if (saved !== undefined) {
      process.env["AZURE_STORAGE_ACCOUNT_NAME"] = saved;
    } else {
      delete process.env["AZURE_STORAGE_ACCOUNT_NAME"];
    }
  }
});

test("openObjectStore requires an az container", async () => {
  await expect(openObjectStore("az:///prefix-only")).rejects.toThrow(/must name a container/);
});

test("openObjectStore rejects s3-only knobs for az", async () => {
  await expect(openObjectStore("az://container", { profileName: "prod" })).rejects.toThrow(
    /does not accept/,
  );
});
