/**
 * The `az://` backend of the object-storage boundary (ADR 0007 §13's
 * deferred Azure Blob connector, issue #414): `AzureBlobObjectStore`
 * implements the same `ObjectStore` (objectstore.ts) protocol
 * `S3ObjectStore` does, as an optional `@azure/storage-blob` +
 * `@azure/identity` peer dependency — completing `src/ship.rs`
 * `open_store`'s scheme set (`s3://`, `gs://`, `az://`, `file://`) on
 * this package's side. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.objectstore_azure` module (issue
 * #415).
 *
 * Credentials come from `DefaultAzureCredential` only — Azure's standard
 * chain (environment, workload identity, managed identity, Azure CLI, …),
 * the analog of `AmazonS3Builder::from_env()` (`src/ship.rs:212`) and of
 * `S3ObjectStore`'s standard-chain-only posture. `account` (the storage
 * account NAME, not a key) and `endpointUrl` (an Azurite or
 * sovereign-cloud endpoint) are the only construction knobs, and neither
 * is credential material; there is deliberately no parameter this class
 * could accept an account key or SAS token through. When `account` is
 * omitted it falls back to `$AZURE_STORAGE_ACCOUNT_NAME` (the variable
 * `MicrosoftAzureBuilder::from_env()` reads in `src/ship.rs`'s Rust path)
 * and then `$AZURE_STORAGE_ACCOUNT` (the Azure CLI's own spelling).
 *
 * Mappings onto the protocol's fingerprint tiers (ADR 0007 §9):
 * `versionId` is populated only when the container's versioning stamps
 * one on the listing — the same versioning-gated semantics as S3;
 * `checksum` carries the listing's `Content-MD5` when the service stored
 * one; `etag` stays the last resort. `objectTags` maps to **blob index
 * tags** — Azure's direct analog of S3 object tags, a separate API a
 * policy can deny narrowly, so the same tags-only degrade applies.
 *
 * Cloud SDK boundary: `@azure/storage-blob`/`@azure/identity` are
 * referenced ONLY via a dynamic `import()` inside `clientOrBuild`, never
 * at this module's top level. Error classification is STRUCTURAL
 * (`code`/`statusCode`/`name` fields), never `instanceof` a type from
 * either optional package.
 */

import {
  errorMessage,
  FetchedObject,
  type ObjectStore,
  ObjectMeta,
  ObjectNotFoundError,
  PermanentStoreError,
  TransientStoreError,
} from "./objectstore.js";

/** One listed Azure blob's properties, in the shape this store's client
 * seam reads — the TypeScript analogue of the real `@azure/storage-blob`
 * `BlobItem.properties`, camelCased and flattened. */
export interface AzureBlobPropertiesLike {
  readonly name: string;
  readonly size: number;
  readonly lastModified: Date | null;
  readonly etag: string | null;
  readonly versionId: string | null;
  readonly contentMd5: Uint8Array | null;
}

/** One `download`'s result — enough to read the body and content-type,
 * without committing to the real SDK's stream-vs-buffer surface. */
export interface AzureDownloadResultLike {
  readonly contentType: string | null;
  readAll(): Promise<Uint8Array>;
}

export interface AzureBlobClientLike {
  getTags(): Promise<Record<string, string> | null>;
}

/** The injectable client seam — `AzureBlobObjectStore`'s `client` escape
 * hatch, mirroring the real `@azure/storage-blob` `ContainerClient`'s
 * shape closely enough for a network-free fake to stand in for it in
 * tests. */
export interface AzureContainerClientLike {
  listBlobs(prefix?: string): AsyncIterable<AzureBlobPropertiesLike>;
  downloadBlob(key: string, versionId?: string): Promise<AzureDownloadResultLike>;
  getBlobClient(key: string): AzureBlobClientLike;
}

// ADR 0007 §9's explicit carve-out, Azure dress. `ContainerNotFound` sits
// beside the credential codes for the same reason `NoSuchBucket` does in
// the S3 backend: a missing container is a configuration error (a typo, a
// container never created), not one object vanishing mid-pass.
const AZURE_PERMANENT_ERROR_CODES: ReadonlySet<string> = new Set([
  "AuthenticationFailed",
  "AuthorizationFailure",
  "AuthorizationPermissionMismatch",
  "InsufficientAccountPermissions",
  "AccountIsDisabled",
  "InvalidAuthenticationInfo",
  "ContainerNotFound",
]);

const AZURE_PERMANENT_STATUS: ReadonlySet<number> = new Set([401, 403]);

// The structural stand-in for the Python twin's
// `isinstance(error, ClientAuthenticationError)` (which also catches
// `azure-identity`'s `CredentialUnavailableError` subclass) — a
// credential-resolution failure has no HTTP status at all, so it must be
// recognized by name rather than by `code`/`statusCode`. Extend this list
// if the installed SDK surfaces a different `error.name`.
const AZURE_AUTH_ERROR_NAMES: ReadonlySet<string> = new Set([
  "ClientAuthenticationError",
  "CredentialUnavailableError",
  "AuthenticationError",
]);

function errorField(error: unknown, key: string): unknown {
  if (typeof error !== "object" || error === null) {
    return undefined;
  }
  return (error as Record<string, unknown>)[key];
}

/**
 * One Azure failure → the protocol's exception vocabulary. As in the GCS
 * backend, `notFound` is what "does not exist" means at this call site —
 * but Azure names the missing thing itself (`ContainerNotFound` vs
 * `BlobNotFound`), so a missing container classifies permanent even on
 * the fetch path (checked via `code`, before the generic 404/`notFound`
 * branch below).
 */
function classifyAzureError(error: unknown, notFound: () => Error): Error {
  const name = errorField(error, "name");
  if (typeof name === "string" && AZURE_AUTH_ERROR_NAMES.has(name)) {
    return new PermanentStoreError(errorMessage(error));
  }
  const code = errorField(error, "code");
  if (typeof code === "string" && AZURE_PERMANENT_ERROR_CODES.has(code)) {
    return new PermanentStoreError(errorMessage(error));
  }
  const statusRaw = errorField(error, "statusCode");
  const status = typeof statusRaw === "number" ? statusRaw : undefined;
  if (status === 404) {
    return notFound();
  }
  if (status !== undefined && AZURE_PERMANENT_STATUS.has(status)) {
    return new PermanentStoreError(errorMessage(error));
  }
  // ServiceRequestError (connection/DNS/timeout, no status at all) and
  // every other unnamed failure stays transient — `src/ship.rs`'s
  // "assumed transient" default, narrowed only by the carve-outs above.
  return new TransientStoreError(errorMessage(error));
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Azure quotes `ETag` the same way S3 does; strip only a leading/trailing
 * quote run (mirroring Python's `str.strip('"')`), for the same reason
 * `objectstore.ts`'s S3 backend documents. */
function cleanEtag(etag: string | null): string | null {
  if (etag === null) {
    return null;
  }
  const stripped = etag.replace(/^"+/, "").replace(/"+$/, "");
  return stripped || null;
}

function metaOfBlob(blob: AzureBlobPropertiesLike): ObjectMeta {
  const md5 = blob.contentMd5;
  const checksum = md5 && md5.length > 0 ? `md5:${bytesToHex(md5)}` : null;
  return new ObjectMeta({
    key: blob.name,
    size: blob.size,
    lastModified: blob.lastModified,
    etag: cleanEtag(blob.etag),
    versionId: blob.versionId,
    checksum,
  });
}

function resolveAccountUrl(account: string | undefined, endpointUrl: string | undefined): string {
  if (endpointUrl !== undefined) {
    const parsed = new URL(endpointUrl);
    if (parsed.username || parsed.password) {
      throw new Error("endpointUrl must not embed credentials");
    }
    return endpointUrl;
  }
  const resolvedAccount =
    account ?? process.env["AZURE_STORAGE_ACCOUNT_NAME"] ?? process.env["AZURE_STORAGE_ACCOUNT"];
  if (!resolvedAccount) {
    throw new Error(
      "AzureBlobObjectStore needs the storage account name — pass account, or set " +
        "AZURE_STORAGE_ACCOUNT_NAME (the same variable the server's replication path reads)",
    );
  }
  return `https://${resolvedAccount}.blob.core.windows.net`;
}

/** Minimal shape of the real `@azure/storage-blob` `ContainerClient` this
 * module's real (non-injected) construction path adapts onto
 * `AzureContainerClientLike` — unverified against the live package (an
 * optional peer dependency not installed in this workspace) and
 * exercised by no test here; a caller who wants the real client behavior
 * proven should inject one they already trust via `client`. */
interface RealAzureBlobItem {
  name: string;
  properties: {
    contentLength?: number;
    lastModified?: Date;
    etag?: string;
    contentMD5?: Uint8Array;
  };
  versionId?: string;
  isCurrentVersion?: boolean;
}
interface RealAzureDownloadResponse {
  contentType?: string;
  readableStreamBody?: NodeJS.ReadableStream;
}
interface RealAzureBlobClient {
  download(): Promise<RealAzureDownloadResponse>;
  getTags(): Promise<{ tags: Record<string, string> } | undefined>;
}
interface RealAzureContainerClient {
  listBlobsFlat(options?: {
    prefix?: string;
    includeVersions?: boolean;
  }): AsyncIterable<RealAzureBlobItem>;
  getBlobClient(key: string): RealAzureBlobClient;
}

async function streamToBytes(stream: NodeJS.ReadableStream): Promise<Uint8Array> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream as AsyncIterable<Buffer | string>) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  const buffer = Buffer.concat(chunks);
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

function adaptRealContainerClient(real: RealAzureContainerClient): AzureContainerClientLike {
  return {
    async *listBlobs(prefix) {
      // Without `includeVersions`, `BlobItem.versionId` is never populated,
      // so a versioning-enabled container could not use objectFingerprint's
      // top tier. With it, every version is listed — keep only the current
      // one per blob; a container without versioning yields items with no
      // `versionId`/`isCurrentVersion` at all, which must still pass.
      for await (const item of real.listBlobsFlat({ prefix, includeVersions: true })) {
        if (item.versionId !== undefined && item.isCurrentVersion !== true) {
          continue;
        }
        yield {
          name: item.name,
          size: item.properties.contentLength ?? 0,
          lastModified: item.properties.lastModified ?? null,
          etag: item.properties.etag ?? null,
          versionId: item.versionId ?? null,
          contentMd5: item.properties.contentMD5 ?? null,
        };
      }
    },
    async downloadBlob(key) {
      const response = await real.getBlobClient(key).download();
      return {
        contentType: response.contentType ?? null,
        readAll: async () =>
          response.readableStreamBody ? streamToBytes(response.readableStreamBody) : new Uint8Array(),
      };
    },
    getBlobClient(key) {
      const blobClient = real.getBlobClient(key);
      return {
        async getTags() {
          const result = await blobClient.getTags();
          return result?.tags ?? null;
        },
      };
    },
  };
}

/**
 * Reads/lists one Azure Blob container via `@azure/storage-blob`,
 * authenticated through `DefaultAzureCredential` only (the standard
 * chain) — `account` names the storage account, `endpointUrl` overrides
 * the whole account URL for an Azurite or sovereign-cloud endpoint, and
 * neither is credential material; there is deliberately no parameter this
 * class could accept an account key or SAS token through.
 *
 * `client` is the same escape hatch the S3/GCS backends document: a
 * caller who already has a configured `ContainerClient` (custom
 * retry/transport config, a connection-string-built client of their own,
 * or — this package's own tests — a network-free fake) passes it here,
 * mutually exclusive with `account`/`endpointUrl`, which exist only to
 * build one.
 */
export class AzureBlobObjectStore implements ObjectStore {
  private readonly container: string;
  private client: AzureContainerClientLike | null;
  private readonly accountUrl: string | null;

  constructor(
    container: string,
    options?: { account?: string; endpointUrl?: string; client?: AzureContainerClientLike },
  ) {
    const { account, endpointUrl, client } = options ?? {};
    if (client !== undefined) {
      if (account !== undefined || endpointUrl !== undefined) {
        throw new Error("client is mutually exclusive with account/endpointUrl");
      }
      this.container = container;
      this.client = client;
      this.accountUrl = null;
      return;
    }
    // Resolved (and validated) eagerly, unlike the real client below —
    // this is plain string/env logic, no I/O, so it can stay synchronous
    // and preserve the Python twin's construction-time throw for a
    // missing account name or a credential-embedding endpoint.
    this.accountUrl = resolveAccountUrl(account, endpointUrl);
    this.container = container;
    this.client = null;
  }

  get baseUri(): string {
    return `az://${this.container}`;
  }

  /**
   * Built on first use, not in the constructor. The Python twin builds
   * its real `ContainerClient` (including `DefaultAzureCredential()`)
   * EAGERLY in `__init__` — but `await import(...)` cannot run inside a
   * synchronous constructor, so this port defers both "the package isn't
   * installed" and "credentials can't be resolved" to first use here.
   * Constructing a store object still touches neither the package
   * registry nor the environment when a `client` was injected.
   */
  private async clientOrBuild(): Promise<AzureContainerClientLike> {
    if (this.client !== null) {
      return this.client;
    }
    const blobModuleSpecifier = "@azure/storage-blob";
    const identityModuleSpecifier = "@azure/identity";
    let blobModule: {
      ContainerClient: new (url: string, credential: unknown) => RealAzureContainerClient;
    };
    let identityModule: { DefaultAzureCredential: new () => unknown };
    try {
      [blobModule, identityModule] = await Promise.all([
        import(blobModuleSpecifier) as Promise<typeof blobModule>,
        import(identityModuleSpecifier) as Promise<typeof identityModule>,
      ]);
    } catch (error) {
      throw new Error(
        "AzureBlobObjectStore requires @azure/storage-blob and @azure/identity — install " +
          `them as dependencies (npm install @azure/storage-blob @azure/identity): ${errorMessage(error)}`,
      );
    }
    try {
      const credential = new identityModule.DefaultAzureCredential();
      const real = new blobModule.ContainerClient(`${this.accountUrl}/${this.container}`, credential);
      this.client = adaptRealContainerClient(real);
    } catch (error) {
      throw new PermanentStoreError(errorMessage(error));
    }
    return this.client;
  }

  async *list(prefix: string): AsyncGenerator<ObjectMeta> {
    const client = await this.clientOrBuild();
    try {
      for await (const blob of client.listBlobs(prefix)) {
        yield metaOfBlob(blob);
      }
    } catch (error) {
      // A 404 here is the CONTAINER missing — permanent, like
      // `NoSuchBucket`; `list()`'s contract never raises
      // `ObjectNotFoundError`.
      throw classifyAzureError(error, () => new PermanentStoreError(errorMessage(error)));
    }
  }

  async get(key: string, options?: { versionId?: string }): Promise<FetchedObject> {
    const client = await this.clientOrBuild();
    try {
      const downloader = await client.downloadBlob(key, options?.versionId);
      const body = await downloader.readAll();
      return new FetchedObject({ body, contentType: downloader.contentType });
    } catch (error) {
      throw classifyAzureError(error, () => new ObjectNotFoundError(errorMessage(error)));
    }
  }

  /**
   * The blob's index tags — Azure's direct analog of S3 object tags,
   * down to being a separate permission (`Microsoft.Storage/…/blobs/
   * tags/read`) a policy can deny while the blob itself stays readable,
   * so the same tags-only degrade to `{}` applies (ADR 0007 §9).
   */
  async objectTags(key: string): Promise<Record<string, string>> {
    try {
      const client = await this.clientOrBuild();
      const tags = await client.getBlobClient(key).getTags();
      return { ...(tags ?? {}) };
    } catch (error) {
      if (error instanceof PermanentStoreError) {
        return {};
      }
      const classified = classifyAzureError(error, () => new ObjectNotFoundError(errorMessage(error)));
      if (classified instanceof TransientStoreError) {
        throw classified;
      }
      return {};
    }
  }
}
