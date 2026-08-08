/**
 * The `gs://` backend of the object-storage boundary (ADR 0007 §13's
 * deferred GCS connector, issue #414): `GCSObjectStore` implements the
 * same `ObjectStore` (objectstore.ts) protocol `S3ObjectStore` does, as an
 * optional `@google-cloud/storage` peer dependency — so `src/ship.rs`'s
 * `open_store` scheme set (`s3://`, `gs://`, `az://`, `file://`) and this
 * package's own `openObjectStore` stop disagreeing about which clouds
 * exist. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.objectstore_gcs` module (issue
 * #415).
 *
 * Credentials come from Application Default Credentials only — the GCS
 * analog of `AmazonS3Builder::from_env()` (`src/ship.rs:212`) and of
 * `S3ObjectStore`'s standard-chain-only posture. `project` is the one
 * construction knob and is not credential material; there is deliberately
 * no parameter this class could accept a service-account key through.
 *
 * Two GCS-specific mappings, both strictly better fingerprints than S3's:
 *
 * - `versionId` carries the blob's **generation** on every listing — GCS
 *   stamps a new generation on every content overwrite whether or not the
 *   bucket has object versioning enabled, so the top fingerprint tier (ADR
 *   0007 §9) is always available here, not gated on a bucket setting the
 *   way S3's `VersionId` is. Fetching a listed generation after the
 *   object was overwritten (versioning off) surfaces as
 *   `ObjectNotFoundError` — exactly the "skip this object this pass" the
 *   protocol already means by it; the new content lists under its new
 *   generation on the next pass.
 * - `checksum` is populated AT LISTING TIME (`crc32c`, falling back to
 *   `md5Hash`) — the tier `ObjectMeta` documents as "only when a store
 *   cheaply returns one at listing time; real S3 does not." GCS does.
 *
 * `objectTags` maps to the blob's **custom metadata** (user key/value
 * pairs): GCS has no separate object-tagging API, and custom metadata is
 * the same operator-attached, non-content key/value surface S3 tags are —
 * the downstream use (source tags, `_mapped_tags`) reads identically.
 *
 * Cloud SDK boundary: `@google-cloud/storage` is referenced ONLY via a
 * dynamic `import()` inside `clientOrBuild`, never at this module's top
 * level, so the package need not be installed at all unless a caller
 * actually constructs a store without an injected `client`. Error
 * classification is STRUCTURAL (`code`/`name` fields), never `instanceof`
 * a type from that optional package — the same fields a fake test client
 * and the real SDK's own errors both surface.
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

/**
 * One listed/fetched GCS object, in the shape this store's own client
 * seam reads — the TypeScript analogue of the real `@google-cloud/
 * storage` `File`'s metadata fields, camelCased. A test injects a fake
 * implementing this directly (`GCSObjectStore`'s own `client` escape
 * hatch); the real (non-injected) construction path adapts an actual
 * `Storage` client instance onto it.
 */
export interface GCSBlobLike {
  readonly name: string | null;
  readonly size: number | null;
  readonly updated: Date | null;
  readonly etag: string | null;
  readonly generation: string | number | null;
  readonly crc32c: string | null;
  readonly md5Hash: string | null;
  readonly contentType: string | null;
  readonly metadata: Record<string, string> | null;
  download(): Promise<Uint8Array>;
}

export interface GCSBucketLike {
  getBlob(key: string, generation?: string | null): Promise<GCSBlobLike | null>;
}

/** The injectable client seam — `GCSObjectStore`'s `client` escape hatch,
 * mirroring the real `@google-cloud/storage` `Storage` client's shape
 * closely enough for a network-free fake to stand in for it in tests. */
export interface GCSClientLike {
  listBlobs(bucket: string, prefix: string): AsyncIterable<GCSBlobLike>;
  bucket(name: string): GCSBucketLike;
}

// ADR 0007 §9's explicit carve-out, GCS dress: credential problems and
// permission denial are terminal; everything else keeps `src/ship.rs`'s
// "assumed transient" default. 404 is NOT here — a missing bucket is
// permanent but a missing object is `ObjectNotFoundError`, and only the
// call site knows which of the two it asked about (`classifyGcsError`
// takes `notFound`).
//
// A real `@google-cloud/storage`/`google-auth-library` credential-
// resolution failure is not, in general, distinguishable from any other
// plain `Error` by a numeric status code alone — unlike an API-call
// failure (which always carries one). This structural allowlist of
// `error.name` values stands in for the Python twin's
// `isinstance(error, GoogleAuthError)` check without importing that class
// from the optional dependency; extend it if the installed SDK surfaces a
// different name.
const GCS_AUTH_ERROR_NAMES: ReadonlySet<string> = new Set([
  "GoogleAuthError",
  "DefaultCredentialsError",
]);

const GCS_PERMANENT_STATUS: ReadonlySet<number> = new Set([401, 403]);

/**
 * One GCS failure → the protocol's exception vocabulary. `notFound` is
 * what a 404 means AT THIS CALL SITE: a `PermanentStoreError` factory for
 * a listing (the bucket itself is a typo or was never created — retrying
 * forever would mask a fixable configuration problem), an
 * `ObjectNotFoundError` factory for a fetch (the object vanished between
 * listing and get — skip it this pass).
 */
function classifyGcsError(error: unknown, notFound: () => Error): Error {
  const name = errorField(error, "name");
  if (typeof name === "string" && GCS_AUTH_ERROR_NAMES.has(name)) {
    return new PermanentStoreError(errorMessage(error));
  }
  const code = errorField(error, "code");
  const status = typeof code === "number" ? code : undefined;
  if (status !== undefined && GCS_PERMANENT_STATUS.has(status)) {
    return new PermanentStoreError(errorMessage(error));
  }
  if (status === 404) {
    return notFound();
  }
  // Everything unnamed stays transient (connection resets, timeouts, any
  // other API-call status) — the same narrowed-only-by-carve-out posture
  // `objectstore.ts`'s S3 backend takes.
  return new TransientStoreError(errorMessage(error));
}

function errorField(error: unknown, key: string): unknown {
  if (typeof error !== "object" || error === null) {
    return undefined;
  }
  return (error as Record<string, unknown>)[key];
}

function metaOfBlob(blob: GCSBlobLike): ObjectMeta {
  const generation = blob.generation;
  let checksum: string | null;
  if (blob.crc32c) {
    checksum = `crc32c:${blob.crc32c}`;
  } else if (blob.md5Hash) {
    checksum = `md5:${blob.md5Hash}`;
  } else {
    checksum = null;
  }
  return new ObjectMeta({
    key: blob.name ?? "",
    size: blob.size ?? 0,
    lastModified: blob.updated,
    etag: blob.etag,
    versionId: generation ? String(generation) : null,
    checksum,
  });
}

/** Minimal shape of the real `@google-cloud/storage` `File`/`Bucket`/
 * `Storage` classes this module's real (non-injected) construction path
 * adapts onto `GCSClientLike` — unverified against the live package
 * (an optional peer dependency not installed in this workspace) and
 * exercised by no test here; a caller who wants the real client behavior
 * proven should inject one they already trust via `client`. */
interface RealGCSFile {
  name: string;
  metadata: Record<string, unknown>;
  getMetadata(): Promise<[Record<string, unknown>]>;
  download(): Promise<[Buffer]>;
}
interface RealGCSBucket {
  getFiles(options: { prefix: string }): Promise<[RealGCSFile[]]>;
  file(name: string, options?: { generation?: string }): RealGCSFile;
}
interface RealGCSStorage {
  bucket(name: string): RealGCSBucket;
}

function adaptRealFile(file: RealGCSFile, metadata: Record<string, unknown>): GCSBlobLike {
  return {
    name: file.name,
    size: metadata.size !== undefined ? Number(metadata.size) : null,
    updated: typeof metadata.updated === "string" ? new Date(metadata.updated) : null,
    etag: typeof metadata.etag === "string" ? metadata.etag : null,
    generation:
      typeof metadata.generation === "string" || typeof metadata.generation === "number"
        ? metadata.generation
        : null,
    crc32c: typeof metadata.crc32c === "string" ? metadata.crc32c : null,
    md5Hash: typeof metadata.md5Hash === "string" ? metadata.md5Hash : null,
    contentType: typeof metadata.contentType === "string" ? metadata.contentType : null,
    metadata: (metadata.metadata as Record<string, string> | undefined) ?? null,
    download: async () => {
      const [buffer] = await file.download();
      return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    },
  };
}

function adaptRealClient(real: RealGCSStorage): GCSClientLike {
  return {
    async *listBlobs(bucketName, prefix) {
      const [files] = await real.bucket(bucketName).getFiles({ prefix });
      for (const file of files) {
        yield adaptRealFile(file, file.metadata ?? {});
      }
    },
    bucket(name) {
      const realBucket = real.bucket(name);
      return {
        async getBlob(key, generation) {
          const file = realBucket.file(key, generation ? { generation } : undefined);
          try {
            const [metadata] = await file.getMetadata();
            return adaptRealFile(file, metadata);
          } catch (error) {
            if (errorField(error, "code") === 404) {
              return null;
            }
            throw error;
          }
        },
      };
    },
  };
}

/**
 * Reads/lists one GCS bucket via `@google-cloud/storage`, built from
 * Application Default Credentials only — `project` is the only
 * construction knob, and it is not credential material; there is
 * deliberately no parameter this class could accept a service-account key
 * through.
 *
 * `client` is the same escape hatch `S3ObjectStore` documents: a caller
 * who already has a configured client (custom retry/timeout config, an
 * emulator endpoint, or — this package's own tests — a network-free fake)
 * passes it here, mutually exclusive with `project`, which exists only to
 * build one.
 */
export class GCSObjectStore implements ObjectStore {
  private readonly bucketName: string;
  private readonly project: string | null;
  private client: GCSClientLike | null;

  constructor(bucket: string, options?: { project?: string; client?: GCSClientLike }) {
    const { project, client } = options ?? {};
    if (client !== undefined && project !== undefined) {
      throw new Error("client is mutually exclusive with project");
    }
    this.bucketName = bucket;
    this.project = project ?? null;
    this.client = client ?? null;
  }

  get baseUri(): string {
    return `gs://${this.bucketName}`;
  }

  /**
   * Built on first use, not in the constructor: unlike the Python twin
   * (whose `storage` import is resolved eagerly at module load, so
   * `__init__` can raise `ImportError` synchronously and `storage.Client`
   * resolution failures are the only thing deferred to
   * `_client_or_build`), this port cannot `await` a dynamic `import()`
   * inside a synchronous constructor — BOTH "the package isn't installed"
   * and "credentials can't be resolved" are therefore deferred here to
   * first use, one stage later than the Python twin's construction-time
   * package check. Constructing a store object still touches neither the
   * environment nor raises anything when a `client` was injected.
   */
  private async clientOrBuild(): Promise<GCSClientLike> {
    if (this.client !== null) {
      return this.client;
    }
    const moduleSpecifier = "@google-cloud/storage";
    let storageModule: { Storage: new (options?: { projectId?: string }) => RealGCSStorage };
    try {
      storageModule = (await import(moduleSpecifier)) as typeof storageModule;
    } catch (error) {
      throw new Error(
        "GCSObjectStore requires @google-cloud/storage — install it as a dependency " +
          `(npm install @google-cloud/storage): ${errorMessage(error)}`,
      );
    }
    try {
      const real = new storageModule.Storage({ projectId: this.project ?? undefined });
      this.client = adaptRealClient(real);
    } catch (error) {
      throw new PermanentStoreError(errorMessage(error));
    }
    return this.client;
  }

  async *list(prefix: string): AsyncGenerator<ObjectMeta> {
    const client = await this.clientOrBuild();
    try {
      for await (const blob of client.listBlobs(this.bucketName, prefix)) {
        yield metaOfBlob(blob);
      }
    } catch (error) {
      // A 404 here is the BUCKET missing — a configuration error, never
      // an `ObjectNotFoundError`, which `list()`'s contract promises not
      // to raise.
      throw classifyGcsError(error, () => new PermanentStoreError(errorMessage(error)));
    }
  }

  async get(key: string, options?: { versionId?: string }): Promise<FetchedObject> {
    const generation = options?.versionId ?? null;
    const client = await this.clientOrBuild();
    let blob: GCSBlobLike | null;
    try {
      blob = await client.bucket(this.bucketName).getBlob(key, generation);
    } catch (error) {
      throw classifyGcsError(error, () => new ObjectNotFoundError(errorMessage(error)));
    }
    if (blob === null) {
      throw new ObjectNotFoundError(`gs://${this.bucketName}/${key}: no such object`);
    }
    // The download itself must classify too — a network drop, timeout, or
    // permission denial here would otherwise propagate as a raw Error,
    // which `syncObjectStorage`'s per-object catch (transient/permanent/
    // not-found only) treats as fatal for the whole run.
    try {
      const body = await blob.download();
      return new FetchedObject({ body, contentType: blob.contentType });
    } catch (error) {
      throw classifyGcsError(error, () => new ObjectNotFoundError(errorMessage(error)));
    }
  }

  /**
   * The blob's custom metadata — GCS's operator-attached key/value
   * surface, standing in for S3's object tags (no separate tagging API
   * exists to be denied narrowly, but the same degrade applies: any
   * failure reading it yields `{}`, never a failed object).
   */
  async objectTags(key: string): Promise<Record<string, string>> {
    let blob: GCSBlobLike | null;
    try {
      const client = await this.clientOrBuild();
      blob = await client.bucket(this.bucketName).getBlob(key);
    } catch (error) {
      if (error instanceof PermanentStoreError) {
        return {};
      }
      const classified = classifyGcsError(error, () => new ObjectNotFoundError(errorMessage(error)));
      if (classified instanceof TransientStoreError) {
        // Transient failures re-raise (retry next pass) exactly as the
        // S3 backend's `objectTags` does; only the permanent/not-found
        // cases degrade to "no tags."
        throw classified;
      }
      return {};
    }
    if (blob === null || blob.metadata === null) {
      return {};
    }
    return { ...blob.metadata };
  }
}
