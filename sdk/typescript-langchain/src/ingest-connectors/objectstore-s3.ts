/**
 * The `s3://` backend of the object-storage boundary (ADR 0007 §9, issue
 * #351): `S3ObjectStore` implements the same `ObjectStore` (objectstore.ts)
 * protocol `GCSObjectStore`/`AzureBlobObjectStore` do, as an optional
 * `@aws-sdk/client-s3` peer dependency. Unlike the Python original — where
 * `S3ObjectStore` lives inside `objectstore.py` itself and is imported
 * eagerly — this port splits it into its own module, dynamic-imported by
 * `objectstore.ts`'s `openObjectStore` only on the `s3://` branch, the same
 * `objectstore-<cloud>.ts` convention `objectstore-gcs.ts`/
 * `objectstore-azure.ts` already established (issue #414). The mechanical
 * mirror of the Python `taguru_langchain.ingest_connectors.objectstore`
 * module's own `S3ObjectStore` section (issue #415).
 *
 * Credentials come from the standard AWS credential chain only (the analog
 * of `AmazonS3Builder::from_env()`, `src/ship.rs:212`) — `endpointUrl` (an
 * S3-compatible service, e.g. MinIO/R2), `regionName`, and `profileName`
 * are the only construction knobs, and none of them is credential material;
 * there is deliberately no parameter this class could accept an access key
 * or secret through.
 *
 * Mappings onto the protocol's fingerprint tiers (ADR 0007 §9): `versionId`
 * is populated only when the bucket has versioning enabled (checked via
 * `GetBucketVersioning`, degrading to "not versioned" on ANY failure —
 * never propagated, since a listing this class can otherwise complete must
 * not fail over a narrower permission this one optional check needs);
 * `checksum` is never populated — a real `ListObjectsV2`/
 * `ListObjectVersions` response carries no cheap-at-listing-time checksum,
 * unlike GCS's `crc32c`/`md5Hash` (objectstore-gcs.ts); `etag` is the last
 * resort, quote-stripped the same way `objectstore-azure.ts`'s `cleanEtag`
 * documents. `objectTags` maps to S3's own object tagging API
 * (`GetObjectTagging`), degrading to `{}` on a tags-only permission gap
 * (`s3:GetObjectTagging` denied while `s3:GetObject` is allowed) exactly as
 * ADR 0007 §9 requires.
 *
 * Cloud SDK boundary: `@aws-sdk/client-s3` (and, only when `profileName` is
 * given, `@aws-sdk/credential-providers`) is referenced ONLY via a dynamic
 * `import()` inside `clientOrBuild`, never at this module's top level, so
 * the package need not be installed at all unless a caller actually
 * constructs a store without an injected `client`. Error classification is
 * STRUCTURAL (`name`/`$metadata.httpStatusCode` fields — the shape AWS SDK
 * v3's own modeled exceptions and a network-free fake test double both
 * surface), never `instanceof` a type from that optional package.
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
 * One listed S3 object, in the shape this store's own client seam reads —
 * the TypeScript analogue of one `ListObjectsV2`/`ListObjectVersions`
 * response entry, camelCased and flattened. A test injects a fake
 * implementing `S3ClientLike` directly (`S3ObjectStore`'s own `client`
 * escape hatch); the real (non-injected) construction path adapts an
 * actual `S3Client` instance onto it.
 */
export interface S3ObjectLike {
  readonly key: string;
  readonly size: number;
  readonly lastModified: Date | null;
  readonly etag: string | null;
  /** Only ever populated by `listObjectVersions` — a plain
   * `listObjectsV2` listing has no version id at all (ADR 0007 §9: the top
   * fingerprint tier is gated on the bucket having versioning enabled). */
  readonly versionId?: string | null;
}

/** The injectable client seam — `S3ObjectStore`'s `client` escape hatch,
 * mirroring the real `@aws-sdk/client-s3` `S3Client`'s call surface
 * closely enough for a network-free fake to stand in for it in tests.
 * Method names are the camelCased boto3/AWS-SDK operation names, the same
 * convention `GCSClientLike`/`AzureContainerClientLike` (objectstore-gcs.ts/
 * objectstore-azure.ts) already use for their own real SDKs' surfaces.
 * `listObjectVersions` yields only `IsLatest` entries (excluding delete
 * markers) — the same filter `S3ObjectStore._list_versions` (the Python
 * original) applies inline, pushed here onto the client seam so a fake
 * test double controls it directly. */
export interface S3ClientLike {
  bucketVersioningEnabled(bucket: string): Promise<boolean>;
  listObjectsV2(bucket: string, prefix: string): AsyncIterable<S3ObjectLike>;
  listObjectVersions(bucket: string, prefix: string): AsyncIterable<S3ObjectLike>;
  getObject(
    bucket: string,
    key: string,
    versionId: string | null,
  ): Promise<{ body: Uint8Array; contentType: string | null }>;
  getObjectTagging(bucket: string, key: string): Promise<Record<string, string>>;
}

// ADR 0007 §9's explicit carve-out, S3 dress — the same code set
// objectstore.py's `_classify_client_error` uses, keyed by AWS SDK v3's own
// exception `name` field (its analogue of botocore's `Error.Code`). A
// missing bucket (`NoSuchBucket`) sits beside the credential codes for the
// same reason the GCS/Azure backends' own `NoSuchBucket`/`ContainerNotFound`
// carve-outs do: a missing bucket is a configuration error, not one object
// vanishing mid-pass — `list()`'s contract promises never to raise
// `ObjectNotFoundError`.
const S3_PERMANENT_ERROR_NAMES: ReadonlySet<string> = new Set([
  "AccessDenied",
  "AllAccessDisabled",
  "InvalidAccessKeyId",
  "SignatureDoesNotMatch",
  "ExpiredToken",
  "TokenRefreshRequired",
  "InvalidToken",
  "AuthFailure",
  "UnauthorizedAccess",
  "AccountProblem",
  "InvalidSecurity",
  "NoSuchBucket",
]);

const S3_NOT_FOUND_ERROR_NAMES: ReadonlySet<string> = new Set(["NoSuchKey", "NoSuchVersion"]);

// The structural stand-in for the Python twin's
// `isinstance(error, NoCredentialsError)` — a credential-resolution
// failure has no HTTP status at all, so it must be recognized by name
// rather than by a status code. Extend this if the installed SDK surfaces
// a different name (`@aws-sdk/credential-providers`'s own errors vary by
// provider).
const S3_AUTH_ERROR_NAMES: ReadonlySet<string> = new Set(["CredentialsProviderError"]);

const S3_PERMANENT_STATUS: ReadonlySet<number> = new Set([401, 403]);

function errorField(error: unknown, key: string): unknown {
  if (typeof error !== "object" || error === null) {
    return undefined;
  }
  return (error as Record<string, unknown>)[key];
}

function statusOf(error: unknown): number | undefined {
  const metadata = errorField(error, "$metadata");
  if (typeof metadata !== "object" || metadata === null) {
    return undefined;
  }
  const status = (metadata as Record<string, unknown>)["httpStatusCode"];
  return typeof status === "number" ? status : undefined;
}

/**
 * One S3 failure → the protocol's exception vocabulary. As in the GCS/Azure
 * backends, `notFound` is what "does not exist" means at this call site: a
 * `PermanentStoreError` factory for a listing (a missing bucket is a
 * configuration error), an `ObjectNotFoundError` factory for a fetch (the
 * object vanished between listing and get — skip it this pass). Code-based
 * classification wins over a bare 404 status the same way
 * `objectstore.py`'s `_classify_client_error` documents: S3 answers a
 * missing bucket with HTTP 404 too, and a bare status code alone cannot
 * distinguish "this bucket doesn't exist" from "this object doesn't
 * exist."
 */
function classifyS3Error(error: unknown, notFound: () => Error): Error {
  const name = errorField(error, "name");
  if (typeof name === "string" && S3_AUTH_ERROR_NAMES.has(name)) {
    return new PermanentStoreError(errorMessage(error));
  }
  if (typeof name === "string" && S3_PERMANENT_ERROR_NAMES.has(name)) {
    return new PermanentStoreError(errorMessage(error));
  }
  if (typeof name === "string" && S3_NOT_FOUND_ERROR_NAMES.has(name)) {
    return notFound();
  }
  const status = statusOf(error);
  if (status === 404) {
    return notFound();
  }
  if (status !== undefined && S3_PERMANENT_STATUS.has(status)) {
    return new PermanentStoreError(errorMessage(error));
  }
  // Everything unnamed stays transient (connection resets, timeouts, any
  // other API-call status) — the same narrowed-only-by-carve-out posture
  // the GCS/Azure backends take.
  return new TransientStoreError(errorMessage(error));
}

/** S3 wraps `ETag` in literal double quotes in every listing/response API;
 * strip them so a caller comparing values never has to know that — the
 * same trim `objectstore-azure.ts`'s `cleanEtag` documents for Azure's own
 * quoted `ETag`. */
function cleanEtag(etag: string | null): string | null {
  if (etag === null) {
    return null;
  }
  const stripped = etag.replace(/^"+/, "").replace(/"+$/, "");
  return stripped || null;
}

function metaOf(obj: S3ObjectLike): ObjectMeta {
  return new ObjectMeta({
    key: obj.key,
    size: obj.size,
    lastModified: obj.lastModified,
    etag: cleanEtag(obj.etag),
    versionId: obj.versionId ?? null,
    // Never populated here — a real ListObjectsV2/ListObjectVersions
    // response carries no cheap-at-listing-time checksum (see this
    // module's own doc comment); `checksum` stays `null` so
    // `objectFingerprint` (objectstore.ts) falls through to
    // `size_last_modified`/`etag` exactly as it does for the Python
    // original's own `S3ObjectStore`.
    checksum: null,
  });
}

/** Minimal shape of the real `@aws-sdk/client-s3` `S3Client`/commands this
 * module's real (non-injected) construction path adapts onto
 * `S3ClientLike` — unverified against the live package (an optional peer
 * dependency not installed in this workspace) and exercised by no test
 * here; a caller who wants the real client behavior proven should inject
 * one they already trust via `client`. */
interface RealS3Object {
  Key?: string;
  Size?: number;
  LastModified?: Date;
  ETag?: string;
}
interface RealS3Version extends RealS3Object {
  VersionId?: string;
  IsLatest?: boolean;
}
interface RealS3ListObjectsV2Output {
  Contents?: RealS3Object[];
  IsTruncated?: boolean;
  NextContinuationToken?: string;
}
interface RealS3ListObjectVersionsOutput {
  Versions?: RealS3Version[];
  IsTruncated?: boolean;
  NextKeyMarker?: string;
  NextVersionIdMarker?: string;
}
interface RealS3GetObjectOutput {
  Body?: { transformToByteArray(): Promise<Uint8Array> };
  ContentType?: string;
}
interface RealS3GetObjectTaggingOutput {
  TagSet?: Array<{ Key?: string; Value?: string }>;
}
interface RealS3GetBucketVersioningOutput {
  Status?: string;
}
interface RealS3Client {
  send(command: unknown): Promise<unknown>;
}
type S3Module = {
  S3Client: new (config: { endpoint?: string; region?: string; credentials?: unknown }) => RealS3Client;
  ListObjectsV2Command: new (input: {
    Bucket: string;
    Prefix: string;
    ContinuationToken?: string;
  }) => unknown;
  ListObjectVersionsCommand: new (input: {
    Bucket: string;
    Prefix: string;
    KeyMarker?: string;
    VersionIdMarker?: string;
  }) => unknown;
  GetObjectCommand: new (input: { Bucket: string; Key: string; VersionId?: string }) => unknown;
  GetObjectTaggingCommand: new (input: { Bucket: string; Key: string }) => unknown;
  GetBucketVersioningCommand: new (input: { Bucket: string }) => unknown;
};

function adaptRealClient(real: RealS3Client, commands: S3Module): S3ClientLike {
  return {
    async bucketVersioningEnabled(bucket) {
      const response = (await real.send(
        new commands.GetBucketVersioningCommand({ Bucket: bucket }),
      )) as RealS3GetBucketVersioningOutput;
      return response.Status === "Enabled";
    },
    async *listObjectsV2(bucket, prefix) {
      let token: string | undefined;
      for (;;) {
        const response = (await real.send(
          new commands.ListObjectsV2Command({ Bucket: bucket, Prefix: prefix, ContinuationToken: token }),
        )) as RealS3ListObjectsV2Output;
        for (const obj of response.Contents ?? []) {
          yield {
            key: obj.Key ?? "",
            size: obj.Size ?? 0,
            lastModified: obj.LastModified ?? null,
            etag: obj.ETag ?? null,
          };
        }
        if (!response.IsTruncated || response.NextContinuationToken === undefined) {
          break;
        }
        token = response.NextContinuationToken;
      }
    },
    async *listObjectVersions(bucket, prefix) {
      let keyMarker: string | undefined;
      let versionIdMarker: string | undefined;
      for (;;) {
        const response = (await real.send(
          new commands.ListObjectVersionsCommand({
            Bucket: bucket,
            Prefix: prefix,
            KeyMarker: keyMarker,
            VersionIdMarker: versionIdMarker,
          }),
        )) as RealS3ListObjectVersionsOutput;
        for (const obj of response.Versions ?? []) {
          // `IsLatest` alone selects each key's current content version and
          // naturally excludes a key whose latest entry is a delete marker
          // (delete markers are returned separately, under `DeleteMarkers`,
          // never inside `Versions`) — mirrors the Python original's own
          // `_list_versions` filter exactly.
          if (!obj.IsLatest) {
            continue;
          }
          yield {
            key: obj.Key ?? "",
            size: obj.Size ?? 0,
            lastModified: obj.LastModified ?? null,
            etag: obj.ETag ?? null,
            versionId: obj.VersionId ?? null,
          };
        }
        // Also stop when a truncated response carries neither marker —
        // an S3-compatible store (MinIO, R2, ...) that omits both would
        // otherwise make the next iteration resend the identical request
        // forever, re-yielding the same objects. `listObjectsV2` above
        // already treats a missing NextContinuationToken the same way.
        if (
          !response.IsTruncated ||
          (response.NextKeyMarker === undefined && response.NextVersionIdMarker === undefined)
        ) {
          break;
        }
        keyMarker = response.NextKeyMarker;
        versionIdMarker = response.NextVersionIdMarker;
      }
    },
    async getObject(bucket, key, versionId) {
      const input: { Bucket: string; Key: string; VersionId?: string } = { Bucket: bucket, Key: key };
      if (versionId !== null) {
        input.VersionId = versionId;
      }
      const response = (await real.send(new commands.GetObjectCommand(input))) as RealS3GetObjectOutput;
      const body = response.Body ? await response.Body.transformToByteArray() : new Uint8Array();
      return { body, contentType: response.ContentType ?? null };
    },
    async getObjectTagging(bucket, key) {
      const response = (await real.send(
        new commands.GetObjectTaggingCommand({ Bucket: bucket, Key: key }),
      )) as RealS3GetObjectTaggingOutput;
      const tags: Record<string, string> = {};
      for (const tag of response.TagSet ?? []) {
        if (tag.Key !== undefined && tag.Value !== undefined) {
          tags[tag.Key] = tag.Value;
        }
      }
      return tags;
    },
  };
}

/**
 * Reads/lists one S3 (or S3-compatible) bucket via `@aws-sdk/client-s3`,
 * built from the standard AWS credential chain only — `endpointUrl` (for
 * an S3-compatible service, e.g. MinIO/R2), `regionName`, and
 * `profileName` are the only construction knobs, and none of them is
 * credential material itself; there is deliberately no parameter this
 * class could accept an access key or secret through.
 *
 * `client` is the same escape hatch the GCS/Azure backends document: a
 * caller who already has a configured client (custom retry/timeout config,
 * or — this package's own tests — a network-free fake) passes it here,
 * mutually exclusive with `endpointUrl`/`regionName`/`profileName`, which
 * exist only to build one.
 */
export class S3ObjectStore implements ObjectStore {
  private readonly bucketName: string;
  private readonly endpointUrl: string | null;
  private readonly regionName: string | null;
  private readonly profileName: string | null;
  private client: S3ClientLike | null;

  constructor(
    bucket: string,
    options?: {
      endpointUrl?: string;
      regionName?: string;
      profileName?: string;
      client?: S3ClientLike;
    },
  ) {
    const { endpointUrl, regionName, profileName, client } = options ?? {};
    if (client !== undefined) {
      if (endpointUrl !== undefined || regionName !== undefined || profileName !== undefined) {
        throw new Error("client is mutually exclusive with endpointUrl/regionName/profileName");
      }
      this.bucketName = bucket;
      this.client = client;
      this.endpointUrl = null;
      this.regionName = null;
      this.profileName = null;
      return;
    }
    // Resolved (and validated) eagerly, unlike the real client below —
    // this is plain string parsing, no I/O, so it can stay synchronous and
    // preserve the Python twin's construction-time throw for a
    // credential-embedding endpoint.
    if (endpointUrl !== undefined) {
      const parsed = new URL(endpointUrl);
      if (parsed.username || parsed.password) {
        throw new Error("endpointUrl must not embed credentials");
      }
    }
    this.bucketName = bucket;
    this.endpointUrl = endpointUrl ?? null;
    this.regionName = regionName ?? null;
    this.profileName = profileName ?? null;
    this.client = null;
  }

  get baseUri(): string {
    return `s3://${this.bucketName}`;
  }

  /**
   * Built on first use, not in the constructor: unlike the Python twin
   * (whose `boto3`/`botocore` imports are resolved eagerly at module load,
   * so `__init__` can raise `ImportError` synchronously), this port cannot
   * `await` a dynamic `import()` inside a synchronous constructor — BOTH
   * "the package isn't installed" and "the client can't be built" are
   * therefore deferred here to first use, one stage later than the Python
   * twin's construction-time package check. Constructing a store object
   * still touches neither the package registry nor the environment when a
   * `client` was injected.
   */
  private async clientOrBuild(): Promise<S3ClientLike> {
    if (this.client !== null) {
      return this.client;
    }
    const moduleSpecifier = "@aws-sdk/client-s3";
    let s3Module: S3Module;
    try {
      s3Module = (await import(moduleSpecifier)) as unknown as S3Module;
    } catch (error) {
      throw new Error(
        "S3ObjectStore requires @aws-sdk/client-s3 — install it as a dependency " +
          `(npm install @aws-sdk/client-s3): ${errorMessage(error)}`,
      );
    }
    try {
      let credentials: unknown;
      if (this.profileName !== null) {
        const credentialsModuleSpecifier = "@aws-sdk/credential-providers";
        const credentialsModule = (await import(credentialsModuleSpecifier)) as {
          fromIni: (options: { profile: string }) => unknown;
        };
        credentials = credentialsModule.fromIni({ profile: this.profileName });
      }
      const real = new s3Module.S3Client({
        endpoint: this.endpointUrl ?? undefined,
        region: this.regionName ?? undefined,
        credentials,
      });
      this.client = adaptRealClient(real, s3Module);
    } catch (error) {
      throw new PermanentStoreError(errorMessage(error));
    }
    return this.client;
  }

  /** A best-effort check, not itself part of ADR 0007 §9's two-way retry
   * split: ANY failure here (permission denial included) degrades to
   * "assume not versioned" rather than propagating, since a listing this
   * class can otherwise complete must not fail over a narrower permission
   * this one optional check needs — mirrors the Python original's own
   * `_versioning_enabled` exactly. */
  private async versioningEnabled(client: S3ClientLike): Promise<boolean> {
    try {
      return await client.bucketVersioningEnabled(this.bucketName);
    } catch {
      return false;
    }
  }

  async *list(prefix: string): AsyncGenerator<ObjectMeta> {
    const client = await this.clientOrBuild();
    const versioned = await this.versioningEnabled(client);
    try {
      const iterable = versioned
        ? client.listObjectVersions(this.bucketName, prefix)
        : client.listObjectsV2(this.bucketName, prefix);
      for await (const obj of iterable) {
        yield metaOf(obj);
      }
    } catch (error) {
      // A missing bucket here is a configuration error — permanent, like
      // `NoSuchBucket`; `list()`'s contract never raises
      // `ObjectNotFoundError`.
      throw classifyS3Error(error, () => new PermanentStoreError(errorMessage(error)));
    }
  }

  async get(key: string, options?: { versionId?: string }): Promise<FetchedObject> {
    const client = await this.clientOrBuild();
    try {
      const result = await client.getObject(this.bucketName, key, options?.versionId ?? null);
      return new FetchedObject({ body: result.body, contentType: result.contentType });
    } catch (error) {
      throw classifyS3Error(error, () => new ObjectNotFoundError(errorMessage(error)));
    }
  }

  /**
   * `key`'s object tags — S3's own tagging API, degrading to `{}` on a
   * tags-only permission gap or a vanished object exactly as ADR 0007 §9
   * requires (mirrors the GCS/Azure backends' own `objectTags`).
   */
  async objectTags(key: string): Promise<Record<string, string>> {
    try {
      const client = await this.clientOrBuild();
      return { ...(await client.getObjectTagging(this.bucketName, key)) };
    } catch (error) {
      if (error instanceof PermanentStoreError) {
        return {};
      }
      const classified = classifyS3Error(error, () => new ObjectNotFoundError(errorMessage(error)));
      if (classified instanceof TransientStoreError) {
        // Transient failures re-raise (retry next pass) exactly as the
        // GCS/Azure backends' own `objectTags` does; only the
        // permanent/not-found cases degrade to "no tags."
        throw classified;
      }
      return {};
    }
  }
}
