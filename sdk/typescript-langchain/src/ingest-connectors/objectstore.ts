/**
 * The object-storage boundary (ADR 0007 §9, issue #351): a thin store
 * abstraction several backends implement — `S3ObjectStore` (the real
 * thing, optional `@aws-sdk/client-s3` peer dependency), `GCSObjectStore`
 * (objectstore-gcs.ts) and `AzureBlobObjectStore` (objectstore-azure.ts,
 * both issue #414), and `FileObjectStore` (node:fs only, a `file://`
 * directory tree). The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.objectstore` module (issue #415).
 *
 * Deliberately NOT a reimplementation of `src/ship.rs`'s Rust
 * `object_store`-backed replication path: this package's own packaging
 * boundary (ADR 0007 §3/§4, Option C) keeps every connector client-side
 * and `src/` untouched. What IS reused verbatim is the *shape* of
 * `open_store`: parse a URL, dispatch on scheme, return a store scoped to
 * one bucket/directory plus the prefix the URL itself named — and the
 * same refusal posture (`file://` never silently `mkdir`s a missing
 * "bucket"; an unsupported scheme names the ones that are supported).
 *
 * Two exception kinds only (ADR 0007 §9's explicit two-way split):
 * `TransientStoreError` (network/timeout/5xx/throttling — safe to retry
 * on the next enumeration pass) and `PermanentStoreError`
 * (invalid/expired/revoked credential, or object-level permission
 * denial — terminal, reported once, never auto-retried).
 * `ObjectNotFoundError` is a third, narrower case (the object vanished
 * between listing and fetch) that is neither: the caller treats it as
 * "skip this object this pass," not a failure at all.
 */

/**
 * A network/timeout/5xx/throttling failure — the caller's next
 * enumeration pass retries from where this one left off.
 */
export class TransientStoreError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TransientStoreError";
  }
}

/**
 * Invalid/expired/revoked credentials, or object-level permission denial
 * (403/`AccessDenied`-shaped). Terminal immediately: retrying an auth
 * failure on every future enumeration pass would silently mask a real,
 * fixable problem behind an endless quiet retry loop (ADR 0007 §9).
 */
export class PermanentStoreError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PermanentStoreError";
  }
}

/**
 * The object named no longer exists — it vanished between this run's
 * listing and its fetch. Neither transient nor permanent: the caller
 * treats this as "skip this object this pass," not a failure.
 */
export class ObjectNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ObjectNotFoundError";
  }
}

/** `error.message` when `error` is an `Error`, else `String(error)` — the
 * TS stand-in for Python's `str(error)` (which never prefixes the
 * exception's own class name the way `String(error)`/`error.toString()`
 * would). Shared by the GCS/Azure backends' own error classification. */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * One listed object's identity and the fingerprint fields ADR 0007 §9
 * ranks: `versionId` (only when the bucket has versioning enabled),
 * `checksum` (only when a store cheaply returns one AT LISTING TIME — a
 * real S3 `ListObjectsV2`/`ListObjectVersions` response does not, so
 * `S3ObjectStore` never populates this; it exists for a store that can),
 * `size`/`lastModified`, and `etag` (last resort: many S3-compatible
 * stores compute a multipart upload's `ETag` in a way that is not a
 * reliable content hash).
 */
export class ObjectMeta {
  readonly key: string;
  readonly size: number;
  readonly lastModified: Date | null;
  readonly etag: string | null;
  readonly versionId: string | null;
  readonly checksum: string | null;

  constructor(fields: {
    key: string;
    size: number;
    lastModified: Date | null;
    etag?: string | null;
    versionId?: string | null;
    checksum?: string | null;
  }) {
    this.key = fields.key;
    this.size = fields.size;
    this.lastModified = fields.lastModified;
    this.etag = fields.etag ?? null;
    this.versionId = fields.versionId ?? null;
    this.checksum = fields.checksum ?? null;
  }
}

/**
 * One object's full raw bytes plus whatever content-type the store itself
 * reports for it (only available once the object is actually fetched — a
 * listing alone never carries it). Used as dispatch's fallback signal
 * (ADR 0007 §9's "extension first, MIME as fallback") when a key's own
 * extension does not already resolve to a supported format.
 */
export class FetchedObject {
  readonly body: Uint8Array;
  readonly contentType: string | null;

  constructor(fields: { body: Uint8Array; contentType?: string | null }) {
    this.body = fields.body;
    this.contentType = fields.contentType ?? null;
  }
}

export type FingerprintTier = "version_id" | "checksum" | "size_last_modified" | "etag";

/**
 * ADR 0007 §9's checkpoint-fingerprint priority, applied to one listed
 * object: `versionId` > `checksum` > `(size, lastModified)` > `etag`
 * alone > `null` (no usable fingerprint at all — the object is always
 * re-fetched). Never `etag` as the FIRST choice, per #217's explicit
 * requirement.
 *
 * The `size_last_modified` tier's string uses `Date#toISOString()`
 * (`"...Z"`) rather than Python's timezone-aware `datetime.isoformat()`
 * (`"...+00:00"`) — both are stable, unambiguous serializations of the
 * same instant; nothing in this SDK compares a fingerprint string minted
 * by one language's runtime against the other's.
 */
export function objectFingerprint(meta: ObjectMeta): [FingerprintTier, string] | null {
  if (meta.versionId) {
    return ["version_id", meta.versionId];
  }
  if (meta.checksum) {
    return ["checksum", meta.checksum];
  }
  if (meta.lastModified !== null) {
    return ["size_last_modified", `${meta.size}:${meta.lastModified.toISOString()}`];
  }
  if (meta.etag) {
    return ["etag", meta.etag];
  }
  return null;
}

/**
 * The structural interface `S3ObjectStore`/`GCSObjectStore`/
 * `AzureBlobObjectStore`/`FileObjectStore` all implement — a TypeScript
 * `interface`, not a base class, the same extension-point convention
 * `Connector` (protocol.ts) and `CheckpointStore` (checkpoints.ts)
 * already use: structural typing, no forced inheritance, no runtime
 * `instanceof` check.
 */
export interface ObjectStore {
  /**
   * The store's own root, e.g. `s3://my-bucket` or `file:///abs/dir` —
   * the prefix every source id this store's objects mint is built from
   * (ADR 0007 §6.1: `s3://bucket/key`).
   */
  readonly baseUri: string;

  /**
   * Every object whose key starts with `prefix`, oldest-listing-API
   * pagination handled internally. Rejects with `TransientStoreError`/
   * `PermanentStoreError` on failure; never rejects with
   * `ObjectNotFoundError` (a prefix that matches nothing is an empty
   * iterable, not a missing-object error).
   */
  list(prefix: string): AsyncIterable<ObjectMeta>;

  /**
   * The full raw bytes of one object, plus its content-type if the store
   * knows one. Rejects with `ObjectNotFoundError` if `key` (or
   * `versionId`) no longer exists, `TransientStoreError`/
   * `PermanentStoreError` on any other failure.
   */
  get(key: string, options?: { versionId?: string }): Promise<FetchedObject>;

  /**
   * `key`'s object tags. Degrades to `{}` when tags specifically are
   * unreadable (e.g. `s3:GetObjectTagging` denied while the object itself
   * is readable) — never fails the object over a tags-only permission gap
   * (ADR 0007 §9).
   */
  objectTags(key: string): Promise<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// file:// — node:fs only. The test/air-gapped backend (issue #351's own
// "minio/localstack を使わず file:// バケットでテスト" requirement), the same
// role `src/ship.rs`'s own `file://` scheme plays for replication.
// ---------------------------------------------------------------------------

// Node's stdlib carries no MIME database analogous to Python's `mimetypes`
// module — this is a small, deliberately narrow table covering the
// formats this SDK's own connectors care about, not a general-purpose MIME
// guesser. `FileObjectStore.get` is the only reader.
const EXTENSION_CONTENT_TYPES: ReadonlyMap<string, string> = new Map([
  [".pdf", "application/pdf"],
  [".txt", "text/plain"],
  [".md", "text/markdown"],
  [".html", "text/html"],
  [".htm", "text/html"],
  [".csv", "text/csv"],
  [".json", "application/json"],
  [".xml", "application/xml"],
  [".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
  [".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"],
  [".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
  [".zip", "application/zip"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
]);

function guessContentType(fileName: string): string | null {
  const dot = fileName.lastIndexOf(".");
  if (dot < 0) {
    return null;
  }
  return EXTENSION_CONTENT_TYPES.get(fileName.slice(dot).toLowerCase()) ?? null;
}

async function collectFiles(
  fsp: typeof import("node:fs/promises"),
  pathMod: typeof import("node:path"),
  dir: string,
): Promise<string[]> {
  const entries = await fsp.readdir(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const full = pathMod.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectFiles(fsp, pathMod, full)));
    } else if (entry.isFile()) {
      files.push(full);
    }
  }
  return files;
}

/**
 * A directory tree addressed the way an S3 bucket is: `list(prefix)`
 * matches on the relative POSIX path as a plain string prefix (not a
 * directory boundary — `prefix="2026/q1"` matches `2026/q1-report.pdf`
 * too, mirroring S3's own prefix semantics), never mtime/size beyond what
 * `stat` gives for free. `etag`/`versionId`/`checksum` are always `null`
 * — a local directory has none of those — so `objectFingerprint` degrades
 * to the `(size, lastModified)` tier here, exactly the tier real
 * (non-versioned) S3 buckets use too.
 *
 * Construction is a static async factory (`FileObjectStore.open`) rather
 * than the constructor itself — the Python twin's `__init__` validates the
 * bucket directory synchronously (`Path.is_dir()`, a blocking stdlib
 * call), but this port's I/O is async-only (`await import("node:fs/...")`
 * inside methods), and a constructor cannot `await`. The bucket must still
 * exist before anything reads from it — hold `file://` to the same
 * contract `src/ship.rs:196-204` already does for replication, instead of
 * silently mkdir-ing a typo into a fresh empty "bucket".
 */
export class FileObjectStore implements ObjectStore {
  private readonly directory: string;

  private constructor(directory: string) {
    this.directory = directory;
  }

  static async open(directory: string): Promise<FileObjectStore> {
    const fsp = await import("node:fs/promises");
    let isDirectory = false;
    try {
      isDirectory = (await fsp.stat(directory)).isDirectory();
    } catch {
      isDirectory = false;
    }
    if (!isDirectory) {
      throw new Error(`${directory}: object-storage bucket directory does not exist`);
    }
    // Mirrors Python's `Path.resolve()`, which resolves symlinks
    // (`realpath`) rather than merely absolutizing the path the way
    // `path.resolve` alone would.
    const resolved = await fsp.realpath(directory);
    return new FileObjectStore(resolved);
  }

  get baseUri(): string {
    return `file://${this.directory.replaceAll("\\", "/")}`;
  }

  async *list(prefix: string): AsyncGenerator<ObjectMeta> {
    const fsp = await import("node:fs/promises");
    const pathMod = await import("node:path");
    const files = await collectFiles(fsp, pathMod, this.directory);
    const keyed = files
      .map((full) => ({
        full,
        key: pathMod.relative(this.directory, full).split(pathMod.sep).join("/"),
      }))
      .filter(({ key }) => key.startsWith(prefix))
      .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
    for (const { full, key } of keyed) {
      const stat = await fsp.stat(full);
      yield new ObjectMeta({ key, size: stat.size, lastModified: new Date(stat.mtimeMs) });
    }
  }

  async get(key: string, options?: { versionId?: string }): Promise<FetchedObject> {
    void options; // file:// has no versions; a caller never passes one here
    const resolved = await this.resolveKey(key);
    const fsp = await import("node:fs/promises");
    let body: Buffer;
    try {
      body = await fsp.readFile(resolved);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        throw new ObjectNotFoundError(errorMessage(error));
      }
      throw new TransientStoreError(errorMessage(error));
    }
    const pathMod = await import("node:path");
    return new FetchedObject({
      body: new Uint8Array(body.buffer, body.byteOffset, body.byteLength),
      contentType: guessContentType(pathMod.basename(resolved)),
    });
  }

  async objectTags(key: string): Promise<Record<string, string>> {
    void key; // a local directory carries no object tags at all
    return {};
  }

  private async resolveKey(key: string): Promise<string> {
    const pathMod = await import("node:path");
    const resolved = pathMod.resolve(this.directory, key);
    const rel = pathMod.relative(this.directory, resolved);
    if (rel !== "" && (rel.startsWith(`..${pathMod.sep}`) || rel === ".." || pathMod.isAbsolute(rel))) {
      throw new Error(`${JSON.stringify(key)} escapes the bucket directory`);
    }
    return resolved;
  }
}

// ---------------------------------------------------------------------------
// open_object_store — scheme dispatch, mirroring src/ship.rs's open_store
// ---------------------------------------------------------------------------

function tolerantDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function stripLeadingSlashes(value: string): string {
  return value.replace(/^\/+/, "");
}

/**
 * Opens the store `url` names, mirroring `src/ship.rs:176`'s own
 * `open_store` in shape AND scheme set (`s3://`, `gs://`, `az://`,
 * `file://`): parse the URL, dispatch on scheme, resolve
 * `[store, prefix]` — a caller lists `store.list(prefix)` to enumerate
 * exactly what the URL named. For the cloud schemes the prefix is the
 * URL's own path (the bucket/container is the host); for `file://` the
 * whole path IS the bucket directory, so the prefix is always `""`.
 * `endpointUrl`/`regionName`/`profileName` apply to `s3://` only (an
 * S3-compatible endpoint, or a non-default region/profile) and are
 * rejected everywhere else — `gs://`/`az://` read their standard
 * credential chains, and a caller needing an emulator endpoint or a
 * preconfigured client constructs `GCSObjectStore`/`AzureBlobObjectStore`
 * directly. The cloud backends' own modules are dynamic-imported here
 * (never at this module's top level), so none of their optional peer
 * dependencies has to be installed to open a URL of a different scheme.
 *
 * Unlike the Python twin — where `S3ObjectStore` lives in this same
 * module and is imported eagerly, while `GCSObjectStore`/
 * `AzureBlobObjectStore` are imported lazily only inside this function —
 * this port dynamic-imports all three cloud backends uniformly: every
 * cloud SDK is an equally optional peer dependency here (`@aws-sdk/
 * client-s3`/`@google-cloud/storage`/`@azure/storage-blob`), so there is
 * no reason for `s3://` to behave differently from `gs://`/`az://`. The
 * `s3://` backend (`objectstore-s3.ts`, `S3ObjectStore`) is ported under
 * this same issue by a separate module owner and is out of this file's
 * scope; its dynamic import path is named here by the established
 * `objectstore-<cloud>.ts` convention so dispatch is forward-compatible
 * the moment that module lands.
 */
export async function openObjectStore(
  url: string,
  options?: { endpointUrl?: string; regionName?: string; profileName?: string },
): Promise<[ObjectStore, string]> {
  const parsed = new URL(url);
  const scheme = parsed.protocol.slice(0, -1).toLowerCase();
  const { endpointUrl, regionName, profileName } = options ?? {};

  if (scheme === "s3") {
    const bucket = parsed.host;
    if (!bucket) {
      throw new Error(`${url}: s3:// URL must name a bucket, e.g. s3://my-bucket/prefix`);
    }
    const prefix = stripLeadingSlashes(tolerantDecode(parsed.pathname));
    const s3ModuleSpecifier = "./objectstore-s3.js";
    const s3Module = (await import(s3ModuleSpecifier)) as {
      S3ObjectStore: new (
        bucket: string,
        options?: { endpointUrl?: string; regionName?: string; profileName?: string },
      ) => ObjectStore;
    };
    const store: ObjectStore = new s3Module.S3ObjectStore(bucket, {
      endpointUrl,
      regionName,
      profileName,
    });
    return [store, prefix];
  }
  if (endpointUrl !== undefined || regionName !== undefined || profileName !== undefined) {
    throw new Error(
      `${scheme}:// does not accept endpointUrl/regionName/profileName — construct the ` +
        "store class directly for a custom endpoint or client",
    );
  }
  if (scheme === "gs") {
    const bucket = parsed.host;
    if (!bucket) {
      throw new Error(`${url}: gs:// URL must name a bucket, e.g. gs://my-bucket/prefix`);
    }
    const { GCSObjectStore } = await import("./objectstore-gcs.js");
    const prefix = stripLeadingSlashes(tolerantDecode(parsed.pathname));
    return [new GCSObjectStore(bucket), prefix];
  }
  if (scheme === "az") {
    const container = parsed.host;
    if (!container) {
      throw new Error(`${url}: az:// URL must name a container, e.g. az://my-container/prefix`);
    }
    const { AzureBlobObjectStore } = await import("./objectstore-azure.js");
    const prefix = stripLeadingSlashes(tolerantDecode(parsed.pathname));
    return [new AzureBlobObjectStore(container), prefix];
  }
  if (scheme === "file") {
    const directory = tolerantDecode(parsed.pathname);
    return [await FileObjectStore.open(directory), ""];
  }
  throw new Error(
    `${url}: unsupported object-storage scheme — use s3://, gs://, az://, or file://`,
  );
}
