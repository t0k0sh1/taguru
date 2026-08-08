/**
 * The S3 (object-storage) connector (ADR 0007 §9, issue #351): lists a
 * bucket/prefix, dispatches each object to whichever installed format
 * connector (#347-#350, #352) its extension or content-type names, and
 * syncs the result into a `TaguruIngester` (ingest.ts) — with a two-layer
 * checkpoint (skip the fetch when listing metadata is unchanged; skip the
 * model call/import when the fetched-and-parsed content is unchanged), a
 * deletion policy that never retracts by default, and a `dryRun` that
 * touches nothing. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.s3` module (issue #415).
 *
 * Two independent pieces, mirroring the Python original exactly:
 *
 * - `S3Connector` — dispatch. Given one listed object, fetches its raw
 *   bytes, hands them (via a suffix-named temporary file — the existing
 *   format connectors' own contract is "read a local path," never changed
 *   here) to whichever of the installed connectors `supports` it, then
 *   re-stamps the result's identity (`source`, `metadata.originUri`/
 *   `displayName`, each diagnostic's `source`) onto the S3 source id — the
 *   delegate's OWN `fingerprintInputs` (`parser`/`parserVersion`/the raw
 *   content hash/its own effective options) is kept untouched, exactly as
 *   ADR 0007 §5's own JSON example does.
 * - `syncObjectStorage` — orchestration. Enumerates a store/prefix,
 *   applies both checkpoint layers, calls `ingestConnectorDocument`
 *   (bridge.ts) for anything that needs it, detects deletions against a
 *   persisted prefix inventory, and returns a `RunReport`
 *   (observability.ts) (`S3SyncReport` here is a deprecated alias of it —
 *   ADR 0007 §11.1).
 *
 * Deliberately NOT a reimplementation of `src/ship.rs`'s Rust
 * `object_store`-backed path — see `objectstore.ts`'s own module doc
 * comment for why the packaging boundary (ADR 0007 §3/§4) puts this here
 * instead.
 *
 * Deliberate deviations from the Python original, all forced by this SDK's
 * own, already-established shape rather than chosen here:
 *
 * - **`syncObjectStorage`'s `deletionPolicy: "retract"/"mirror"` never
 *   raises for a "missing sync client."** Python's `TaguruIngester.client`
 *   is `Taguru | None` — an async-only ingester built with only
 *   `async_client` has none, so `sync_object_storage` raises a
 *   `ValueError` rather than guessing which client to retract through.
 *   This SDK's `TaguruIngester` (ingest.ts) has no such split: its
 *   constructor always builds or receives exactly one `Taguru` client
 *   (`this.client = fields.client ?? new Taguru(...)`), so there is no
 *   "async-only ingester" state to guard against — `clientOf` below
 *   always resolves to a real client. See `clientOf`'s own doc comment.
 * - **Diagnostic/error message text uses `JSON.stringify` for quoting**,
 *   not Python's `repr()` — the same convention `references.ts` already
 *   established (`` JSON.stringify(suffix) `` rather than `%r`). This
 *   means the `s3_sync` golden fixture's wording differs from the Python
 *   suite's own `s3_sync` fixture text, even though both pin the same
 *   scenario — the two are independent, language-native goldens, not a
 *   byte-shared file.
 * - **A staged temp file uses `crypto.randomUUID()` + `os.tmpdir()`**
 *   rather than Python's `tempfile.mkstemp` — the same pattern
 *   `checkpoints.ts`'s `writeAtomic` already uses for its own staged file.
 */

import type { Taguru } from "taguru";

import type { CheckpointStore } from "../checkpoints.js";
import type { ShouldStop, TaguruIngester } from "../ingest.js";
import { ingestConnectorDocument, normalizeStop } from "./bridge.js";
import { ConnectorCheckpoint } from "./checkpoint.js";
import {
  byteLen,
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  type DiagnosticCode,
  FingerprintInputs,
  MAX_TAG_BYTES,
  MAX_TAGS_PER_SOURCE,
  optionsDigest,
} from "./document.js";
import {
  FetchedObject,
  type FingerprintTier,
  ObjectMeta,
  ObjectNotFoundError,
  type ObjectStore,
  objectFingerprint,
  PermanentStoreError,
  TransientStoreError,
} from "./objectstore.js";
import {
  type Phase,
  RunRecorder,
  RunReport,
  S3SyncReport,
  SourceEvent,
  type SourceEventSinkTarget,
} from "./observability.js";
import type { Connector } from "./protocol.js";
import { defaultConnectors } from "./references.js";
import { checkSourceId, SourceIdRegistry } from "./sources.js";

// Re-exported: issue #351/#352 published `Phase`/`SourceEvent` from this
// submodule; issue #353 moved their canonical definition to
// observability.ts but kept both importable from here too (`S3SyncReport`
// already was) — mirrors the Python original's own re-export comment.
export { type Phase, RunReport, S3SyncReport, SourceEvent };

export const PARSER_NAME = "taguru-s3-connector";
export const PARSER_VERSION = "1.0.0";

const DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024;

// SHA-256 of the empty byte string — a fixed, well-known constant (unlike
// the Python original's `hashlib.sha256(b"").hexdigest()`, computed at
// import time), needed because `crypto.subtle.digest` is async and this
// value must be usable as a plain default parameter.
const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

// ADR 0007 §9: "extension first, MIME as fallback" — only consulted when a
// key's own extension does not already resolve to an installed connector.
const CONTENT_TYPE_SUFFIXES: ReadonlyMap<string, string> = new Map([
  ["application/pdf", ".pdf"],
  ["text/html", ".html"],
  ["application/xhtml+xml", ".xhtml"],
  ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"],
  ["application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"],
  ["text/markdown", ".md"],
  ["text/plain", ".txt"],
]);

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function sha256HexBytes(bytes: Uint8Array): Promise<string> {
  // Copied into a fresh, plain-ArrayBuffer-backed Uint8Array first: `bytes`
  // may be a view over a `SharedArrayBuffer`-compatible `ArrayBufferLike`
  // (e.g. a `Buffer`-derived view), which `crypto.subtle.digest`'s
  // `BufferSource` type does not accept directly.
  const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(bytes));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function keyBasename(key: string): string {
  const slashIndex = key.lastIndexOf("/");
  return slashIndex === -1 ? key : key.slice(slashIndex + 1);
}

function suffixFromKey(key: string): string {
  const name = keyBasename(key);
  const dotIndex = name.lastIndexOf(".");
  if (dotIndex <= 0) {
    return "";
  }
  return name.slice(dotIndex).toLowerCase();
}

function suffixFromContentType(contentType: string | null): string | null {
  if (!contentType) {
    return null;
  }
  const base = (contentType.split(";", 1)[0] ?? "").trim().toLowerCase();
  return CONTENT_TYPE_SUFFIXES.get(base) ?? null;
}

/**
 * `supports()` on every connector this package ships checks only the
 * reference's own extension (`HtmlConnector` also accepts a URL scheme,
 * irrelevant to a synthetic reference with none) — a fabricated
 * `"object<suffix>"` name is enough to ask "would you read this," without
 * needing each connector's private suffix set.
 */
function connectorForSuffix(connectors: readonly Connector[], suffix: string): Connector | null {
  const probe = `object${suffix}`;
  for (const connector of connectors) {
    if (connector.supports(probe)) {
      return connector;
    }
  }
  return null;
}

/**
 * A document that never reached a delegate connector at all
 * (`source_id_too_long`/`content_too_large`/`unsupported_format`) — its
 * `fingerprintInputs.parser` is `S3Connector`'s own identity, since no
 * format connector was ever chosen to attribute it to.
 */
function failureDocument(
  source: string,
  displayName: string,
  code: DiagnosticCode,
  message: string,
  rawContentSha256: string = EMPTY_SHA256,
  contentType: string | null = null,
): ConnectorDocument {
  return new ConnectorDocument({
    source,
    text: "",
    metadata: new ConnectorMetadata({ originUri: source, displayName, contentType }),
    fingerprintInputs: new FingerprintInputs({
      rawContentSha256,
      parser: PARSER_NAME,
      parserVersion: PARSER_VERSION,
      parseOptionsDigest: "",
    }),
    diagnostics: [new Diagnostic({ code, message, source })],
  });
}

/**
 * Replaces only the S3-identity-shaped fields — never `fingerprintInputs`
 * (the delegate's own parser/version/raw-hash/options stay exactly as it
 * produced them, ADR 0007 §5's own example).
 */
function reidentify(
  document: ConnectorDocument,
  options: { source: string; displayName: string; contentType: string | null },
): ConnectorDocument {
  const metadata = new ConnectorMetadata({
    originUri: options.source,
    displayName: options.displayName,
    title: document.metadata.title,
    canonicalUrl: document.metadata.canonicalUrl,
    tags: document.metadata.tags,
    contentType: document.metadata.contentType ?? options.contentType,
  });
  const diagnostics = document.diagnostics.map(
    (d) => new Diagnostic({ code: d.code, message: d.message, source: options.source }),
  );
  return new ConnectorDocument({
    source: options.source,
    text: document.text,
    locators: document.locators,
    sections: document.sections,
    metadata,
    fingerprintInputs: document.fingerprintInputs,
    diagnostics,
  });
}

function withTags(document: ConnectorDocument, tags: readonly string[]): ConnectorDocument {
  const metadata = new ConnectorMetadata({
    originUri: document.metadata.originUri,
    displayName: document.metadata.displayName,
    title: document.metadata.title,
    canonicalUrl: document.metadata.canonicalUrl,
    tags,
    contentType: document.metadata.contentType,
  });
  return new ConnectorDocument({
    source: document.source,
    text: document.text,
    locators: document.locators,
    sections: document.sections,
    metadata,
    fingerprintInputs: document.fingerprintInputs,
    diagnostics: document.diagnostics,
  });
}

/**
 * Object tags → `"k=v"` (bare `"k"` for an empty value) strings, sorted
 * for determinism, capped at `MAX_TAGS_PER_SOURCE`/`MAX_TAG_BYTES` —
 * dropped and counted, never truncated (ADR 0007 §9, the same posture
 * `questions_dropped`/`sections_dropped` already take). Any failure
 * reading tags at all (including the `AccessDenied`-only-on-tagging
 * degrade `ObjectStore.objectTags` itself already applies) yields no tags
 * and no drops — never a failed object over a tags-only gap.
 */
async function mappedTags(store: ObjectStore, key: string): Promise<[readonly string[], number]> {
  let rawTags: Record<string, string>;
  try {
    rawTags = await store.objectTags(key);
  } catch (error) {
    if (
      error instanceof TransientStoreError ||
      error instanceof PermanentStoreError ||
      error instanceof ObjectNotFoundError
    ) {
      return [[], 0];
    }
    throw error;
  }
  const formatted = Object.entries(rawTags)
    .map(([name, value]) => (value ? `${name}=${value}` : name))
    .sort();
  const kept: string[] = [];
  let dropped = 0;
  for (const tag of formatted) {
    if (kept.length >= MAX_TAGS_PER_SOURCE || byteLen(tag) > MAX_TAG_BYTES) {
      dropped += 1;
      continue;
    }
    kept.push(tag);
  }
  return [kept, dropped];
}

/** `S3Connector.readObjectResult`'s own return shape — the document plus
 * how many of the object's tags this pass dropped (surfaced separately
 * from the document itself so `syncObjectStorage` can fold it into the
 * run's `tagsDropped` total without the document carrying a field that
 * belongs to the run, not the document). */
export class ReadResult {
  readonly document: ConnectorDocument;
  readonly tagsDropped: number;

  constructor(fields: { document: ConnectorDocument; tagsDropped?: number }) {
    this.document = fields.document;
    this.tagsDropped = fields.tagsDropped ?? 0;
  }
}

/**
 * Dispatches one object-storage object (S3, or a `file://`-backed
 * test/air-gapped bucket) to whichever installed format connector handles
 * it.
 *
 * `connectors` (default: every connector this package installed —
 * `defaultConnectors()`, references.ts, shared with `syncReferences`) is
 * tried in order via each one's own `supports()`; a caller may pass a
 * customized subset/ordering. `maxFileBytes` (default 64 MiB) caps the raw
 * object size, checked against the LISTING's own size — before any fetch,
 * mirroring every other connector's own pre-fetch size cap.
 */
export class S3Connector implements Connector {
  private readonly store: ObjectStore;
  private readonly connectorsList: readonly Connector[];
  private readonly maxFileBytes: number;

  constructor(store: ObjectStore, options?: { connectors?: readonly Connector[]; maxFileBytes?: number }) {
    this.store = store;
    this.connectorsList = options?.connectors ?? defaultConnectors();
    this.maxFileBytes = options?.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
  }

  get parser(): string {
    return PARSER_NAME;
  }

  get parserVersion(): string {
    return PARSER_VERSION;
  }

  async parseOptionsDigest(): Promise<string> {
    return optionsDigest({
      max_file_bytes: this.maxFileBytes,
      connectors: this.connectorsList.map((c) => `${c.parser}:${c.parserVersion}`),
    });
  }

  private keyFromReference(reference: string): string | null {
    const prefix = `${this.store.baseUri}/`;
    if (!reference.startsWith(prefix)) {
      return null;
    }
    return reference.slice(prefix.length);
  }

  supports(reference: string): boolean {
    const key = this.keyFromReference(reference);
    if (key === null) {
      return false;
    }
    const suffix = suffixFromKey(key);
    return suffix !== "" && connectorForSuffix(this.connectorsList, suffix) !== null;
  }

  /**
   * Protocol-conformant single-string entry point: fetches the object's
   * CURRENT (latest) version and dispatches it. The richer,
   * version-pinned, tag-aware, drop-counting path
   * (`readObject`/`readObjectResult`) is what `syncObjectStorage` actually
   * drives; this method exists so `S3Connector` is itself usable anywhere
   * a plain `Connector` (protocol.ts) is expected.
   */
  async read(reference: string): Promise<ConnectorDocument> {
    const key = this.keyFromReference(reference);
    if (key === null) {
      throw new Error(
        `${JSON.stringify(reference)} does not belong to this store (${this.store.baseUri}/...)`,
      );
    }
    return this.readObject(new ObjectMeta({ key, size: -1, lastModified: null }));
  }

  /** As `read`, but takes the full listing `ObjectMeta` (so a versioned
   * bucket fetches the exact listed version, and the listing's own size is
   * checked before any fetch). */
  async readObject(meta: ObjectMeta): Promise<ConnectorDocument> {
    return (await this.readObjectResult(meta)).document;
  }

  async readObjectResult(meta: ObjectMeta): Promise<ReadResult> {
    const source = `${this.store.baseUri}/${meta.key}`;
    const displayName = keyBasename(meta.key);

    const sourceDiagnostic = checkSourceId(source);
    if (sourceDiagnostic !== null) {
      return new ReadResult({
        document: failureDocument(source, displayName, sourceDiagnostic.code, sourceDiagnostic.message),
      });
    }

    if (meta.size >= 0 && meta.size > this.maxFileBytes) {
      return new ReadResult({
        document: failureDocument(
          source,
          displayName,
          "content_too_large",
          `${meta.size} bytes exceeds the ${this.maxFileBytes}-byte object cap`,
        ),
      });
    }

    // The listing's own size is authoritative when it exists (the
    // pre-fetch check above) — but `read()` (the plain `Connector`
    // entry point) synthesizes `size=-1` (no listing at all), which skips
    // it entirely, and content-type fallback (below) always fetches
    // before a connector is even chosen. Re-checking the FETCHED bytes
    // here, at every call site right after a fetch, is what makes the cap
    // hold regardless of which path reached it.
    const oversized = (candidate: FetchedObject): ReadResult | null => {
      if (candidate.body.length <= this.maxFileBytes) {
        return null;
      }
      return new ReadResult({
        document: failureDocument(
          source,
          displayName,
          "content_too_large",
          `${candidate.body.length} bytes exceeds the ${this.maxFileBytes}-byte object cap`,
          EMPTY_SHA256,
          candidate.contentType,
        ),
      });
    };

    let suffix = suffixFromKey(meta.key);
    let connector = suffix ? connectorForSuffix(this.connectorsList, suffix) : null;
    let fetched: FetchedObject | null = null;

    if (connector === null) {
      // Extension alone did not resolve to an installed connector — fetch
      // once and fall back to content-type (ADR 0007 §9: "extension
      // first, MIME as fallback"). A key whose extension names a KNOWN but
      // not-installed format still gets one content-type check here
      // rather than assuming the extension is authoritative.
      fetched = await this.store.get(meta.key, { versionId: meta.versionId ?? undefined });
      const tooLarge = oversized(fetched);
      if (tooLarge !== null) {
        return tooLarge;
      }
      const contentSuffix = suffixFromContentType(fetched.contentType);
      if (contentSuffix !== null) {
        const candidate = connectorForSuffix(this.connectorsList, contentSuffix);
        if (candidate !== null) {
          connector = candidate;
          suffix = contentSuffix;
        }
      }
    }

    if (connector === null) {
      const rawSha = fetched !== null ? await sha256HexBytes(fetched.body) : EMPTY_SHA256;
      const extension = suffixFromKey(meta.key) || "(none)";
      let message = `no connector for extension ${JSON.stringify(extension)}`;
      if (fetched !== null) {
        message += ` or content-type ${JSON.stringify(fetched.contentType)}`;
      }
      return new ReadResult({
        document: failureDocument(
          source,
          displayName,
          "unsupported_format",
          message,
          rawSha,
          fetched !== null ? fetched.contentType : null,
        ),
      });
    }

    if (fetched === null) {
      fetched = await this.store.get(meta.key, { versionId: meta.versionId ?? undefined });
      const tooLarge = oversized(fetched);
      if (tooLarge !== null) {
        return tooLarge;
      }
    }

    const [tags, tagsDropped] = await mappedTags(this.store, meta.key);
    const document = await this.delegate(source, displayName, connector, suffix, fetched, tags);
    return new ReadResult({ document, tagsDropped });
  }

  private async delegate(
    source: string,
    displayName: string,
    connector: Connector,
    suffix: string,
    fetched: FetchedObject,
    tags: readonly string[],
  ): Promise<ConnectorDocument> {
    const fsp = await import("node:fs/promises");
    const os = await import("node:os");
    const pathMod = await import("node:path");
    const staged = pathMod.join(os.tmpdir(), `taguru-s3-${crypto.randomUUID()}${suffix}`);
    try {
      await fsp.writeFile(staged, fetched.body);
      let document = await connector.read(staged);
      document = reidentify(document, { source, displayName, contentType: fetched.contentType });
      if (tags.length > 0) {
        document = withTags(document, tags);
      }
      return document;
    } finally {
      await fsp.unlink(staged).catch(() => {});
    }
  }
}

const S3_OBJECT_CHECKPOINT_VERSION = 1;

const FINGERPRINT_TIERS: ReadonlySet<string> = new Set([
  "version_id",
  "checksum",
  "size_last_modified",
  "etag",
]);

function fingerprintEquals(
  a: readonly [FingerprintTier, string] | null,
  b: readonly [FingerprintTier, string] | null,
): boolean {
  if (a === null || b === null) {
    return a === b;
  }
  return a[0] === b[0] && a[1] === b[1];
}

/**
 * ADR 0007 §9's cheap gate: "does this object's LISTING metadata match
 * what we saw last time" — answerable with no fetch at all. Composes with
 * (never replaces) `ConnectorCheckpoint` (checkpoint.ts, ADR 0007 §6.3):
 * this layer only ever prevents an unnecessary FETCH; `ConnectorCheckpoint`
 * still separately governs whether a fetched-and-parsed object's model
 * call/import can be skipped. Degrades exactly like every other checkpoint
 * in this SDK: a missing, corrupt, or store-failure entry means "treat as
 * unseen," never a false hit.
 */
export class S3ObjectCheckpoint {
  private readonly store: CheckpointStore;
  private readonly namespace: string;

  constructor(store: CheckpointStore, options?: { namespace?: string }) {
    this.store = store;
    this.namespace = options?.namespace ?? "s3-object";
  }

  private key(source: string): string {
    return `${this.namespace}:${source}`;
  }

  async load(source: string): Promise<[FingerprintTier, string] | null> {
    let data: Uint8Array | null;
    try {
      data = await this.store.load(this.key(source));
    } catch {
      return null; // Any store failure degrades to "uncached".
    }
    if (data === null) {
      return null;
    }
    let raw: unknown;
    try {
      raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(data));
    } catch {
      return null;
    }
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      return null;
    }
    const record = raw as Record<string, unknown>;
    if (record["version"] !== S3_OBJECT_CHECKPOINT_VERSION || record["source"] !== source) {
      return null;
    }
    const tier = record["tier"];
    const value = record["value"];
    if (typeof tier !== "string" || !FINGERPRINT_TIERS.has(tier) || typeof value !== "string") {
      return null;
    }
    return [tier as FingerprintTier, value];
  }

  async save(source: string, fingerprint: readonly [FingerprintTier, string]): Promise<void> {
    const [tier, value] = fingerprint;
    const payload = new TextEncoder().encode(
      JSON.stringify({ version: S3_OBJECT_CHECKPOINT_VERSION, source, tier, value }),
    );
    try {
      await this.store.save(this.key(source), payload);
    } catch {
      // Best-effort: re-fetched next pass instead.
    }
  }

  async delete(source: string): Promise<void> {
    try {
      await this.store.delete(this.key(source));
    } catch {
      // Nothing correctness-critical depends on cleanup.
    }
  }
}

const INVENTORY_NAMESPACE = "s3-inventory";
const INVENTORY_VERSION = 1;

function inventoryKey(store: ObjectStore, prefix: string): string {
  return `${INVENTORY_NAMESPACE}:${store.baseUri}/${prefix}`;
}

/**
 * `null` — "no prior run to compare against" — on anything but a clean,
 * version-matching, well-typed prior inventory: a missing or corrupt
 * inventory must never masquerade as "nothing was ever here," which would
 * silently manufacture false deletions (ADR 0007 §9).
 */
async function loadInventory(
  checkpoints: CheckpointStore,
  store: ObjectStore,
  prefix: string,
): Promise<ReadonlySet<string> | null> {
  let data: Uint8Array | null;
  try {
    data = await checkpoints.load(inventoryKey(store, prefix));
  } catch {
    return null; // A store failure is "no prior inventory," not a crash.
  }
  if (data === null) {
    return null;
  }
  let raw: unknown;
  try {
    raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(data));
  } catch {
    return null;
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return null;
  }
  const record = raw as Record<string, unknown>;
  if (record["version"] !== INVENTORY_VERSION) {
    return null;
  }
  const sources = record["sources"];
  if (!Array.isArray(sources) || !sources.every((item) => typeof item === "string")) {
    return null;
  }
  return new Set(sources);
}

async function saveInventory(
  checkpoints: CheckpointStore,
  store: ObjectStore,
  prefix: string,
  sources: ReadonlySet<string>,
): Promise<void> {
  const payload = new TextEncoder().encode(
    JSON.stringify({ version: INVENTORY_VERSION, sources: [...sources].sort() }),
  );
  try {
    await checkpoints.save(inventoryKey(store, prefix), payload);
  } catch {
    // Best-effort: next pass just over-reports "discovered".
  }
}

export type DeletionPolicy = "report" | "retract" | "mirror";

const DELETION_POLICIES: ReadonlySet<DeletionPolicy> = new Set(["report", "retract", "mirror"]);

/**
 * Reaches through `TaguruIngester`'s private `client` field via a runtime
 * cast — mirrors `RunRecorder.attach`'s (observability.ts) own documented
 * exception to "this module never edits ingest.ts": TypeScript's
 * `private`/`readonly` are compile-time-only, and the underlying field is
 * an ordinary JS property. See this module's own doc comment for why,
 * unlike the Python original's `ingester.client is None` guard, this never
 * needs to raise for a "missing" client here — this SDK's
 * `TaguruIngester` always has exactly one.
 */
function clientOf(ingester: TaguruIngester): Taguru {
  return (ingester as unknown as { client: Taguru }).client;
}

async function handleDeletions(options: {
  store: ObjectStore;
  prefix: string;
  checkpoints: CheckpointStore;
  objectCheckpoint: S3ObjectCheckpoint;
  connectorCheckpoint: ConnectorCheckpoint;
  ingester: TaguruIngester;
  seenSources: ReadonlySet<string>;
  listingCompleted: boolean;
  deletionPolicy: DeletionPolicy;
  dryRun: boolean;
}): Promise<[number, number]> {
  const {
    store,
    prefix,
    checkpoints,
    objectCheckpoint,
    connectorCheckpoint,
    ingester,
    seenSources,
    listingCompleted,
    deletionPolicy,
    dryRun,
  } = options;

  if (dryRun) {
    // --dry-run touches nothing at all, including the inventory itself
    // (ADR 0007 §11) — the NEXT real pass still compares against whatever
    // the last real pass left behind.
    return [0, 0];
  }

  const prior = await loadInventory(checkpoints, store, prefix);
  const deleted: ReadonlySet<string> =
    prior !== null ? new Set([...prior].filter((source) => !seenSources.has(source))) : new Set();

  let retracted = 0;
  const needsContext = deleted.size > 0 && (deletionPolicy === "retract" || deletionPolicy === "mirror");
  if (needsContext || (deletionPolicy === "mirror" && listingCompleted)) {
    const client = clientOf(ingester);
    const context = client.context(ingester.context);

    const retract = async (source: string): Promise<void> => {
      await context.retractSource(source);
      // Without this, an object deleted then recreated elsewhere with the
      // SAME (size, lastModified) — the fingerprint tier a non-versioned
      // bucket falls back to — would read as "unchanged" on the very next
      // pass (Layer 1) and never be re-ingested, even though the server
      // no longer has its passage at all.
      await objectCheckpoint.delete(source);
      await connectorCheckpoint.delete(source);
    };

    for (const source of deleted) {
      await retract(source);
      retracted += 1;
    }
    if (deletionPolicy === "mirror" && listingCompleted) {
      // Beyond the inventory diff (which only sees what THIS connector
      // itself previously listed), reconcile against the server's own
      // source list under this prefix — so `mirror` self-heals even on a
      // first run with no prior inventory at all (ADR 0007 §9).
      const mirrorPrefix = `${store.baseUri}/${prefix}`;
      for await (const source of context.iterSources({ prefix: mirrorPrefix })) {
        if (seenSources.has(source) || deleted.has(source)) {
          continue;
        }
        await retract(source);
        retracted += 1;
      }
    }
  }

  if (listingCompleted) {
    await saveInventory(checkpoints, store, prefix, seenSources);
  }

  return [deleted.size, retracted];
}

/**
 * Enumerates `store.list(prefix)` and syncs every object into `ingester`,
 * applying both checkpoint layers, ADR 0007 §9's deletion policy, and
 * `dryRun`.
 *
 * The full listing is drained into memory and every object reported
 * `discovered` BEFORE any object is fetched (ADR 0007 §11's "enumerate all
 * discovered work before the first fetch," issue #353) — a useful side
 * effect: an interruption during fetch/import never forfeits the run's own
 * prefix inventory, since `listingCompleted` here tracks only whether
 * `store.list()` itself raised, not whether processing finished.
 *
 * `deletionPolicy` (default `"report"`, #217's explicit
 * default-no-destructive-sync requirement): `"report"` never retracts — a
 * deleted object only shows up in the returned report's `deletedDetected`
 * count and its own events carry no special marker beyond simply no longer
 * appearing as `discovered` this run. `"retract"`/`"mirror"` are both
 * explicit opt-in. `"mirror"` additionally reconciles against the
 * server's actual source list under this prefix, self-healing even
 * without a prior inventory.
 *
 * `dryRun` tallies every listed object as would-be-`parsed` or
 * `unchanged` — never `discovered`, since the report's counts reflect
 * each source's LAST phase only; the full `discovered` → `parsed`/
 * `unchanged` transition is still visible in the returned events (and
 * `eventsOut`, if given). Nothing is fetched, parsed, ingested, or written
 * to the checkpoint/inventory store. S3's own listing already carries ADR
 * 0007 §9's fingerprint fields, so `unchanged` is exactly as reliable
 * here as on a real run (ADR 0007 §11).
 */
export async function syncObjectStorage(
  store: ObjectStore,
  prefix: string,
  options: {
    ingester: TaguruIngester;
    checkpoints: CheckpointStore;
    connector?: S3Connector;
    deletionPolicy?: DeletionPolicy;
    dryRun?: boolean;
    shouldStop?: ShouldStop;
    eventsOut?: SourceEventSinkTarget;
  },
): Promise<RunReport> {
  const deletionPolicy = options.deletionPolicy ?? "report";
  if (!DELETION_POLICIES.has(deletionPolicy)) {
    throw new Error(`deletionPolicy must be one of ${[...DELETION_POLICIES].sort().join(", ")}`);
  }

  const stop = normalizeStop(options.shouldStop);
  const activeConnector = options.connector ?? new S3Connector(store);
  const objectCheckpoint = new S3ObjectCheckpoint(options.checkpoints);
  const connectorCheckpoint = new ConnectorCheckpoint(options.checkpoints);
  const registry = new SourceIdRegistry();

  // try/finally, not a bare construct-then-close: `eventsOut`'s file
  // handle must close even when something below throws — `recorder.close()`
  // at the end of a straight-line function body never runs in that case.
  // Mirrors the Python original's `with RunRecorder(...) as recorder:`.
  const recorder = new RunRecorder({ connector: PARSER_NAME, eventsOut: options.eventsOut });
  try {
    let listingCompleted = true;
    const metas: ObjectMeta[] = [];
    try {
      for await (const meta of store.list(prefix)) {
        metas.push(meta);
      }
    } catch (error) {
      if (error instanceof TransientStoreError || error instanceof PermanentStoreError) {
        // The LISTING itself failed — nothing was safely enumerated this
        // pass, so the inventory must not be overwritten (ADR 0007 §9: an
        // incomplete pass degrades to "no prior run to compare against"
        // next time, never a false deletion).
        listingCompleted = false;
        const listingSource = `${store.baseUri}/${prefix}`;
        recorder.discovered(listingSource);
        recorder.record(listingSource, "failed", {
          diagnostic: new Diagnostic({ code: "unreadable", message: errorMessage(error), source: listingSource }),
        });
        metas.length = 0;
      } else {
        throw error;
      }
    }

    // Pass 1 (ADR 0007 §11): enumerate and report every object as
    // `discovered` before any of them is fetched.
    const seenSources = new Set<string>();
    const work: Array<[string, ObjectMeta]> = [];
    for (const meta of metas) {
      const source = `${store.baseUri}/${meta.key}`;
      if (!registry.claim(source)) {
        // A duplicate key in the store's own listing.
        recorder.duplicate(source, { of: source });
        continue;
      }
      seenSources.add(source);
      recorder.discovered(source, { sourceBytes: Math.max(meta.size, 0) });
      work.push([source, meta]);
    }

    // Pass 2: fetch/parse/import (or, under dryRun, just the cheap
    // fingerprint comparison the listing already carries).
    let interrupted = false;
    for (const [source, meta] of work) {
      if (stop()) {
        interrupted = true;
        break;
      }

      const fingerprint = objectFingerprint(meta);
      if (fingerprint !== null && fingerprintEquals(await objectCheckpoint.load(source), fingerprint)) {
        recorder.record(source, "unchanged");
        continue;
      }

      if (options.dryRun) {
        recorder.record(source, "parsed");
        continue;
      }

      let result: ReadResult;
      try {
        result = await activeConnector.readObjectResult(meta);
      } catch (error) {
        if (error instanceof ObjectNotFoundError) {
          recorder.record(source, "skipped", {
            diagnostic: new Diagnostic({ code: "unreadable", message: "object no longer exists", source }),
          });
          continue;
        }
        if (error instanceof TransientStoreError || error instanceof PermanentStoreError) {
          recorder.record(source, "failed", {
            diagnostic: new Diagnostic({ code: "unreadable", message: errorMessage(error), source }),
          });
          continue;
        }
        throw error;
      }

      recorder.addDropped({ tags: result.tagsDropped });
      const document = result.document;

      if (!document.text) {
        const diagnostic = document.diagnostics.length > 0 ? document.diagnostics[0]! : null;
        recorder.record(source, "failed", { diagnostic });
        continue;
      }

      recorder.record(source, "parsed", {
        sourceBytes: Math.max(meta.size, 0),
        parser: document.fingerprintInputs.parser,
      });

      // ADR 0007 §6.3: skip the model call/import entirely when this
      // exact fetched-and-parsed content was already ingested — composes
      // with, never replaces, `ingester`'s own per-chunk checkpoint.
      if ((await connectorCheckpoint.load(source, document.fingerprintInputs)) !== null) {
        if (fingerprint !== null) {
          await objectCheckpoint.save(source, fingerprint);
        }
        recorder.record(source, "unchanged");
        continue;
      }

      const outcome = await recorder.attached(options.ingester, () =>
        ingestConnectorDocument(options.ingester, document, { shouldStop: options.shouldStop }),
      );

      recorder.addDropped({
        locators: outcome.locators_dropped,
        sections: outcome.sections_dropped,
      });

      if (outcome.interrupted) {
        recorder.record(source, "skipped");
        interrupted = true;
        break;
      }
      if (!outcome.ok) {
        recorder.record(source, "failed", {
          diagnostic: new Diagnostic({
            code: "unreadable",
            message: outcome.error ?? "ingest failed",
            source,
          }),
        });
        continue;
      }

      await connectorCheckpoint.save(document);
      if (fingerprint !== null) {
        await objectCheckpoint.save(source, fingerprint);
      }
      recorder.record(source, "imported", { parser: document.fingerprintInputs.parser });
    }

    const [deleted, retracted] = await handleDeletions({
      store,
      prefix,
      checkpoints: options.checkpoints,
      objectCheckpoint,
      connectorCheckpoint,
      ingester: options.ingester,
      seenSources: new Set(seenSources),
      listingCompleted,
      deletionPolicy,
      dryRun: options.dryRun ?? false,
    });
    recorder.addDeletions({ detected: deleted, retracted });

    return recorder.finish({ interrupted });
  } finally {
    recorder.close();
  }
}
