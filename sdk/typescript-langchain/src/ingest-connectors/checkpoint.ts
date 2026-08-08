/**
 * The connector's own checkpoint (ADR 0007 §6.3, issue #347): "did I
 * already fetch and parse this object," independent of whether extraction
 * or import has run — the layer neither `taguru extract`'s manifest/chunk
 * checkpoint nor `TaguruIngester`'s own `CheckpointStore` usage covers.
 * The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.checkpoint` module (issue #415).
 *
 * Reuses `CheckpointStore`'s 3-method contract directly (issue #347's
 * explicit requirement: no bespoke store). `ConnectorCheckpoint` is the
 * thin layer on top that knows how to serialize a `ConnectorDocument`
 * and gate reuse on its `fingerprint_inputs` — the same composition
 * `TaguruIngester` already does for its own checkpoint layer, one level
 * downstream.
 */

import type { CheckpointStore } from "../checkpoints.js";
import { ConnectorDocument, FingerprintInputs } from "./document.js";

function decodeJson(data: Uint8Array): unknown | null {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(data));
  } catch {
    return null;
  }
}

/**
 * A connector-native checkpoint over any `CheckpointStore`.
 *
 * `namespace` prefixes every key (`` `${namespace}:${source}` ``) so a
 * connector checkpoint and `TaguruIngester`'s own chunk checkpoint can
 * safely share one `FilesystemCheckpointStore` directory —
 * `checkpointFileName` (checkpoints.ts) derives its file name (and hash
 * suffix) from the full key string, so a namespaced key can never collide
 * with the bare source id the ingester's own checkpoint uses for the same
 * source.
 *
 * Degrades exactly like every other checkpoint in this SDK (ADR 0007
 * §6.3): a missing, corrupt, or fingerprint-mismatched entry means "treat
 * as unseen," never a false hit. `load`/`save` failures are reported via
 * `console.warn` and treated as a cache miss / an unsaved run,
 * respectively; a `delete` failure is silently ignored — the same posture
 * `CheckpointStore`'s own contract already documents.
 */
export class ConnectorCheckpoint {
  private readonly store: CheckpointStore;
  private readonly namespace: string;

  constructor(store: CheckpointStore, options?: { namespace?: string }) {
    this.store = store;
    this.namespace = options?.namespace ?? "connector";
  }

  private key(source: string): string {
    return `${this.namespace}:${source}`;
  }

  /**
   * Resolves to the previously checkpointed document for `source` only
   * when its own `source` field matches the one requested AND its
   * `fingerprint_inputs` exactly matches `fingerprint` — i.e. the raw
   * object is byte-identical to last time under the same
   * parser/parser_version/options. The `source` check is not redundant
   * with the namespaced key: a `CheckpointStore` implementation is only
   * obligated to key by the string it was given (`CheckpointStore`'s own
   * contract, checkpoints.ts), not to guarantee no two keys ever resolve
   * to the same stored bytes — two distinct sources whose fetched bytes
   * happen to be byte-identical would otherwise pass the fingerprint
   * check alone and swap checkpoints under a buggy or lossy custom
   * store. Any mismatch, corruption, or store failure resolves to `null`.
   */
  async load(source: string, fingerprint: FingerprintInputs): Promise<ConnectorDocument | null> {
    let data: Uint8Array | null;
    try {
      data = await this.store.load(this.key(source));
    } catch (error) {
      console.warn(
        `ConnectorCheckpoint.load(${JSON.stringify(source)}) rejected with ${String(error)}; ` +
          "treating as uncached",
      );
      return null;
    }
    if (data === null) {
      return null;
    }
    const raw = decodeJson(data);
    if (raw === null) {
      return null;
    }
    const document = ConnectorDocument.fromDict(raw);
    if (
      document === null ||
      document.source !== source ||
      !document.fingerprintInputs.equals(fingerprint)
    ) {
      return null;
    }
    return document;
  }

  /**
   * Persists `document` in full — not only its fingerprint — so a later
   * `load` hit can skip re-fetching AND re-parsing, the whole point of
   * §6.3's connector-native checkpoint layer.
   */
  async save(document: ConnectorDocument): Promise<void> {
    const payload = new TextEncoder().encode(JSON.stringify(document.toDict()));
    try {
      await this.store.save(this.key(document.source), payload);
    } catch (error) {
      console.warn(
        `ConnectorCheckpoint.save(${JSON.stringify(document.source)}) rejected with ` +
          `${String(error)}; the object will be re-fetched next run`,
      );
    }
  }

  /**
   * Removes any checkpointed document for `source`. Failures are silently
   * ignored — nothing correctness-critical depends on prompt cleanup, the
   * same posture `CheckpointStore.delete` documents.
   */
  async delete(source: string): Promise<void> {
    try {
      await this.store.delete(this.key(source));
    } catch {
      // Nothing correctness-critical depends on prompt cleanup.
    }
  }
}

const FILE_PROBE_VERSION = 1;

/**
 * Cheap local-file metadata (ADR 0007 §11, issue #353) — `fs.stat()`
 * without ever opening/reading the file — plus the connector options that
 * were in effect when it was recorded. All five fields must match for a
 * `--dry-run` pass to report `unchanged`; matching `size`/`mtimeNs` alone
 * is not enough, since a changed `parse_options_digest` means the same
 * bytes would now parse differently.
 */
export class FileProbe {
  readonly size: number;
  readonly mtimeNs: bigint;
  readonly parser: string;
  readonly parserVersion: string;
  readonly parseOptionsDigest: string;

  constructor(fields: {
    size: number;
    mtimeNs: bigint;
    parser: string;
    parserVersion: string;
    parseOptionsDigest: string;
  }) {
    this.size = fields.size;
    this.mtimeNs = fields.mtimeNs;
    this.parser = fields.parser;
    this.parserVersion = fields.parserVersion;
    this.parseOptionsDigest = fields.parseOptionsDigest;
  }

  /** Structural equality — the TS stand-in for the Python dataclass `==`. */
  equals(other: FileProbe): boolean {
    return (
      this.size === other.size &&
      this.mtimeNs === other.mtimeNs &&
      this.parser === other.parser &&
      this.parserVersion === other.parserVersion &&
      this.parseOptionsDigest === other.parseOptionsDigest
    );
  }
}

/**
 * ADR 0007 §11's cheap dry-run gate for local files, mirroring
 * `S3ObjectCheckpoint`'s (s3.ts) shape exactly: "does this object's cheap
 * metadata match what we saw last time" — answerable with no file read at
 * all.
 *
 * Written on every successful REAL-run parse; read only under `dryRun`
 * (§6.3: a real run always hashes the fetched bytes and never trusts
 * metadata alone). A mismatch — including a metadata match but a stale
 * reading from a filesystem with coarse mtime granularity, or a
 * touch-without-edit — degrades to "cannot say `unchanged`," reported as
 * `parsed` (would re-read and compare), never a false `unchanged`; this
 * is the safe direction and the only one this checkpoint ever produces on
 * uncertainty. Composes with (never replaces) `ConnectorCheckpoint`
 * (§6.3): this layer only ever prevents an unnecessary dry-run `parsed`
 * report, never the fetch/parse/import itself. Degrades exactly like
 * every other checkpoint in this SDK: a missing, corrupt, or
 * store-failure entry means "treat as unseen," never a false hit.
 *
 * `mtimeNs` round-trips through JSON as a string — a nanosecond mtime
 * exceeds `Number.MAX_SAFE_INTEGER` for any date past 1970-04, so the
 * Python twin's plain JSON integer would silently lose precision here.
 */
export class FileProbeCheckpoint {
  private readonly store: CheckpointStore;
  private readonly namespace: string;

  constructor(store: CheckpointStore, options?: { namespace?: string }) {
    this.store = store;
    this.namespace = options?.namespace ?? "file-probe";
  }

  private key(source: string): string {
    return `${this.namespace}:${source}`;
  }

  async load(source: string): Promise<FileProbe | null> {
    let data: Uint8Array | null;
    try {
      data = await this.store.load(this.key(source));
    } catch {
      return null; // Any store failure degrades to "uncached".
    }
    if (data === null) {
      return null;
    }
    const raw = decodeJson(data);
    if (
      typeof raw !== "object" ||
      raw === null ||
      Array.isArray(raw) ||
      (raw as Record<string, unknown>)["version"] !== FILE_PROBE_VERSION ||
      (raw as Record<string, unknown>)["source"] !== source
    ) {
      return null;
    }
    const record = raw as Record<string, unknown>;
    const size = record["size"];
    const mtime = record["mtime_ns"];
    if (typeof size !== "number" || !Number.isInteger(size)) {
      return null;
    }
    if (typeof mtime !== "string" || !/^-?\d+$/.test(mtime)) {
      return null;
    }
    const parser = record["parser"];
    const parserVersion = record["parser_version"];
    const parseOptionsDigest = record["parse_options_digest"];
    if (
      typeof parser !== "string" ||
      typeof parserVersion !== "string" ||
      typeof parseOptionsDigest !== "string"
    ) {
      return null;
    }
    return new FileProbe({
      size,
      mtimeNs: BigInt(mtime),
      parser,
      parserVersion,
      parseOptionsDigest,
    });
  }

  async save(source: string, probe: FileProbe): Promise<void> {
    const payload = new TextEncoder().encode(
      JSON.stringify({
        version: FILE_PROBE_VERSION,
        source,
        size: probe.size,
        mtime_ns: probe.mtimeNs.toString(),
        parser: probe.parser,
        parser_version: probe.parserVersion,
        parse_options_digest: probe.parseOptionsDigest,
      }),
    );
    try {
      await this.store.save(this.key(source), payload);
    } catch {
      // Best-effort: re-probed as unseen next pass.
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
