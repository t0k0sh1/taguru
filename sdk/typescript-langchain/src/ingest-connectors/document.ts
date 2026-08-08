/**
 * Normalized document contract (ADR 0007 §5, issue #347): the one shape
 * every connector — present or future — produces, and the one shape
 * `TaguruIngester` (ingest.ts) consumes from a connector instead of
 * reading files directly. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.document` module (issue #415).
 *
 * A field-for-field port of the ADR's JSON example. `text` is plain
 * paragraph-joined text — the same blank-line convention
 * `splitParagraphs` (extract.ts, mirroring src/paragraph.rs) already
 * defines; a connector never numbers its own paragraphs, it produces text
 * and lets that splitter decide where paragraphs begin and end.
 * `locators`/`sections` index into whatever that splitter yields, exactly
 * as `taguru extract` already assumes for its own question/section lines.
 *
 * Degradation posture throughout: the constructor throws on a document
 * that would violate a wire-level invariant the server enforces
 * unconditionally (an empty locator/section field, an oversized tag) — a
 * connector bug should fail loudly and locally rather than ship a document
 * the server would refuse anyway. What the server instead drops-and-counts
 * (a paragraph index outside the document's own range) is deliberately NOT
 * validated here; that decision is the server's alone (`filter_markers`,
 * src/passages.rs:203) and a connector cannot know the range without
 * re-running the same paragraph split the server runs.
 */

import type { Locator } from "taguru";

/**
 * Checked for equality, like `taguru`'s own `BATCH_VERSION`/
 * `GROUP_VERSION` (src/ingest.rs) — a protocol-breaking change bumps
 * this, never loosens the check to a range.
 */
export const CONNECTOR_DOCUMENT_VERSION = 1;

// Mirrors the caps src/api.rs enforces server-side (MAX_NAME_BYTES,
// MAX_LOCATOR_KIND_BYTES, MAX_LOCATOR_VALUE_BYTES, MAX_SECTION_BYTES,
// MAX_TAG_BYTES, MAX_TAGS_PER_SOURCE) — checked here too so a connector
// bug fails at construction time, not only after a round trip to the
// server.
export const MAX_NAME_BYTES = 1024;
export const MAX_LOCATOR_KIND_BYTES = 64;
export const MAX_LOCATOR_VALUE_BYTES = 512;
export const MAX_SECTION_BYTES = 512;
export const MAX_TAG_BYTES = 128;
export const MAX_TAGS_PER_SOURCE = 32;

/**
 * ADR 0007 §8's closed, versioned diagnostic vocabulary. Adding a new
 * code is additive (compatible, ADR 0005 §4); renaming or repurposing an
 * existing one is breaking — the same posture src/api.rs's `ErrorCode`
 * documents for itself.
 *
 * `duplicate_source` (added by issue #353's observability driver,
 * `syncReferences`): a reference resolves to a source id already claimed
 * earlier in the same run (an exact duplicate input, or two inputs that
 * normalize to the same id — e.g. two URLs differing only by a tracking
 * query parameter `canonicalizeUrl` strips). The second and later
 * occurrences are reported `skipped` with this code; only the first is
 * ever fetched/parsed/imported.
 */
export type DiagnosticCode =
  | "unreadable"
  | "unsupported_format"
  | "encrypted"
  | "corrupt"
  | "ocr_required"
  | "source_id_too_long"
  | "content_too_large"
  | "partial_extraction"
  | "duplicate_source";

export const DIAGNOSTIC_CODES: ReadonlySet<string> = new Set<string>([
  "unreadable",
  "unsupported_format",
  "encrypted",
  "corrupt",
  "ocr_required",
  "source_id_too_long",
  "content_too_large",
  "partial_extraction",
  "duplicate_source",
]);

export function byteLen(text: string): number {
  return new TextEncoder().encode(text).length;
}

function checkNonemptySize(fieldName: string, value: string, maxBytes: number): void {
  if (!value) {
    throw new Error(`${fieldName} must not be empty`);
  }
  if (byteLen(value) > maxBytes) {
    throw new Error(`${fieldName} exceeds ${maxBytes} bytes`);
  }
}

/**
 * Parses a JSON array field where a single unparseable element
 * invalidates the whole list — the same "one bad unit invalidates the
 * file" posture the checkpoint parser in checkpoints.ts already takes,
 * applied per-field instead of per-file.
 */
function parseList<T>(items: unknown, parser: (item: unknown) => T | null): T[] | null {
  if (!Array.isArray(items)) {
    return null;
  }
  const result: T[] = [];
  for (const item of items) {
    const parsed = parser(item);
    if (parsed === null) {
      return null;
    }
    result.push(parsed);
  }
  return result;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** `String(value)` for present values, `null` passed through — mirrors the
 * Python port's `_optional_str`. */
function optionalStr(value: unknown): string | null {
  return value === null || value === undefined ? null : String(value);
}

/** `String(value)` only when the key is present; a missing required key
 * aborts the parse by returning `null` at the call site. */
function requiredStr(data: Record<string, unknown>, key: string): string | null {
  return key in data ? String(data[key]) : null;
}

/**
 * One structured diagnostic (ADR 0007 §8) — the same three fields
 * `taguru extract`'s own diagnostics sidecar uses for its own per-chunk/
 * per-document records (`DiagnosticsSink`/`AttemptRecord`,
 * src/extract.rs:2372/:2566), so a downstream tool that already parses one
 * JSONL diagnostics shape does not need a second parser.
 */
export class Diagnostic {
  readonly code: DiagnosticCode;
  readonly message: string;
  readonly source: string;

  constructor(fields: { code: DiagnosticCode; message: string; source: string }) {
    this.code = fields.code;
    this.message = fields.message;
    this.source = fields.source;
  }

  toDict(): Record<string, unknown> {
    return { code: this.code, message: this.message, source: this.source };
  }
}

function parseDiagnostic(data: unknown): Diagnostic | null {
  if (!isRecord(data)) {
    return null;
  }
  const code = data["code"];
  if (typeof code !== "string" || !DIAGNOSTIC_CODES.has(code)) {
    return null;
  }
  const message = requiredStr(data, "message");
  const source = requiredStr(data, "source");
  if (message === null || source === null) {
    return null;
  }
  return new Diagnostic({ code: code as DiagnosticCode, message, source });
}

/**
 * One paragraph-indexed typed locator (ADR 0007 §7): a page, slide,
 * sheet, table, or other position. Independent of `SectionEntry` — unlike
 * a section, a locator does not extend to the next paragraph; a connector
 * emits one locator line per paragraph it needs to place.
 */
export class LocatorEntry {
  readonly paragraph: number;
  readonly locator: Locator;

  constructor(fields: { paragraph: number; locator: Locator }) {
    this.paragraph = fields.paragraph;
    this.locator = fields.locator;
  }

  toDict(): Record<string, unknown> {
    return {
      paragraph: this.paragraph,
      locator: { kind: this.locator.kind, value: this.locator.value },
    };
  }
}

/** Mirrors the Python port's `int(...)` coercion: accepts an integral
 * number (or integral numeric string) and rejects everything else. */
function parseIntStrict(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isInteger(value) ? value : null;
  }
  if (typeof value === "string" && /^[+-]?\d+$/.test(value.trim())) {
    return Number.parseInt(value, 10);
  }
  return null;
}

function parseLocatorEntry(data: unknown): LocatorEntry | null {
  if (!isRecord(data)) {
    return null;
  }
  const locatorData = data["locator"];
  if (!isRecord(locatorData)) {
    return null;
  }
  const paragraph = parseIntStrict(data["paragraph"]);
  const kind = requiredStr(locatorData, "kind");
  const value = requiredStr(locatorData, "value");
  if (paragraph === null || kind === null || value === null) {
    return null;
  }
  return new LocatorEntry({ paragraph, locator: { kind, value } });
}

/**
 * One paragraph-indexed free-text section heading — extends to the next
 * marker, exactly like a batch file's `{"paragraph": N, "section": ...}`
 * line (src/ingest.rs's `SectionLine`). `section`'s meaning is unchanged
 * by ADR 0007 §7.1: prose, never overloaded with positional metadata.
 */
export class SectionEntry {
  readonly paragraph: number;
  readonly section: string;

  constructor(fields: { paragraph: number; section: string }) {
    this.paragraph = fields.paragraph;
    this.section = fields.section;
  }

  toDict(): Record<string, unknown> {
    return { paragraph: this.paragraph, section: this.section };
  }
}

function parseSectionEntry(data: unknown): SectionEntry | null {
  if (!isRecord(data)) {
    return null;
  }
  const paragraph = parseIntStrict(data["paragraph"]);
  const section = requiredStr(data, "section");
  if (paragraph === null || section === null) {
    return null;
  }
  return new SectionEntry({ paragraph, section });
}

/**
 * ADR 0007 §5's `metadata` object — only what `Citation`/`SourceEntry`
 * can already represent or what §7 adds. A connector must not invent
 * metadata the contract doesn't declare.
 */
export class ConnectorMetadata {
  readonly originUri: string;
  readonly displayName: string;
  readonly title: string | null;
  readonly canonicalUrl: string | null;
  readonly tags: readonly string[];
  readonly contentType: string | null;

  constructor(fields: {
    originUri: string;
    displayName: string;
    title?: string | null;
    canonicalUrl?: string | null;
    tags?: readonly string[];
    contentType?: string | null;
  }) {
    this.originUri = fields.originUri;
    this.displayName = fields.displayName;
    this.title = fields.title ?? null;
    this.canonicalUrl = fields.canonicalUrl ?? null;
    this.tags = Object.freeze([...(fields.tags ?? [])]);
    this.contentType = fields.contentType ?? null;
  }

  toDict(): Record<string, unknown> {
    return {
      origin_uri: this.originUri,
      display_name: this.displayName,
      title: this.title,
      canonical_url: this.canonicalUrl,
      tags: [...this.tags],
      content_type: this.contentType,
    };
  }

  static fromDict(data: unknown): ConnectorMetadata | null {
    if (!isRecord(data)) {
      return null;
    }
    const rawTags = data["tags"] ?? [];
    if (!Array.isArray(rawTags) || !rawTags.every((tag) => typeof tag === "string")) {
      return null;
    }
    const originUri = requiredStr(data, "origin_uri");
    const displayName = requiredStr(data, "display_name");
    if (originUri === null || displayName === null) {
      return null;
    }
    return new ConnectorMetadata({
      originUri,
      displayName,
      title: optionalStr(data["title"]),
      canonicalUrl: optionalStr(data["canonical_url"]),
      tags: rawTags,
      contentType: optionalStr(data["content_type"]),
    });
  }
}

/**
 * ADR 0007 §5/§6.3's connector-native checkpoint key material.
 *
 * `rawContentSha256` hashes the RAW bytes fetched/opened from the origin,
 * before parsing — deliberately not `text` — so a connector can detect
 * "the object is byte-identical to last time" without re-parsing at all.
 * This is a different, deliberately non-interchangeable question from
 * `taguru extract`'s own manifest fingerprint, which hashes the parsed
 * `text` (ADR 0007 §5's `fingerprint_inputs` note).
 */
export class FingerprintInputs {
  readonly rawContentSha256: string;
  readonly parser: string;
  readonly parserVersion: string;
  readonly parseOptionsDigest: string;

  constructor(fields: {
    rawContentSha256: string;
    parser: string;
    parserVersion: string;
    parseOptionsDigest: string;
  }) {
    this.rawContentSha256 = fields.rawContentSha256;
    this.parser = fields.parser;
    this.parserVersion = fields.parserVersion;
    this.parseOptionsDigest = fields.parseOptionsDigest;
  }

  /** Structural equality — the TS stand-in for the Python dataclass `==`
   * the checkpoint gate compares fingerprints with. */
  equals(other: FingerprintInputs): boolean {
    return (
      this.rawContentSha256 === other.rawContentSha256 &&
      this.parser === other.parser &&
      this.parserVersion === other.parserVersion &&
      this.parseOptionsDigest === other.parseOptionsDigest
    );
  }

  toDict(): Record<string, unknown> {
    return {
      raw_content_sha256: this.rawContentSha256,
      parser: this.parser,
      parser_version: this.parserVersion,
      parse_options_digest: this.parseOptionsDigest,
    };
  }

  static fromDict(data: unknown): FingerprintInputs | null {
    if (!isRecord(data)) {
      return null;
    }
    const rawContentSha256 = requiredStr(data, "raw_content_sha256");
    const parser = requiredStr(data, "parser");
    const parserVersion = requiredStr(data, "parser_version");
    const parseOptionsDigest = requiredStr(data, "parse_options_digest");
    if (
      rawContentSha256 === null ||
      parser === null ||
      parserVersion === null ||
      parseOptionsDigest === null
    ) {
      return null;
    }
    return new FingerprintInputs({ rawContentSha256, parser, parserVersion, parseOptionsDigest });
  }
}

/**
 * A canonical sha256 of a connector's own effective config —
 * `fingerprint_inputs.parse_options_digest` (ADR 0007 §5). Key order
 * never matters (keys are sorted before serializing); this is not itself
 * a wire format, only a stable fingerprint of one. Async because hashing
 * uses the Web Crypto global, the same constraint checkpoints.ts
 * documents.
 *
 * Canonicalization matches the Python port's
 * `json.dumps(options, sort_keys=True, separators=(",", ":"))`: keys
 * sorted at every nesting level, no whitespace, non-ASCII kept verbatim.
 */
export async function optionsDigest(options: Record<string, unknown>): Promise<string> {
  const canonical = JSON.stringify(sortKeysDeep(options));
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function sortKeysDeep(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortKeysDeep);
  }
  if (isRecord(value)) {
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      sorted[key] = sortKeysDeep(value[key]);
    }
    return sorted;
  }
  return value;
}

/**
 * The one shape every connector — present or future — produces (ADR 0007
 * §5), and the one shape a connector hands to `ingestConnectorDocument`
 * (bridge.ts) instead of a bare string.
 *
 * `diagnostics` is empty on a clean parse; a non-empty `diagnostics` with
 * an empty `text` is the required encoding of "nothing usable was
 * extracted" (scanned PDF, encrypted file, unsupported format) — never a
 * silently empty `text` with no diagnostic, which is exactly the "quietly
 * degraded output" #217 forbids and the constructor refuses to construct.
 */
export class ConnectorDocument {
  readonly source: string;
  readonly text: string;
  readonly locators: readonly LocatorEntry[];
  readonly sections: readonly SectionEntry[];
  readonly metadata: ConnectorMetadata;
  readonly fingerprintInputs: FingerprintInputs;
  readonly diagnostics: readonly Diagnostic[];

  constructor(fields: {
    source: string;
    text: string;
    locators?: readonly LocatorEntry[];
    sections?: readonly SectionEntry[];
    metadata: ConnectorMetadata;
    fingerprintInputs: FingerprintInputs;
    diagnostics?: readonly Diagnostic[];
  }) {
    this.source = fields.source;
    this.text = fields.text;
    this.locators = Object.freeze([...(fields.locators ?? [])]);
    this.sections = Object.freeze([...(fields.sections ?? [])]);
    this.metadata = fields.metadata;
    this.fingerprintInputs = fields.fingerprintInputs;
    this.diagnostics = Object.freeze([...(fields.diagnostics ?? [])]);

    if (!this.source) {
      throw new Error("ConnectorDocument.source must not be empty");
    }
    if (byteLen(this.source) > MAX_NAME_BYTES && this.text) {
      throw new Error(
        `source exceeds ${MAX_NAME_BYTES} bytes (MAX_NAME_BYTES, src/api.rs:1137) ` +
          "— refuse with a 'source_id_too_long' diagnostic and empty text instead of " +
          "a source id the server would reject anyway",
      );
    }
    if (!this.text && this.diagnostics.length === 0) {
      throw new Error(
        "an empty text must carry at least one diagnostic explaining why " +
          "(ADR 0007 §5/§10: never a silently empty passage)",
      );
    }
    for (const entry of this.locators) {
      if (entry.paragraph < 0) {
        throw new Error("locator paragraph index must be >= 0");
      }
      checkNonemptySize("locator.kind", entry.locator.kind, MAX_LOCATOR_KIND_BYTES);
      checkNonemptySize("locator.value", entry.locator.value, MAX_LOCATOR_VALUE_BYTES);
    }
    for (const section of this.sections) {
      if (section.paragraph < 0) {
        throw new Error("section paragraph index must be >= 0");
      }
      checkNonemptySize("section", section.section, MAX_SECTION_BYTES);
    }
    if (this.metadata.tags.length > MAX_TAGS_PER_SOURCE) {
      throw new Error(`more than ${MAX_TAGS_PER_SOURCE} tags (MAX_TAGS_PER_SOURCE)`);
    }
    for (const tag of this.metadata.tags) {
      checkNonemptySize("tag", tag, MAX_TAG_BYTES);
    }
  }

  toDict(): Record<string, unknown> {
    return {
      connector_document: CONNECTOR_DOCUMENT_VERSION,
      source: this.source,
      text: this.text,
      locators: this.locators.map((entry) => entry.toDict()),
      sections: this.sections.map((entry) => entry.toDict()),
      metadata: this.metadata.toDict(),
      fingerprint_inputs: this.fingerprintInputs.toDict(),
      diagnostics: this.diagnostics.map((diagnostic) => diagnostic.toDict()),
    };
  }

  /**
   * `null` on any structural mismatch, including a `connector_document`
   * version this reader does not recognize — the same
   * degrade-never-throw posture every `fromDict`-shaped parser in this
   * SDK already takes.
   */
  static fromDict(data: unknown): ConnectorDocument | null {
    if (!isRecord(data)) {
      return null;
    }
    if (data["connector_document"] !== CONNECTOR_DOCUMENT_VERSION) {
      return null;
    }
    const metadata = ConnectorMetadata.fromDict(data["metadata"]);
    const fingerprintInputs = FingerprintInputs.fromDict(data["fingerprint_inputs"]);
    if (metadata === null || fingerprintInputs === null) {
      return null;
    }
    const locators = parseList(data["locators"] ?? [], parseLocatorEntry);
    const sections = parseList(data["sections"] ?? [], parseSectionEntry);
    const diagnostics = parseList(data["diagnostics"] ?? [], parseDiagnostic);
    if (locators === null || sections === null || diagnostics === null) {
      return null;
    }
    const source = requiredStr(data, "source");
    const text = requiredStr(data, "text");
    if (source === null || text === null) {
      return null;
    }
    try {
      return new ConnectorDocument({
        source,
        text,
        locators,
        sections,
        metadata,
        fingerprintInputs,
        diagnostics,
      });
    } catch {
      return null;
    }
  }
}
