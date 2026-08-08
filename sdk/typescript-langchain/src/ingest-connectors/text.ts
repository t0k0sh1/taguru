/**
 * The reference connector (ADR 0007 §5, issue #347): `.md`/`.txt` files
 * into `ConnectorDocument`. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.text` module (issue #415).
 *
 * The minimal implementation the issue asks for — proof that the
 * normalized document contract reaches `TaguruIngester` end to end, not a
 * full Markdown parser. Heading extraction is a single-line ATX-heading
 * heuristic (`#` to `######`, plus an optional closing sequence per the
 * CommonMark rule that it must be preceded by a space); anything more
 * (setext headings, headings sharing a paragraph with body text because a
 * blank line is missing) is simply not recognized as a section boundary,
 * which degrades quietly, never incorrectly — a paragraph that doesn't
 * parse as a heading just carries no `section`, exactly like
 * `taguru extract` itself has never emitted sections.
 */

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../extract.js";
import {
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  FingerprintInputs,
  MAX_SECTION_BYTES,
  optionsDigest,
  SectionEntry,
  type DiagnosticCode,
} from "./document.js";
import type { Connector } from "./protocol.js";
import { checkSourceId, fileSourceId } from "./sources.js";

export const PARSER_NAME = "taguru-text-connector";
export const PARSER_VERSION = "1.0.0";

const SUPPORTED_SUFFIXES: ReadonlySet<string> = new Set([".md", ".txt"]);

// A single-line ATX heading: 1-6 '#', required whitespace, the title, and an
// optional closing '#' run that (per CommonMark) must itself be preceded by
// whitespace to count as a closer rather than literal trailing text (so
// "# C#" keeps its trailing '#', but "## Heading ##" does not).
const HEADING_RE = /^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?$/;

const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

function byteLen(text: string): number {
  return new TextEncoder().encode(text).length;
}

/** SHA-256 of raw bytes (never decoded/re-encoded through a string), as
 * lowercase hex. Distinct from checkpoints.ts's `sha256Hex`, which hashes a
 * string's UTF-8 encoding — this connector must hash bytes that are not
 * necessarily valid UTF-8 at all (that's exactly what the `corrupt`
 * diagnostic below detects). */
async function sha256HexBytes(data: Uint8Array): Promise<string> {
  // `data` may be a Node `Buffer` (typed as `Uint8Array<ArrayBufferLike>`,
  // since it can be backed by a pooled/shared allocation), which
  // `SubtleCrypto.digest`'s DOM-lib `BufferSource` type does not accept
  // directly — copy into a plain `ArrayBuffer`-backed view first.
  const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(data));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

/** The final path component — the TS stand-in for `pathlib.Path(...).name`,
 * used only for display and for extension sniffing, never for identity
 * (`fileSourceId` keeps the caller-given reference verbatim). */
function pathBasename(reference: string): string {
  const trimmed = reference.replace(/[/\\]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index === -1 ? trimmed : trimmed.slice(index + 1);
}

/** The final path component's extension, dot included — the TS stand-in
 * for `pathlib.Path(...).suffix` (a leading dot with no further dot, e.g.
 * `.bashrc`, has no suffix, matching pathlib). */
function pathSuffix(reference: string): string {
  const name = pathBasename(reference);
  const dotIndex = name.lastIndexOf(".");
  return dotIndex <= 0 ? "" : name.slice(dotIndex);
}

/**
 * Reference connector for `.md`/`.txt` files.
 *
 * `extractHeadings` (default `true`) controls whether `.md` ATX headings
 * are emitted as `sections` — set `false` to reproduce a plain-text
 * passage with no section metadata even for Markdown input. `.txt` never
 * carries sections regardless of this flag: it has no heading syntax to
 * extract.
 */
export class TextFileConnector implements Connector {
  readonly parser: string = PARSER_NAME;
  readonly parserVersion: string = PARSER_VERSION;

  private readonly extractHeadings: boolean;

  constructor(options?: { extractHeadings?: boolean }) {
    this.extractHeadings = options?.extractHeadings ?? true;
  }

  /**
   * The digest `read` would stamp into
   * `fingerprintInputs.parseOptionsDigest` for THIS instance's effective
   * config, computable without reading anything (ADR 0007 §6.3, added for
   * issue #351's connector-level checkpoint: it gates skipping a re-parse
   * before ever calling `read`, which needs the full candidate fingerprint
   * up front).
   */
  async parseOptionsDigest(): Promise<string> {
    return this.optionsDigestValue();
  }

  supports(reference: string): boolean {
    return SUPPORTED_SUFFIXES.has(pathSuffix(reference).toLowerCase());
  }

  private async optionsDigestValue(): Promise<string> {
    return optionsDigest({ extract_headings: this.extractHeadings });
  }

  private async fingerprint(rawContentSha256: string): Promise<FingerprintInputs> {
    return new FingerprintInputs({
      rawContentSha256,
      parser: this.parser,
      parserVersion: this.parserVersion,
      parseOptionsDigest: await this.optionsDigestValue(),
    });
  }

  private async failure(fields: {
    source: string;
    displayName: string;
    contentType: string;
    code: DiagnosticCode;
    message: string;
    rawContentSha256: string;
  }): Promise<ConnectorDocument> {
    return new ConnectorDocument({
      source: fields.source,
      text: "",
      metadata: new ConnectorMetadata({
        originUri: fields.source,
        displayName: fields.displayName,
        contentType: fields.contentType,
      }),
      fingerprintInputs: await this.fingerprint(fields.rawContentSha256),
      diagnostics: [new Diagnostic({ code: fields.code, message: fields.message, source: fields.source })],
    });
  }

  async read(reference: string): Promise<ConnectorDocument> {
    // `reference` itself is the source id — mirrors taguru extract's
    // `path.to_string_lossy()` verbatim (ADR 0007 §6.1), never a
    // path-library-normalized rewrite of it (normalizing silently drops a
    // leading "./", collapses "a//b", ...), which would make the same
    // caller-given reference mint two different source ids depending on
    // incidental spelling. `pathBasename`/`pathSuffix` are still used
    // below, but only for filesystem access (suffix/stat/read), never for
    // identity.
    const source = fileSourceId(reference);
    const displayName = pathBasename(reference);
    const isMarkdown = pathSuffix(reference).toLowerCase() === ".md";
    const contentType = isMarkdown ? "text/markdown" : "text/plain";

    const failure = (
      code: DiagnosticCode,
      message: string,
      rawContentSha256: string = EMPTY_SHA256,
    ): Promise<ConnectorDocument> =>
      this.failure({ source, displayName, contentType, code, message, rawContentSha256 });

    const sourceDiagnostic = checkSourceId(source);
    if (sourceDiagnostic !== null) {
      return failure(sourceDiagnostic.code, sourceDiagnostic.message);
    }

    if (!this.supports(reference)) {
      return failure(
        "unsupported_format",
        `unsupported extension ${JSON.stringify(pathSuffix(reference))} (only .md/.txt)`,
      );
    }

    const { readFile, stat } = await import("node:fs/promises");

    let size: number;
    try {
      size = (await stat(reference)).size;
    } catch (error) {
      return failure("unreadable", String(error));
    }
    if (size > MAX_PASSAGE_BYTES) {
      return failure(
        "content_too_large",
        `${size} bytes exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`,
      );
    }

    let raw: Uint8Array;
    try {
      raw = await readFile(reference);
    } catch (error) {
      return failure("unreadable", String(error));
    }
    const rawContentSha256 = await sha256HexBytes(raw);

    let text: string;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
    } catch (error) {
      return failure("corrupt", String(error), rawContentSha256);
    }

    // paragraph splitting (and its Rust twin, src/paragraph.rs) does no
    // normalization at all — a leading BOM would otherwise land inside
    // paragraph 0 verbatim. Mirrors extract.ts's read-document path.
    if (text.startsWith("﻿")) {
      text = text.slice(1);
    }

    let sections: SectionEntry[] = [];
    let title: string | null = null;
    if (isMarkdown && this.extractHeadings) {
      const headings = this.headings(text);
      sections = headings.sections;
      title = headings.title;
    }

    return new ConnectorDocument({
      source,
      text,
      sections,
      metadata: new ConnectorMetadata({
        originUri: source,
        displayName,
        title,
        contentType,
      }),
      fingerprintInputs: await this.fingerprint(rawContentSha256),
    });
  }

  /**
   * One pass over `splitParagraphs(text)` yielding both the section
   * entries and the title, so a single-line-ATX-heading paragraph is
   * matched against `HEADING_RE` once rather than twice (previously two
   * independent methods, one per result).
   *
   * A section entry is one per paragraph whose ENTIRE content is a single
   * ATX heading line — never a heading sharing a paragraph with body text,
   * since that would require guessing where the heading ends without a
   * blank-line boundary to trust. Paragraph indices are
   * `splitParagraphs(text)`'s own indices, the exact contract ADR 0007 §5
   * requires (a connector never numbers its own paragraphs).
   *
   * The title is the first H1 (`# ...`) heading, if any — evaluated
   * independently of a section entry's `MAX_SECTION_BYTES` cap, since a
   * title is metadata, not a citation locator, and carries no wire-level
   * size limit of its own.
   */
  private headings(text: string): { sections: SectionEntry[]; title: string | null } {
    const entries: SectionEntry[] = [];
    let title: string | null = null;
    splitParagraphs(text).forEach((paragraph, index) => {
      const match = HEADING_RE.exec(paragraph);
      if (match === null) {
        return;
      }
      const hashes = match[1] ?? "";
      const label = (match[2] ?? "").trim();
      if (title === null && hashes.length === 1 && label) {
        title = label;
      }
      if (label && byteLen(label) <= MAX_SECTION_BYTES) {
        entries.push(new SectionEntry({ paragraph: index, section: label }));
      }
    });
    return { sections: entries, title };
  }
}
