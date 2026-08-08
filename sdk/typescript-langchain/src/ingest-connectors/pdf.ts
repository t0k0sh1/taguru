/**
 * The PDF connector (ADR 0007 §7/§8/§10, issue #348): `.pdf` files into
 * `ConnectorDocument` (document.ts), one `{"kind": "page", "value": ...}`
 * locator per paragraph and, when the document carries one, its outline
 * (bookmarks) as `sections`. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.pdf` module (issue #415).
 *
 * The thing `docs/local-rag-walkthrough.html` used to require of every
 * caller — "Converting a PDF to text is this pipeline's job — plain
 * `pypdf`... before any Taguru code runs at all" — done once, here, against
 * the normalized document contract instead of a bare string.
 *
 * No OCR engine ships in this connector (ADR 0007 §10): a page whose
 * extracted text has fewer than `minCharsPerPage` non-whitespace characters
 * is treated as having no usable text layer and is named in an
 * `ocr_required` diagnostic — never silently passed through as low-quality
 * text. Absent an `ocrAdapter` (issue #352, `OcrAdapter` in ocr.ts), that
 * diagnostic is terminal for the pages it names, exactly as before this
 * connector could accept one at all. When one IS configured, it is asked to
 * recover exactly the pages this connector itself found unusable — never
 * the whole document, and never a page this connector already extracted
 * usable text from — and each recovered page that clears the same
 * `minCharsPerPage` bar every other page's text must clear is spliced back
 * in as if this connector had extracted it itself: its own
 * `{"kind": "page", ...}` locator, its own place in the outline
 * fall-forward, its own contribution to `sections`. An adapter's own
 * failure (a rejection, or simply recovering nothing) leaves the pages it
 * was asked about exactly as `ocr_required` as if no adapter had been
 * configured — the adapter's own error is itself never surfaced to a
 * caller, since it is arbitrary external code, no more trusted than the
 * parser itself.
 *
 * Parser: optional `pdfjs-dist` (the Node-friendly
 * `pdfjs-dist/legacy/build/pdf.mjs` entry, workers disabled — Node
 * auto-disables pdfjs-dist's worker thread, so no extra configuration is
 * needed), loaded ONLY via a dynamic `import()` inside `read()` — never at
 * module scope — so importing this module never requires `pdfjs-dist` to be
 * installed; only calling `read()` does. Unlike the Python twin (which
 * raises `ImportError` synchronously at `PdfConnector` construction, since
 * `import pypdf` happens at module load), a dynamic `import()` cannot be
 * awaited inside a constructor — the equivalent check necessarily happens on
 * the first call to `read()` instead, with the same message pattern pypdf's
 * ImportError uses.
 *
 * pypdf -> pdfjs-dist API mapping this module leans on:
 * - `PdfReader(bytes)` -> `getDocument({ data }).promise`; pypdf's two-step
 *   "open regardless of encryption, then explicit `reader.decrypt("")`"
 *   collapses into pdfjs-dist's own single-step open, which already
 *   attempts an empty password and rejects with a `PasswordException`-shaped
 *   error (checked structurally on `error.name`, never `instanceof`, since
 *   the class is not part of pdfjs-dist's public export surface) when that
 *   fails — mapped to the same `encrypted` diagnostic.
 * - `reader.pages[i].extract_text()` -> `page.getTextContent()`, reassembled
 *   from its `items` (`str`, `hasEOL`, `transform`, `height`) by
 *   `assemblePageText` below. This is deliberately NOT a byte-identical
 *   port: pdfjs-dist's `items` carry no embedded line breaks the way a
 *   pypdf string does (a literal `\n` inside a PDF content-stream string has
 *   no glyph in a base font's encoding and is dropped, not preserved), so
 *   line/paragraph structure is instead reconstructed from each item's
 *   vertical position — a normal line-height gap becomes `\n`, a
 *   double-or-more gap becomes `\n\n` (see `PARAGRAPH_GAP_RATIO` below). The
 *   goal, per issue #415, is behavioral parity for what this connector DOES
 *   with the text (per-page text, the `minCharsPerPage` usable-text gate,
 *   paragraph derivation) — not byte-identical extraction.
 * - pypdf's nested-list outline walk -> `doc.getOutline()`'s tree of nodes
 *   (each with its own `items` children), page-resolved via
 *   `doc.getPageIndex()` on the destination's page reference; the same
 *   fall-forward-to-the-next-usable-page logic this connector already
 *   implements is preserved unchanged.
 * - Any parse rejection -> the same diagnostics (`corrupt`, etc.) this
 *   connector emits for pypdf's own parse exceptions.
 */

import type * as PdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../extract.js";
import {
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  type DiagnosticCode,
  FingerprintInputs,
  LocatorEntry,
  MAX_SECTION_BYTES,
  SectionEntry,
  optionsDigest,
} from "./document.js";
import { OcrRequest, type OcrAdapter, type OcrResult } from "./ocr.js";
import type { Connector } from "./protocol.js";
import { checkSourceId, fileSourceId } from "./sources.js";

export const PARSER_NAME = "taguru-pdf-connector";
export const PARSER_VERSION = "1.0.0";

const SUPPORTED_SUFFIXES: ReadonlySet<string> = new Set([".pdf"]);

// The raw-file cap, checked before any parsing — independent of
// MAX_PASSAGE_BYTES (checked again below, against the EXTRACTED text): a
// huge scanned PDF can stay well under the text cap, and a huge but
// well-compressed text PDF can still exceed it, so a PDF connector needs
// both, unlike a plain text connector where the file bytes ARE the text.
const DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024;

// ADR 0007 §10's own words: "a connector-defined, documented threshold —
// e.g. fewer than N extractable characters per page after whitespace
// normalization." 16 was picked as comfortably above what a stray running
// header/footer or page number alone would produce, comfortably below a
// single real sentence.
const DEFAULT_MIN_CHARS_PER_PAGE = 16;

// How many page numbers one diagnostic message lists — a document with
// hundreds of scanned pages must not make one diagnostic's message
// unbounded.
const MAX_LISTED_PAGES = 20;

// sha256("") — computed once, verbatim, rather than hashed at every
// construction of a pre-parse failure document.
const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

// -- assemblePageText's own vertical-gap heuristic (see the module docstring) --
//
// Both ratios are relative to the CURRENT item's own font-derived `height`
// (pdfjs-dist's per-item text height, not a fixed pixel constant), so the
// heuristic scales with whatever font size a page actually uses. A gap
// under LINE_GAP_RATIO is read as "still the same visual line" (no
// separator); at or above it but under PARAGRAPH_GAP_RATIO, a single
// line break; at or above PARAGRAPH_GAP_RATIO (roughly a skipped line), a
// paragraph break. This is intentionally the SAME heuristic tests/pdfs.ts's
// fixture builder is written against — see that module's own docstring.
const LINE_GAP_RATIO = 0.1;
const PARAGRAPH_GAP_RATIO = 1.5;

function byteLen(text: string): number {
  return new TextEncoder().encode(text).length;
}

/**
 * Non-whitespace character count — the OCR-required threshold (ADR 0007
 * §10) is judged on this "after whitespace normalization", so a page of
 * only line breaks/blank runs (a common near-empty-page artifact, e.g. a
 * section divider) is correctly treated as having no usable text layer.
 */
function pageCharCount(text: string): number {
  let count = 0;
  for (const char of text) {
    if (!/\s/u.test(char)) {
      count += 1;
    }
  }
  return count;
}

function listPages(pages: readonly number[]): string {
  let shown = pages.slice(0, MAX_LISTED_PAGES).join(", ");
  const remainder = pages.length - MAX_LISTED_PAGES;
  if (remainder > 0) {
    shown += `, … and ${remainder} more`;
  }
  return shown;
}

function ocrRequiredMessage(pages: readonly number[], adapterError: string | null): string {
  let message = `no extractable text layer on page(s) ${listPages(pages)}`;
  if (adapterError !== null) {
    message += ` (OCR adapter failed: ${adapterError})`;
  }
  return message;
}

/** Structural check (never `instanceof`, since pdfjs-dist does not export
 * its `PasswordException` class from the entry point this module imports)
 * for "this rejection means the document needs a password this connector
 * does not have." */
function isPasswordException(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as { name?: unknown }).name === "PasswordException"
  );
}

/** SHA-256 of raw bytes (never decoded/re-encoded through a string), as
 * lowercase hex. Distinct from checkpoints.ts's `sha256Hex`, which hashes a
 * string's UTF-8 encoding — this connector must hash bytes that need not be
 * valid UTF-8 at all. */
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

function parseIntStrict(value: string): number | null {
  const trimmed = value.trim();
  return /^[+-]?\d+$/.test(trimmed) ? Number.parseInt(trimmed, 10) : null;
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
 * `.pdf` alone, has no suffix, matching pathlib). */
function pathSuffix(reference: string): string {
  const name = pathBasename(reference);
  const dotIndex = name.lastIndexOf(".");
  return dotIndex <= 0 ? "" : name.slice(dotIndex);
}

/**
 * Reassembles one page's `getTextContent()` items into plain text — see the
 * module docstring's pypdf -> pdfjs-dist mapping notes for why this is a
 * vertical-gap heuristic rather than a literal join. `TextMarkedContent`
 * entries (no `str` field) carry no text of their own and are skipped.
 */
function assemblePageText(items: readonly unknown[]): string {
  let text = "";
  let prevY: number | null = null;
  for (const rawItem of items) {
    if (typeof rawItem !== "object" || rawItem === null || !("str" in rawItem)) {
      continue;
    }
    const item = rawItem as { str: string; height: number; transform: readonly number[] };
    const y = item.transform[5] ?? 0;
    if (prevY !== null) {
      const gap = prevY - y;
      const unit = Math.max(item.height, 1);
      if (gap > unit * PARAGRAPH_GAP_RATIO) {
        text += "\n\n";
      } else if (gap > unit * LINE_GAP_RATIO) {
        text += "\n";
      }
    }
    text += item.str;
    prevY = y;
  }
  return text;
}

/** Loads `pdfjs-dist` on demand — see the module docstring for why this
 * check cannot happen synchronously at construction the way the Python
 * twin's `ImportError` does. */
async function loadPdfjs(): Promise<typeof PdfjsLib> {
  try {
    return (await import("pdfjs-dist/legacy/build/pdf.mjs")) as typeof PdfjsLib;
  } catch (error) {
    throw new Error("PdfConnector requires pdfjs-dist — install with npm install pdfjs-dist", {
      cause: error,
    });
  }
}

/**
 * Offers every page in `emptyPages` to `adapter` for recovery (ADR 0007
 * §10), splicing back in whatever it recovers that clears the same
 * `minCharsPerPage` bar every other page's text must clear — never a page
 * this connector did not ask about, never text too thin to count as
 * recovered. Returns the (possibly updated) `pageTexts`, the pages STILL
 * unusable after the attempt, any diagnostics the adapter itself reported,
 * and — only if the adapter's `recognize` rejected — a message naming its
 * own failure. An adapter's rejection is never re-thrown: the pages it was
 * asked about are left exactly as unusable as if no adapter had been
 * configured at all, matching §10's "absent a configured adapter, the
 * `ocr_required` diagnostic is terminal for that document" for a
 * misbehaving one too.
 */
async function recoverWithOcr(
  adapter: OcrAdapter,
  options: {
    source: string;
    raw: Uint8Array;
    pageTexts: readonly string[];
    emptyPages: readonly number[];
    minCharsPerPage: number;
  },
): Promise<{
  pageTexts: string[];
  emptyPages: number[];
  diagnostics: Diagnostic[];
  adapterError: string | null;
}> {
  const { source, raw, pageTexts, emptyPages, minCharsPerPage } = options;
  const request = new OcrRequest({
    source,
    content: raw,
    contentType: "application/pdf",
    locators: emptyPages.map((page) => ({ kind: "page", value: String(page) })),
  });

  let result: OcrResult;
  try {
    result = await adapter.recognize(request);
  } catch (error) {
    // An adapter is arbitrary external code this connector does not
    // control; its own failure must not fail the whole document, only
    // leave the pages it was asked about unrecovered. `error.message`
    // (never `String(error)`, which would prefix e.g. `Error: `) mirrors
    // Python's `str(error)` for the adapter-failure message this feeds.
    const message = error instanceof Error ? error.message : String(error);
    return {
      pageTexts: [...pageTexts],
      emptyPages: [...emptyPages],
      diagnostics: [],
      adapterError: message,
    };
  }

  const requested = new Set(emptyPages);
  const recovered = new Map<number, string>();
  for (const unit of result.units) {
    if (unit.locator.kind !== "page") {
      continue;
    }
    const pageNumber = parseIntStrict(unit.locator.value);
    if (pageNumber === null || !requested.has(pageNumber) || recovered.has(pageNumber)) {
      continue;
    }
    if (pageCharCount(unit.text) < minCharsPerPage) {
      continue;
    }
    recovered.set(pageNumber, unit.text);
  }

  if (recovered.size === 0) {
    return {
      pageTexts: [...pageTexts],
      emptyPages: [...emptyPages],
      diagnostics: [...result.diagnostics],
      adapterError: null,
    };
  }

  const updatedTexts = [...pageTexts];
  for (const [pageNumber, text] of recovered) {
    updatedTexts[pageNumber - 1] = text;
  }
  const remainingEmpty = emptyPages.filter((page) => !recovered.has(page));
  return {
    pageTexts: updatedTexts,
    emptyPages: remainingEmpty,
    diagnostics: [...result.diagnostics],
    adapterError: null,
  };
}

/** One flattened outline (bookmark) entry, document order, regardless of
 * nesting depth — see the module docstring's outline-walk mapping note. */
interface FlatOutlineItem {
  title: string;
  pageIndex: number | null;
}

async function resolvePageIndex(doc: PdfjsLib.PDFDocumentProxy, dest: unknown): Promise<number | null> {
  try {
    let resolved = dest;
    if (typeof resolved === "string") {
      resolved = await doc.getDestination(resolved);
    }
    if (!Array.isArray(resolved) || resolved.length === 0) {
      return null;
    }
    const ref: unknown = resolved[0];
    if (typeof ref === "object" && ref !== null && "num" in ref && "gen" in ref) {
      return await doc.getPageIndex(ref as { num: number; gen: number });
    }
    return null;
  } catch {
    // A malformed/dangling /Dest degrades to "unplaced" — the same
    // fall-forward-friendly posture the Python twin's own
    // get_destination_page_number try/except takes.
    return null;
  }
}

async function flattenOutline(
  doc: PdfjsLib.PDFDocumentProxy,
  items: ReadonlyArray<{ title?: unknown; dest?: unknown; items?: unknown }>,
): Promise<FlatOutlineItem[]> {
  const result: FlatOutlineItem[] = [];
  for (const item of items) {
    if (typeof item.title === "string") {
      result.push({ title: item.title, pageIndex: await resolvePageIndex(doc, item.dest) });
    }
    if (Array.isArray(item.items) && item.items.length > 0) {
      result.push(...(await flattenOutline(doc, item.items)));
    }
  }
  return result;
}

function firstParagraphAtOrAfter(
  starts: ReadonlyArray<number | null>,
  pageIndex: number,
): number | null {
  for (let index = Math.max(pageIndex, 0); index < starts.length; index += 1) {
    const candidate = starts[index];
    if (candidate !== null && candidate !== undefined) {
      return candidate;
    }
  }
  return null;
}

async function documentTitle(
  doc: PdfjsLib.PDFDocumentProxy,
  outlineTitles: readonly string[],
): Promise<string | null> {
  try {
    const { info } = await doc.getMetadata();
    const title = (info as Record<string, unknown> | null)?.["Title"];
    if (typeof title === "string" && title.trim()) {
      return title.trim();
    }
  } catch {
    // A malformed /Info degrades to "no title", same as the Python twin.
  }
  if (outlineTitles.length > 0) {
    return outlineTitles[0]!.trim() || null;
  }
  return null;
}

/**
 * One `SectionEntry` per outline (bookmark) entry that resolves to a page
 * carrying at least one paragraph — at the FIRST paragraph of that page,
 * since an outline entry names a page, never a paragraph (ADR 0007 §7.4's
 * "derived from the document's own structure"). An entry whose page has no
 * usable text (scanned, failed, or simply blank) falls forward to the next
 * page that does; an entry with nothing to fall forward to is dropped,
 * silently — the same degrade-quietly posture this connector takes for
 * every other outline defect.
 */
async function outlineSections(
  doc: PdfjsLib.PDFDocumentProxy,
  pageParagraphStarts: ReadonlyArray<number | null>,
): Promise<{ sections: SectionEntry[]; title: string | null }> {
  let outlineItems: FlatOutlineItem[] = [];
  try {
    const raw = await doc.getOutline();
    outlineItems = await flattenOutline(doc, raw ?? []);
  } catch {
    // A malformed/cyclic/deeply-nested outline degrades to none, same as
    // every other parse failure here — never thrown out of this connector.
    outlineItems = [];
  }

  const entries: SectionEntry[] = [];
  const seenParagraphs = new Set<number>();
  const titles: string[] = [];
  for (const { title: rawTitle, pageIndex } of outlineItems) {
    titles.push(rawTitle);
    if (pageIndex === null) {
      continue;
    }
    const paragraph = firstParagraphAtOrAfter(pageParagraphStarts, pageIndex);
    if (paragraph === null || seenParagraphs.has(paragraph)) {
      continue;
    }
    const label = rawTitle.trim();
    if (!label || byteLen(label) > MAX_SECTION_BYTES) {
      continue;
    }
    seenParagraphs.add(paragraph);
    entries.push(new SectionEntry({ paragraph, section: label }));
  }
  entries.sort((a, b) => a.paragraph - b.paragraph);
  return { sections: entries, title: await documentTitle(doc, titles) };
}

/**
 * Reference connector for `.pdf` files (issue #348).
 *
 * `extractOutline` (default `true`) controls whether the PDF's own outline
 * (bookmarks) becomes `sections` and the document `title` — set `false` to
 * reproduce a plain paginated passage with neither. `minCharsPerPage`
 * (default 16) is the ADR 0007 §10-required, connector-documented per-page
 * text-layer threshold. `maxFileBytes` (default 64 MiB) caps the raw file
 * size checked before any parsing. `ocrAdapter` (default `null`,
 * `OcrAdapter` from ocr.ts) is the external OCR engine boundary §10
 * defines: when configured, every page this connector itself finds
 * unusable is offered to it for recovery before `ocr_required` is decided;
 * when absent, behavior is unchanged from before this connector could
 * accept one at all.
 */
export class PdfConnector implements Connector {
  readonly parser: string = PARSER_NAME;
  readonly parserVersion: string = PARSER_VERSION;

  private readonly extractOutlineFlag: boolean;
  private readonly minCharsPerPage: number;
  private readonly maxFileBytes: number;
  private readonly ocrAdapter: OcrAdapter | null;

  constructor(options?: {
    extractOutline?: boolean;
    minCharsPerPage?: number;
    maxFileBytes?: number;
    ocrAdapter?: OcrAdapter | null;
  }) {
    this.extractOutlineFlag = options?.extractOutline ?? true;
    this.minCharsPerPage = options?.minCharsPerPage ?? DEFAULT_MIN_CHARS_PER_PAGE;
    this.maxFileBytes = options?.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
    this.ocrAdapter = options?.ocrAdapter ?? null;
  }

  /**
   * The digest `read` would stamp into `fingerprintInputs.parseOptionsDigest`
   * for THIS instance's effective config, computable without reading
   * anything (ADR 0007 §6.3) — and, unlike `read`, never needs
   * `pdfjs-dist` at all.
   */
  async parseOptionsDigest(): Promise<string> {
    return this.optionsDigestValue();
  }

  supports(reference: string): boolean {
    return SUPPORTED_SUFFIXES.has(pathSuffix(reference).toLowerCase());
  }

  private async optionsDigestValue(): Promise<string> {
    return optionsDigest({
      extract_outline: this.extractOutlineFlag,
      min_chars_per_page: this.minCharsPerPage,
      max_file_bytes: this.maxFileBytes,
      // Not the adapter object itself (not JSON-digestible, and not
      // something two adapter instances should ever need to be identical
      // BY VALUE to compare equal) — its declared identity, so
      // configuring, swapping, or removing an adapter changes this
      // connector's effective fingerprint and §6.3's checkpoint knows to
      // re-parse rather than skip.
      ocr_adapter: this.ocrAdapter === null ? null : `${this.ocrAdapter.name}@${this.ocrAdapter.version}`,
    });
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
        contentType: "application/pdf",
      }),
      fingerprintInputs: await this.fingerprint(fields.rawContentSha256),
      diagnostics: [new Diagnostic({ code: fields.code, message: fields.message, source: fields.source })],
    });
  }

  async read(reference: string): Promise<ConnectorDocument> {
    // As in a plain text connector: `reference` itself is the source id
    // (ADR 0007 §6.1); it is only ever used for filesystem access below,
    // never for identity.
    const source = fileSourceId(reference);
    const displayName = pathBasename(reference);

    const failure = (
      code: DiagnosticCode,
      message: string,
      rawContentSha256: string = EMPTY_SHA256,
    ): Promise<ConnectorDocument> =>
      this.failure({ source, displayName, code, message, rawContentSha256 });

    const sourceDiagnostic = checkSourceId(source);
    if (sourceDiagnostic !== null) {
      return failure(sourceDiagnostic.code, sourceDiagnostic.message);
    }

    if (!this.supports(reference)) {
      return failure(
        "unsupported_format",
        `unsupported extension ${JSON.stringify(pathSuffix(reference))} (only .pdf)`,
      );
    }

    const { readFile, stat } = await import("node:fs/promises");

    let size: number;
    try {
      size = (await stat(reference)).size;
    } catch (error) {
      return failure("unreadable", String(error));
    }
    if (size > this.maxFileBytes) {
      return failure(
        "content_too_large",
        `${size} bytes exceeds the ${this.maxFileBytes}-byte file cap`,
      );
    }

    let raw: Uint8Array;
    try {
      raw = new Uint8Array(await readFile(reference));
    } catch (error) {
      return failure("unreadable", String(error));
    }
    // The raw bytes, before parsing — ADR 0007 §5's connector-native
    // checkpoint fingerprint, deliberately distinct from any hash of the
    // extracted text (§6.2).
    const rawContentSha256 = await sha256HexBytes(raw);

    const pdfjsLib = await loadPdfjs();

    // pdfjs-dist takes ownership of (and detaches) a TypedArray `data`
    // buffer it is given — `raw.slice()` hands it a throwaway copy so
    // `raw` itself stays readable afterward (it is still needed below,
    // e.g. as an `OcrRequest.content` payload).
    const loadingTask = pdfjsLib.getDocument({
      data: raw.slice(),
      verbosity: pdfjsLib.VerbosityLevel.ERRORS,
    });
    let doc: PdfjsLib.PDFDocumentProxy;
    try {
      doc = await loadingTask.promise;
    } catch (error) {
      await loadingTask.destroy().catch(() => {});
      if (isPasswordException(error)) {
        return failure(
          "encrypted",
          "the document requires a password this connector does not have",
          rawContentSha256,
        );
      }
      return failure("corrupt", String(error), rawContentSha256);
    }

    try {
      return await this.readOpenDocument(doc, {
        source,
        displayName,
        raw,
        rawContentSha256,
        failure,
      });
    } finally {
      // The loading task keeps transport, page, and font caches alive
      // until destroy() (pdfjs-dist v6 puts destroy on the task, not the
      // document proxy) — without this, a long syncReferences/
      // syncObjectStorage run over many PDFs would accumulate them all
      // across every early-return path.
      await loadingTask.destroy().catch(() => {});
    }
  }

  /** The body of `read` past document opening — split out so `read` can
   * `finally`-destroy the `PDFDocumentProxy` across every return path. */
  private async readOpenDocument(
    doc: PdfjsLib.PDFDocumentProxy,
    context: {
      source: string;
      displayName: string;
      raw: Uint8Array;
      rawContentSha256: string;
      failure: (
        code: DiagnosticCode,
        message: string,
        rawContentSha256?: string,
      ) => Promise<ConnectorDocument>;
    },
  ): Promise<ConnectorDocument> {
    const { source, displayName, raw, rawContentSha256, failure } = context;
    const pageCount = doc.numPages;
    if (pageCount === 0) {
      return failure("corrupt", "the document has no pages", rawContentSha256);
    }

    let pageTexts: string[] = [];
    let failedPages: number[] = [];
    for (let index = 0; index < pageCount; index += 1) {
      try {
        const page = await doc.getPage(index + 1);
        const content = await page.getTextContent();
        pageTexts.push(assemblePageText(content.items));
      } catch {
        // One page's own decode failure must not fail the whole document;
        // it is reported as partial_extraction below instead.
        failedPages.push(index + 1);
        pageTexts.push("");
      }
    }

    // Set lookups keep the per-page skip checks O(1) — Array.includes here
    // would make a mostly-unusable large PDF quadratic.
    const failedPageSet = new Set(failedPages);
    let emptyPages: number[] = [];
    for (let index = 0; index < pageCount; index += 1) {
      const pageNumber = index + 1;
      if (failedPageSet.has(pageNumber)) {
        continue;
      }
      if (pageCharCount(pageTexts[index]!) < this.minCharsPerPage) {
        emptyPages.push(pageNumber);
      }
    }

    let ocrDiagnostics: Diagnostic[] = [];
    let ocrAdapterError: string | null = null;
    if (this.ocrAdapter !== null) {
      // A page whose text extraction THREW is exactly as unusable to this
      // connector as one that decoded to nothing — both are offered to a
      // configured adapter together, in page order, so it gets one chance
      // to recover every page this connector itself could not read, not
      // only the ones that merely decoded empty.
      const recoveryCandidates = [...new Set([...emptyPages, ...failedPages])].sort(
        (a, b) => a - b,
      );
      if (recoveryCandidates.length > 0) {
        const recovered = await recoverWithOcr(this.ocrAdapter, {
          source,
          raw,
          pageTexts,
          emptyPages: recoveryCandidates,
          minCharsPerPage: this.minCharsPerPage,
        });
        pageTexts = recovered.pageTexts;
        const unrecovered = new Set(recovered.emptyPages);
        const recoveredPages = new Set(recoveryCandidates.filter((page) => !unrecovered.has(page)));
        // A recovered page stops counting toward whichever diagnostic
        // (ocr_required, or corrupt/partial_extraction) it would otherwise
        // have produced; a page still unrecovered keeps its original
        // classification, unchanged from before recovery was attempted.
        emptyPages = emptyPages.filter((page) => !recoveredPages.has(page));
        failedPages = failedPages.filter((page) => !recoveredPages.has(page));
        ocrDiagnostics = recovered.diagnostics;
        ocrAdapterError = recovered.adapterError;
      }
    }

    const paragraphs: string[] = [];
    const locators: LocatorEntry[] = [];
    const pageParagraphStarts: Array<number | null> = new Array(pageCount).fill(null);
    // Built after OCR recovery, which may have shrunk both lists.
    const emptyPageSet = new Set(emptyPages);
    const unusablePageSet = new Set(failedPages);
    for (let index = 0; index < pageCount; index += 1) {
      const pageNumber = index + 1;
      if (unusablePageSet.has(pageNumber) || emptyPageSet.has(pageNumber)) {
        continue;
      }
      const pageParagraphs = splitParagraphs(pageTexts[index]!);
      if (pageParagraphs.length === 0) {
        continue;
      }
      pageParagraphStarts[index] = paragraphs.length;
      for (const paragraph of pageParagraphs) {
        locators.push(
          new LocatorEntry({
            paragraph: paragraphs.length,
            locator: { kind: "page", value: String(pageNumber) },
          }),
        );
        paragraphs.push(paragraph);
      }
    }

    if (paragraphs.length === 0) {
      if (emptyPages.length > 0) {
        const unusable = Array.from(new Set([...emptyPages, ...failedPages])).sort((a, b) => a - b);
        return failure(
          "ocr_required",
          ocrRequiredMessage(unusable, ocrAdapterError),
          rawContentSha256,
        );
      }
      return failure(
        "corrupt",
        `failed to extract text on page(s) ${listPages(failedPages)}`,
        rawContentSha256,
      );
    }

    const text = paragraphs.join("\n\n");
    if (byteLen(text) > MAX_PASSAGE_BYTES) {
      return failure(
        "content_too_large",
        `${byteLen(text)} bytes of extracted text exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`,
        rawContentSha256,
      );
    }

    let sections: SectionEntry[] = [];
    let title: string | null = null;
    if (this.extractOutlineFlag) {
      const outline = await outlineSections(doc, pageParagraphStarts);
      sections = outline.sections;
      title = outline.title;
    }

    const diagnostics: Diagnostic[] = [];
    if (emptyPages.length > 0) {
      diagnostics.push(
        new Diagnostic({
          code: "ocr_required",
          message: ocrRequiredMessage(emptyPages, ocrAdapterError),
          source,
        }),
      );
    }
    diagnostics.push(...ocrDiagnostics);
    if (failedPages.length > 0) {
      diagnostics.push(
        new Diagnostic({
          code: "partial_extraction",
          message: `failed to extract text on page(s) ${listPages(failedPages)}`,
          source,
        }),
      );
    }

    return new ConnectorDocument({
      source,
      text,
      locators,
      sections,
      metadata: new ConnectorMetadata({
        originUri: source,
        displayName,
        title,
        contentType: "application/pdf",
      }),
      fingerprintInputs: await this.fingerprint(rawContentSha256),
      diagnostics,
    });
  }
}
