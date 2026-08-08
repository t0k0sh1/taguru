/**
 * The DOCX connector (ADR 0007 §7/§8, issue #350): `.docx` files into
 * {@link ConnectorDocument}, preserving paragraphs, heading hierarchy, and
 * tables. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.docx` module (issue #415).
 *
 * The Python twin parses through `python-docx`'s object model
 * (`document.iter_inner_content()`, `paragraph.style`, `table.rows`, …).
 * Node has no equivalent OOXML library, so this port reads the same OOXML
 * parts DIRECTLY — `word/document.xml` and `word/styles.xml` — through two
 * optional dependencies loaded ONLY via dynamic `import()` inside `read()`
 * (never at module top level, so a caller who never ingests DOCX never
 * pays for them): `fflate` (`unzipSync`, the zip container) and
 * `fast-xml-parser` (`XMLParser` with `preserveOrder: true`, so element
 * order — paragraphs vs tables vs runs — survives exactly as OOXML wrote
 * it, plus `XMLValidator` to reject malformed XML `XMLParser.parse` itself
 * would otherwise silently mis-parse rather than throw on).
 *
 * The document body is walked over `word/document.xml`'s own `w:body`
 * children in document order — the same one traversal
 * `document.iter_inner_content()` gives the Python twin, since a `w:body`
 * child is always exactly one `w:p` or one `w:tbl` (`w:sectPr`, the body's
 * final child, is skipped by construction: it matches neither tag).
 * Headers and footers are never visited: they are separate parts, not
 * children of `w:body`, so nothing needs to filter them out.
 *
 * A heading is a paragraph whose style name matches `Heading N` (a real
 * Word-authored `.docx` names its own built-in style `heading N`,
 * lowercase — this connector's own regex is case-insensitive precisely to
 * still recognize it without `python-docx`'s `BabelFish` UI-name
 * translation table, which this port has no equivalent of) — or, when a
 * document uses a custom style with no such name at all, a paragraph
 * carrying its own `w:pPr/w:outlineLvl` (0-based; mapped to 1-based levels
 * the same way Word's own UI numbers them). Each heading is its own
 * paragraph in `text`, and its breadcrumb (`"Guide > Installation"`, since
 * `sections` is flat — the same convention `html.py`'s TS twin already
 * uses) becomes a {@link SectionEntry}.
 *
 * A table (top-level or nested) becomes exactly ONE paragraph — its rows
 * joined with `\n`, cells with `" | "` — carrying a
 * `{"kind": "table", "value": "N"}` locator, `N` the table's 1-based
 * position among its document-order siblings (a nested table gets a
 * dotted path, `"3.1"`, under its parent's own number). This is the
 * opposite trade-off from `html.ts`'s `fragment` locator (attached to
 * every ordinary paragraph so a citation can reach a real in-page anchor):
 * a DOCX has no page boundary of its own to spend that
 * one-locator-per-paragraph budget on (ADR 0007 §7.2), so this connector
 * spends it on tables instead — an ordinary body paragraph never carries a
 * locator, which is exactly what makes "this paragraph has a locator"
 * mean "this paragraph is a table" for a citation reading this document's
 * `locators`.
 *
 * No OCR engine, no rasterizer: a document left with no extractable text
 * after the walk (e.g. an image-only DOCX) degrades to `ocr_required` with
 * empty text. Encrypted (password-protected) DOCX files use a different
 * container entirely — MS-OFFCRYPTO wraps the whole package in an
 * OLE2/CFB structure, not a plain zip — so this connector recognizes that
 * container's own magic bytes and reports `encrypted` before ever handing
 * the bytes to the zip/XML parsers, rather than letting them fail as
 * `corrupt`. A *restricted-editing* password (`w:documentProtection`,
 * stored in `word/settings.xml`) is a different, weaker mechanism that
 * leaves the package a normal, readable zip; this connector does not —
 * and should not — treat it as `encrypted`.
 *
 * Footnote/endnote text, comment text, and text-box content are none of
 * them reachable through a paragraph's own run text (a footnote/endnote/
 * comment reference is a marker inside a run, never the referenced text
 * itself — which lives in a separate part this connector never reads —
 * and text-box content lives in a nested `w:txbxContent` a paragraph's own
 * run walk never descends into) — a document whose body references any of
 * them is named in a single `partial_extraction` diagnostic rather than
 * silently short-changed. Nested `<table>`-in-`<table>` content the reader
 * cannot express any other way (a merged cell spanning columns, for
 * instance) is read the way row-major text naturally reads it: this port
 * additionally simplifies `docx.py`'s own `row.cells` (which repeats a
 * horizontally-merged cell across every grid column it spans) to "one
 * `w:tc` per `w:tr` child, in document order" — no `w:gridSpan`/`w:vMerge`
 * resolution — since no ported test exercises a merged cell; see the
 * report for this deviation.
 *
 * `.doc` (legacy binary) and `.docm` (macro-enabled) are both
 * `unsupported_format` — only the plain `.docx` OOXML format is read.
 */

import type { Locator } from "taguru";

import { MAX_PASSAGE_BYTES } from "../extract.js";
import type { Connector } from "./protocol.js";
import { byteLen, fitBreadcrumb, sanitizeParagraphText } from "./structure.js";
import { checkSourceId, fileSourceId } from "./sources.js";
import {
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  type DiagnosticCode,
  FingerprintInputs,
  LocatorEntry,
  MAX_SECTION_BYTES,
  optionsDigest,
  SectionEntry,
} from "./document.js";

export const PARSER_NAME = "taguru-docx-connector";
export const PARSER_VERSION = "1.0.0";

const SUPPORTED_SUFFIXES: ReadonlySet<string> = new Set([".docx"]);
const CONTENT_TYPE =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

// The raw-file cap, checked before any parsing — same two-cap shape
// html.ts's TS twin uses for the same reason: markup/XML overhead means a
// large-but-sparse document can stay well under the extracted-text cap,
// and a heavily-styled small document's raw bytes can still exceed a
// naive per-file cap.
const DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024;

// OOXML is XML and compresses extremely well: a package under the raw
// file cap above can still DECLARE tens of GiB of decompressed entries (a
// zip bomb). The central directory's declared sizes are summed BEFORE any
// entry is inflated, and a package past this cap is refused with a
// `content_too_large` diagnostic — the cap and the diagnostic are kept
// identical between docx.ts and pptx.ts.
const MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024;

class DecompressedSizeExceededError extends Error {}

function unzipBounded(
  unzipSync: OoxmlDeps["unzipSync"],
  raw: Uint8Array,
): Record<string, Uint8Array> {
  let total = 0;
  return unzipSync(raw, {
    filter: (file) => {
      total += file.originalSize;
      if (total > MAX_DECOMPRESSED_BYTES) {
        throw new DecompressedSizeExceededError(
          `the package declares more than ${MAX_DECOMPRESSED_BYTES} decompressed bytes`,
        );
      }
      return true;
    },
  });
}

// MS-OFFCRYPTO password-protected Office documents (including .docx) are
// re-packaged as a whole OLE2/Compound File Binary container, not a zip —
// this is that container format's own magic number (the first 8 bytes of
// every CFB file, regardless of what it holds). Checked BEFORE any
// zip/XML parsing is attempted, so a password-protected file is reported
// `encrypted` rather than falling through to the zip parser's own
// open failure and being misreported as `corrupt`.
const OLE2_MAGIC = Uint8Array.from([
  0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1,
]);

// A real Word-authored `.docx` names its Heading-N built-in style
// `heading N`, lowercase (confirmed against real `python-docx` writer
// output) — this matches that name case-insensitively (a defensive margin
// for a custom style authored directly under a differently-cased name).
// `[1-9]` alone is enough: `w:outlineLvl` (the fallback below) tops out at
// 8 (0-based), i.e. heading level 9.
const HEADING_STYLE_RE = /^heading\s*([1-9])$/i;

// Content this connector's own paragraph walk cannot reach at all — named
// in a single `partial_extraction` diagnostic rather than silently
// omitted. Every marker is a literal tag search over `word/document.xml`'s
// decoded text (never a full second parse): a footnote/endnote/comment
// REFERENCE lives in the body even though its own text lives in a
// separate part this connector never reads, and a text box's content
// lives in a nested `w:txbxContent` a paragraph's own run walk never
// descends into.
const PARTIAL_CONTENT_MARKERS: ReadonlyArray<readonly [string, string]> = [
  ["<w:footnoteReference", "footnote"],
  ["<w:endnoteReference", "endnote"],
  ["<w:commentReference", "comment"],
  ["<w:txbxContent", "text box"],
];

// -- minimal preserveOrder XML node helpers (shared shape with pptx.ts, --
// -- duplicated rather than imported: docx.py/pptx.py duplicate their own
// -- OLE2_MAGIC/PARTIAL_CONTENT_MARKERS the same way, and neither this
// -- module nor pptx.ts is meant to depend on the other) -------------------

/** One `fast-xml-parser` `preserveOrder` node: exactly one tag-name key
 * mapping to its children (or `#text` mapping to a text value), plus an
 * optional `:@` key holding its attributes (`@_`-prefixed). */
type XmlNode = Record<string, unknown>;

function isXmlRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function xmlTagName(node: XmlNode): string | null {
  for (const key of Object.keys(node)) {
    if (key !== ":@") {
      return key;
    }
  }
  return null;
}

function xmlChildren(node: XmlNode): XmlNode[] {
  const tag = xmlTagName(node);
  if (tag === null) {
    return [];
  }
  const value = node[tag];
  return Array.isArray(value) ? (value as XmlNode[]) : [];
}

function xmlAttrs(node: XmlNode): Record<string, string> {
  const attrs = node[":@"];
  return isXmlRecord(attrs) ? (attrs as Record<string, string>) : {};
}

function xmlChildrenNamed(node: XmlNode, tag: string): XmlNode[] {
  return xmlChildren(node).filter((child) => xmlTagName(child) === tag);
}

function xmlFirstChildNamed(node: XmlNode, tag: string): XmlNode | null {
  return xmlChildren(node).find((child) => xmlTagName(child) === tag) ?? null;
}

/** A `w:t`-shaped node's own text content ("" for a self-closing/empty
 * element) — mirrors the `.text` python-docx gives a `w:t` element. */
function xmlTextOf(node: XmlNode): string {
  const textNode = xmlChildren(node).find((child) => "#text" in child);
  return textNode ? String((textNode as Record<string, unknown>)["#text"]) : "";
}

function decodeUtf8(bytes: Uint8Array): string {
  return new TextDecoder("utf-8").decode(bytes);
}

async function sha256HexBytes(data: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data as BufferSource);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

function startsWithOle2Magic(bytes: Uint8Array): boolean {
  if (bytes.length < OLE2_MAGIC.length) {
    return false;
  }
  for (let index = 0; index < OLE2_MAGIC.length; index += 1) {
    if (bytes[index] !== OLE2_MAGIC[index]) {
      return false;
    }
  }
  return true;
}

/** `Path(reference).suffix.lower()` mirror, without a `node:path` import
 * (this must stay callable from the synchronous `supports()` method). */
function suffixOf(reference: string): string {
  const lastSlash = Math.max(
    reference.lastIndexOf("/"),
    reference.lastIndexOf("\\"),
  );
  const base = lastSlash >= 0 ? reference.slice(lastSlash + 1) : reference;
  const dotIndex = base.lastIndexOf(".");
  return dotIndex > 0 ? base.slice(dotIndex).toLowerCase() : "";
}

/** `Path(reference).name` mirror, same `node:path`-free constraint. */
function basenameOf(reference: string): string {
  const lastSlash = Math.max(
    reference.lastIndexOf("/"),
    reference.lastIndexOf("\\"),
  );
  return lastSlash >= 0 ? reference.slice(lastSlash + 1) : reference;
}

interface OoxmlDeps {
  unzipSync: typeof import("fflate").unzipSync;
  XMLParser: typeof import("fast-xml-parser").XMLParser;
  XMLValidator: typeof import("fast-xml-parser").XMLValidator;
}

/**
 * Loads the two optional OOXML deps ONLY when a `.docx` is actually about
 * to be parsed — never at module top level, so a caller who never
 * constructs (or never calls `read` on) a `DocxConnector` never pays their
 * import cost. Throws the same clear, install-instruction-bearing error
 * the Python twin's constructor raises for a missing `python-docx` (this
 * port's own gate happens inside `read` instead: a TS constructor cannot
 * `await` a dynamic `import()`, so it cannot check synchronously).
 */
async function loadOoxmlDeps(): Promise<OoxmlDeps> {
  try {
    const [{ unzipSync }, { XMLParser, XMLValidator }] = await Promise.all([
      import("fflate"),
      import("fast-xml-parser"),
    ]);
    return { unzipSync, XMLParser, XMLValidator };
  } catch (error) {
    throw new Error(
      "DocxConnector requires fflate and fast-xml-parser — install with " +
        "npm install fflate fast-xml-parser",
      { cause: error },
    );
  }
}

function newXmlParser(
  deps: OoxmlDeps,
): InstanceType<typeof import("fast-xml-parser").XMLParser> {
  return new deps.XMLParser({
    preserveOrder: true,
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    trimValues: false,
    ignoreDeclaration: true,
  });
}

/**
 * Parses `text` (one OOXML part) into its `preserveOrder` node array,
 * throwing (never silently mis-parsing) when `text` isn't well-formed XML
 * — `XMLParser.parse` alone is lenient enough to need `XMLValidator`
 * checked first, mirroring the eager, all-parts-parsed-up-front loading
 * every `python-docx`-based tool gets from `lxml` for free.
 */
function parseXmlPart(
  deps: OoxmlDeps,
  text: string,
  partName: string,
): XmlNode[] {
  const validation = deps.XMLValidator.validate(text);
  if (validation !== true) {
    throw new Error(`malformed XML in ${partName}: ${validation.err.msg}`);
  }
  return newXmlParser(deps).parse(text) as XmlNode[];
}

// -- run/paragraph text --------------------------------------------------

/** A `w:r` node's own text: `w:t` verbatim, `w:tab` as `"\t"`, `w:br`/
 * `w:cr` as `"\n"` — mirrors `python-docx`'s `Run.text`. Only the run's
 * OWN direct children are visited, never a nested `w:drawing`/
 * `w:txbxContent`'s content — the same "not reachable through a
 * paragraph's own text" posture this module's docstring documents for
 * text-box content. */
function runText(run: XmlNode): string {
  let text = "";
  for (const child of xmlChildren(run)) {
    const tag = xmlTagName(child);
    if (tag === "w:t") {
      text += xmlTextOf(child);
    } else if (tag === "w:tab") {
      text += "\t";
    } else if (tag === "w:br" || tag === "w:cr") {
      text += "\n";
    }
  }
  return text;
}

/** A `w:p` node's own text: its direct `w:r` children's `runText`,
 * concatenated — mirrors `python-docx`'s `Paragraph.text` (`self.runs`,
 * i.e. direct-child runs only, never a run nested inside a `w:hyperlink`
 * or other wrapper). */
function paragraphText(paragraph: XmlNode): string {
  return xmlChildrenNamed(paragraph, "w:r").map(runText).join("");
}

function paragraphStyleId(paragraph: XmlNode): string | null {
  const pPr = xmlFirstChildNamed(paragraph, "w:pPr");
  if (pPr === null) {
    return null;
  }
  const pStyle = xmlFirstChildNamed(pPr, "w:pStyle");
  if (pStyle === null) {
    return null;
  }
  return xmlAttrs(pStyle)["@_w:val"] ?? null;
}

function paragraphOutlineLevel(paragraph: XmlNode): number | null {
  const pPr = xmlFirstChildNamed(paragraph, "w:pPr");
  if (pPr === null) {
    return null;
  }
  const outline = xmlFirstChildNamed(pPr, "w:outlineLvl");
  if (outline === null) {
    return null;
  }
  const val = xmlAttrs(outline)["@_w:val"];
  if (val === undefined) {
    return null;
  }
  const parsed = Number.parseInt(val, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

/**
 * The 1-based heading level `paragraph` represents, or `null` if it isn't
 * a heading — style name first, `w:outlineLvl` fallback, `null` on any
 * lookup failure (a missing style reference degrades to "not a heading",
 * mirroring `python-docx`'s own fallback to its "Normal" style for a
 * `w:pStyle` referencing an undefined style).
 */
function headingLevel(
  paragraph: XmlNode,
  styleNames: ReadonlyMap<string, string>,
): number | null {
  const styleId = paragraphStyleId(paragraph);
  if (styleId !== null) {
    const styleName = styleNames.get(styleId);
    if (styleName) {
      const match = HEADING_STYLE_RE.exec(styleName.trim());
      if (match) {
        return Number.parseInt(match[1]!, 10);
      }
    }
  }
  const outline = paragraphOutlineLevel(paragraph);
  // ECMA-376: `w:outlineLvl` values 0-8 are outline levels 1-9; the value
  // 9 explicitly means "body text, no outline level" and must not read as
  // a level-10 heading.
  if (outline !== null && outline >= 0 && outline <= 8) {
    return outline + 1;
  }
  return null;
}

/** `word/styles.xml`'s paragraph styles, keyed by `w:styleId` to
 * `w:name/@w:val` — the lookup `headingLevel` uses in place of
 * `python-docx`'s `paragraph.style.name`. */
function parseStyleNames(
  deps: OoxmlDeps,
  stylesXmlText: string,
): ReadonlyMap<string, string> {
  const map = new Map<string, string>();
  const root = parseXmlPart(deps, stylesXmlText, "word/styles.xml");
  const stylesNode = root.find((node) => xmlTagName(node) === "w:styles");
  if (!stylesNode) {
    return map;
  }
  for (const style of xmlChildrenNamed(stylesNode, "w:style")) {
    const styleId = xmlAttrs(style)["@_w:styleId"];
    const nameNode = xmlFirstChildNamed(style, "w:name");
    const name = nameNode ? xmlAttrs(nameNode)["@_w:val"] : undefined;
    if (styleId && name) {
      map.set(styleId, name);
    }
  }
  return map;
}

function readCoreTitle(
  deps: OoxmlDeps,
  coreXmlBytes: Uint8Array | undefined,
): string | null {
  if (!coreXmlBytes) {
    return null;
  }
  const root = parseXmlPart(
    deps,
    decodeUtf8(coreXmlBytes),
    "docProps/core.xml",
  );
  const coreNode = root.find(
    (node) => xmlTagName(node) === "cp:coreProperties",
  );
  if (!coreNode) {
    return null;
  }
  const titleNode = xmlFirstChildNamed(coreNode, "dc:title");
  if (!titleNode) {
    return null;
  }
  const title = xmlTextOf(titleNode).trim();
  return title ? title : null;
}

// -- body walk -------------------------------------------------------------

/** The one pass over `word/document.xml`'s `w:body` builds all three at
 * once — mirrors `docx.py`'s own `_BodyResult`, one paragraph-indexed pass
 * producing `paragraphs` plus paragraph-indexed `locators`/`sections`
 * together, so they can never drift out of sync with each other. */
interface BodyResult {
  paragraphs: string[];
  locators: LocatorEntry[];
  sections: SectionEntry[];
  firstHeading: string | null;
}

function cellText(cell: XmlNode): string {
  return xmlChildrenNamed(cell, "w:p").map(paragraphText).join("\n");
}

function buildBody(
  documentRoot: readonly XmlNode[],
  styleNames: ReadonlyMap<string, string>,
  options: {
    extractHeadings: boolean;
    headingSeparator: string;
    extractTables: boolean;
  },
): BodyResult {
  const result: BodyResult = {
    paragraphs: [],
    locators: [],
    sections: [],
    firstHeading: null,
  };
  const headingStack: Array<[number, string]> = [];

  const commit = (text: string): number | null => {
    const clean = sanitizeParagraphText(text);
    if (!clean) {
      return null;
    }
    const index = result.paragraphs.length;
    result.paragraphs.push(clean);
    return index;
  };

  const handleHeading = (paragraph: XmlNode, level: number): void => {
    const label = paragraphText(paragraph).trim();
    const index = commit(paragraphText(paragraph));
    if (index === null) {
      return;
    }
    if (result.firstHeading === null) {
      result.firstHeading = label;
    }
    if (!options.extractHeadings) {
      return;
    }
    while (
      headingStack.length > 0 &&
      headingStack[headingStack.length - 1]![0] >= level
    ) {
      headingStack.pop();
    }
    headingStack.push([level, label]);
    const breadcrumb = fitBreadcrumb(
      headingStack.map(([, crumb]) => crumb),
      { separator: options.headingSeparator, maxBytes: MAX_SECTION_BYTES },
    );
    if (breadcrumb !== null) {
      result.sections.push(
        new SectionEntry({ paragraph: index, section: breadcrumb }),
      );
    }
  };

  // Nested tables (a table inside one of this table's own cells) each
  // become their own separate paragraph+locator, dotted under this
  // table's own path — never inlined into this table's one paragraph of
  // text (§7.3: a table's position must stay distinguishable from its
  // container's). Numbered across the WHOLE parent table, never reset per
  // cell (a numbering-bug regression the Python twin's own tests guard
  // against). A horizontally-merged cell is NOT specially resolved here
  // (see this module's own docstring): each `w:tc` child of a `w:tr` is
  // read once, in document order.
  const handleTable = (table: XmlNode, path: readonly number[]): void => {
    const rows = xmlChildrenNamed(table, "w:tr");
    const rowsText: string[] = [];
    for (const row of rows) {
      const cells = xmlChildrenNamed(row, "w:tc").map(cellText);
      if (cells.some((cell) => cell.trim() !== "")) {
        rowsText.push(cells.join(" | "));
      }
    }
    if (rowsText.length > 0) {
      const index = commit(rowsText.join("\n"));
      if (index !== null && options.extractTables) {
        const locator: Locator = { kind: "table", value: path.join(".") };
        result.locators.push(new LocatorEntry({ paragraph: index, locator }));
      }
    }

    let nestedIndex = 0;
    for (const row of rows) {
      for (const cell of xmlChildrenNamed(row, "w:tc")) {
        for (const child of xmlChildren(cell)) {
          if (xmlTagName(child) === "w:tbl") {
            nestedIndex += 1;
            handleTable(child, [...path, nestedIndex]);
          }
        }
      }
    }
  };

  const documentNode = documentRoot.find(
    (node) => xmlTagName(node) === "w:document",
  );
  const bodyNode = documentNode
    ? xmlFirstChildNamed(documentNode, "w:body")
    : null;
  let tableCounter = 0;
  if (bodyNode !== null) {
    for (const item of xmlChildren(bodyNode)) {
      const tag = xmlTagName(item);
      if (tag === "w:p") {
        const level = headingLevel(item, styleNames);
        if (level !== null) {
          handleHeading(item, level);
        } else {
          commit(paragraphText(item));
        }
      } else if (tag === "w:tbl") {
        tableCounter += 1;
        handleTable(item, [tableCounter]);
      }
    }
  }

  return result;
}

function partialExtractionMessage(documentXmlText: string): string | null {
  const kinds = PARTIAL_CONTENT_MARKERS.filter(([marker]) =>
    documentXmlText.includes(marker),
  ).map(([, name]) => name);
  if (kinds.length === 0) {
    return null;
  }
  return `${kinds.join(", ")} content is present but not extracted by this connector`;
}

// -- connector ---------------------------------------------------------------

export interface DocxConnectorOptions {
  /** Whether a detected heading paragraph's breadcrumb becomes a
   * {@link SectionEntry} — the heading's own text is always kept as an
   * ordinary paragraph in `text` either way. Default `true`. */
  extractHeadings?: boolean;
  /** Joins a breadcrumb's levels. Default `" > "`. */
  headingSeparator?: string;
  /** Whether a table paragraph carries its `{"kind": "table"}` locator —
   * the table's row/cell text is always extracted as its own paragraph
   * either way, same posture. Default `true`. */
  extractTables?: boolean;
  /** Caps the raw file size checked before any parsing. Default 64 MiB. */
  maxFileBytes?: number;
}

/** Reference connector for `.docx` files (issue #350). */
export class DocxConnector implements Connector {
  readonly parser: string = PARSER_NAME;
  readonly parserVersion: string = PARSER_VERSION;

  private readonly extractHeadings: boolean;
  private readonly headingSeparator: string;
  private readonly extractTables: boolean;
  private readonly maxFileBytes: number;

  constructor(options: DocxConnectorOptions = {}) {
    this.extractHeadings = options.extractHeadings ?? true;
    this.headingSeparator = options.headingSeparator ?? " > ";
    this.extractTables = options.extractTables ?? true;
    this.maxFileBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
  }

  supports(reference: string): boolean {
    return SUPPORTED_SUFFIXES.has(suffixOf(reference));
  }

  private async optionsDigestValue(): Promise<string> {
    return optionsDigest({
      extract_headings: this.extractHeadings,
      heading_separator: this.headingSeparator,
      extract_tables: this.extractTables,
      max_file_bytes: this.maxFileBytes,
    });
  }

  /**
   * The digest `read` would stamp into
   * `fingerprint_inputs.parse_options_digest` for THIS instance's
   * effective config, computable without reading anything (ADR 0007
   * §6.3).
   */
  async parseOptionsDigest(): Promise<string> {
    return this.optionsDigestValue();
  }

  private async fingerprint(
    rawContentSha256: string,
  ): Promise<FingerprintInputs> {
    return new FingerprintInputs({
      rawContentSha256,
      parser: this.parser,
      parserVersion: this.parserVersion,
      parseOptionsDigest: await this.optionsDigestValue(),
    });
  }

  private async failureDocument(fields: {
    source: string;
    displayName: string;
    code: DiagnosticCode;
    message: string;
    rawContentSha256: string;
    contentType: string | null;
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
      diagnostics: [
        new Diagnostic({
          code: fields.code,
          message: fields.message,
          source: fields.source,
        }),
      ],
    });
  }

  async read(reference: string): Promise<ConnectorDocument> {
    // Loaded first, unconditionally — the closest a `read`-time dynamic
    // import can get to the Python twin's construction-time `ImportError`
    // (a TS constructor cannot `await`, so it cannot gate synchronously).
    const deps = await loadOoxmlDeps();

    // As in every other file connector: `reference` itself is the source
    // id (ADR 0007 §6.1); path helpers are only ever used for filesystem
    // access below, never for identity.
    const source = fileSourceId(reference);
    const displayName = basenameOf(reference);
    const emptySha256 = await sha256HexBytes(new Uint8Array(0));

    const failure = (
      code: DiagnosticCode,
      message: string,
      rawContentSha256: string = emptySha256,
      contentType: string | null = CONTENT_TYPE,
    ): Promise<ConnectorDocument> =>
      this.failureDocument({
        source,
        displayName,
        code,
        message,
        rawContentSha256,
        contentType,
      });

    const sourceDiagnostic = checkSourceId(source);
    if (sourceDiagnostic !== null) {
      return failure(sourceDiagnostic.code, sourceDiagnostic.message);
    }

    if (!this.supports(reference)) {
      // Unlike every other failure below, the extension mismatch here is
      // affirmative evidence the file is NOT a DOCX — content_type stays
      // unclaimed (null) rather than asserting a MIME type this connector
      // has no basis for.
      return failure(
        "unsupported_format",
        `unsupported extension ${JSON.stringify(suffixOf(reference))} (only .docx)`,
        emptySha256,
        null,
      );
    }

    const { stat, readFile } = await import("node:fs/promises");

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

    if (startsWithOle2Magic(raw)) {
      return failure(
        "encrypted",
        "the document requires a password this connector does not have",
        rawContentSha256,
      );
    }

    let body: BodyResult;
    let documentXmlText: string;
    let coreTitle: string | null;
    try {
      const entries = unzipBounded(deps.unzipSync, raw);
      const documentXmlBytes = entries["word/document.xml"];
      if (!documentXmlBytes) {
        throw new Error("missing required part word/document.xml");
      }
      documentXmlText = decodeUtf8(documentXmlBytes);
      const documentRoot = parseXmlPart(
        deps,
        documentXmlText,
        "word/document.xml",
      );

      const stylesXmlBytes = entries["word/styles.xml"];
      const styleNames = stylesXmlBytes
        ? parseStyleNames(deps, decodeUtf8(stylesXmlBytes))
        : new Map<string, string>();

      body = buildBody(documentRoot, styleNames, {
        extractHeadings: this.extractHeadings,
        headingSeparator: this.headingSeparator,
        extractTables: this.extractTables,
      });
      coreTitle = readCoreTitle(deps, entries["docProps/core.xml"]);
    } catch (error) {
      if (error instanceof DecompressedSizeExceededError) {
        return failure("content_too_large", error.message, rawContentSha256);
      }
      // A grab-bag of zip/XML failures for a malformed package, none
      // sharing a common base — every such failure is still this
      // document's own structure failing to parse.
      return failure("corrupt", String(error), rawContentSha256);
    }

    if (body.paragraphs.length === 0) {
      return failure(
        "ocr_required",
        "no extractable text (an image-only document, or a genuinely empty one)",
        rawContentSha256,
      );
    }

    const text = body.paragraphs.join("\n\n");
    if (byteLen(text) > MAX_PASSAGE_BYTES) {
      return failure(
        "content_too_large",
        `${byteLen(text)} bytes of extracted text exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`,
        rawContentSha256,
      );
    }

    const diagnostics: Diagnostic[] = [];
    const partialMessage = partialExtractionMessage(documentXmlText);
    if (partialMessage !== null) {
      diagnostics.push(
        new Diagnostic({
          code: "partial_extraction",
          message: partialMessage,
          source,
        }),
      );
    }

    return new ConnectorDocument({
      source,
      text,
      locators: body.locators,
      sections: body.sections,
      metadata: new ConnectorMetadata({
        originUri: source,
        displayName,
        title: coreTitle ?? body.firstHeading,
        contentType: CONTENT_TYPE,
      }),
      fingerprintInputs: await this.fingerprint(rawContentSha256),
      diagnostics,
    });
  }
}
