/**
 * The PPTX connector (ADR 0007 §5/§7/§8/§10, issue #352): `.pptx` files
 * into {@link ConnectorDocument}, preserving slide number, slide title,
 * and speaker notes. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors.pptx` module (issue #415).
 *
 * The Python twin parses through `python-pptx`'s object model
 * (`presentation.slides`, `slide.shapes`, `shape.text_frame`, …). As with
 * `docx.ts`, Node has no equivalent OOXML library, so this port reads the
 * same OOXML parts directly — `ppt/presentation.xml` (+ its own
 * `.rels`, for slide order), each `ppt/slides/slideN.xml` (+ its own
 * `.rels`, for a speaker-notes relationship), and each referenced
 * `ppt/notesSlides/notesSlideN.xml` — through the same two dynamically
 * `import()`-ed optional deps `docx.ts` uses (`fflate`'s `unzipSync`,
 * `fast-xml-parser`'s `XMLParser`/`XMLValidator`).
 *
 * A slide's shapes are walked in `p:spTree`'s own child order (the slide
 * part's XML document order — the same "trust the format's own
 * structure, never re-sort by position" posture `docx.ts`'s body walk
 * already takes), recursing into a `p:grpSp` group shape's own nested
 * shapes. Each non-empty paragraph (`a:p`) in a text-frame shape (`p:sp`
 * with a `p:txBody`) becomes its own paragraph in `text` — the same
 * one-paragraph-per-`<a:p>` granularity `docx.ts` keeps for `<w:p>` — and
 * a table (a `p:graphicFrame` whose `a:graphicData/@uri` names the table
 * namespace) becomes exactly ONE paragraph, its rows joined with `\n`,
 * cells with `" | "`, the same convention `docx.ts` already uses for a
 * `<w:tbl>`.
 *
 * ADR 0007 §7.2 allows at most one locator per paragraph; this connector
 * spends that budget distinguishing a slide's own body (including its
 * table(s)) from its speaker notes, rather than — as `docx.ts` does for
 * its own, page-less format — distinguishing a table from ordinary body
 * text. Every body paragraph on slide N (an ordinary text-frame paragraph
 * or a table) carries `{"kind": "slide", "value": "N"}`; a slide's
 * speaker notes are read as their own paragraph(s), each carrying
 * `{"kind": "speaker_notes", "value": "N"}` (ADR 0007 §7.3: "never folded
 * into the paragraph next to it") — appended immediately after that
 * slide's own body paragraphs, never interleaved with the next slide's.
 * A slide's title placeholder — the first direct `p:sp` child of
 * `p:spTree` carrying a `p:ph` whose `idx` is 0 (explicit, or omitted:
 * `idx` defaults to 0, the OOXML convention a genuine PowerPoint/
 * `python-pptx` title placeholder relies on) — is read like any other
 * text-frame shape (its text stays an ordinary body paragraph, carrying
 * the same `slide` locator as everything else on that slide) and,
 * additionally, becomes a {@link SectionEntry} anchored at its own first
 * paragraph — the same "prose heading, unchanged meaning" ADR 0007 §7.1
 * keeps `section` to.
 *
 * No OCR engine, no rasterizer: a presentation left with no extractable
 * text after the walk (e.g. an image-only deck) degrades to
 * `ocr_required` with empty text. Encrypted (password-protected) PPTX
 * files use the same MS-OFFCRYPTO OLE2/CFB container every other Office
 * format does, recognized by its own magic bytes and reported `encrypted`
 * before ever handing the bytes to the zip/XML parsers, the same posture
 * `docx.ts` already takes.
 *
 * A chart's own data lives in a separate `chartN.xml` part, a SmartArt
 * diagram's own text in a separate diagram data part, and an embedded/
 * linked OLE object's content is opaque entirely — none of them reachable
 * through this connector's own shape walk. A slide referencing any of
 * them is named in a single `partial_extraction` diagnostic rather than
 * silently short-changed, the same "structurally present but not walked"
 * posture `docx.ts` already takes for footnote/endnote/comment/text-box
 * content. A picture's own visual content is simply not text this
 * connector can read; a picture-only slide contributes no paragraphs of
 * its own, the same silent-omission posture `docx.ts` already takes for
 * an image.
 *
 * `.ppt` (legacy binary) and `.pptm` (macro-enabled) are both
 * `unsupported_format` — only the plain `.pptx` OOXML format is read.
 */

import type { Locator } from "taguru";

import { MAX_PASSAGE_BYTES } from "../extract.js";
import type { Connector } from "./protocol.js";
import { byteLen, sanitizeParagraphText } from "./structure.js";
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

export const PARSER_NAME = "taguru-pptx-connector";
export const PARSER_VERSION = "1.0.0";

const SUPPORTED_SUFFIXES: ReadonlySet<string> = new Set([".pptx"]);
const CONTENT_TYPE =
  "application/vnd.openxmlformats-officedocument.presentationml.presentation";

// The raw-file cap, checked before any parsing — same two-cap shape
// docx.ts uses for the same reason.
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

// MS-OFFCRYPTO password-protected Office documents (including .pptx) are
// re-packaged as a whole OLE2/Compound File Binary container, not a zip —
// the same magic number docx.ts already checks for the same reason.
const OLE2_MAGIC = Uint8Array.from([
  0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1,
]);

// Content this connector's own shape walk cannot reach at all — named in
// a single `partial_extraction` diagnostic rather than silently omitted.
// Every marker is a literal opening-tag search over each `ppt/slides/
// slideN.xml` part's decoded text (never a full second parse, the same
// convention docx.ts's own PARTIAL_CONTENT_MARKERS already uses): a
// chart's own data lives in a separate `chartN.xml` part; a SmartArt
// diagram's own text lives in a separate diagram data part (`<dgm:relIds>`
// is the reference PowerPoint itself leaves in the slide, pointing at
// it); an embedded/linked OLE object's content is opaque to this
// connector entirely.
const PARTIAL_CONTENT_MARKERS: ReadonlyArray<readonly [string, string]> = [
  ["<c:chart", "chart"],
  ["<dgm:relIds", "SmartArt diagram"],
  ["<p:oleObj", "embedded object"],
];

const SLIDE_PART_RE = /^ppt\/slides\/slide\d+\.xml$/;

// python-pptx's own `_shape_tags` (oxml/shapes/groupshape.py): the shape
// element tags recognized as children of a `p:spTree`/`p:grpSp` — used
// here only to skip non-shape metadata siblings (`p:nvGrpSpPr`,
// `p:grpSpPr`) when walking a shape tree's direct children.
const SHAPE_TAGS: ReadonlySet<string> = new Set([
  "p:sp",
  "p:grpSp",
  "p:graphicFrame",
  "p:cxnSp",
  "p:pic",
  "p:contentPart",
]);

const TABLE_GRAPHIC_DATA_URI =
  "http://schemas.openxmlformats.org/drawingml/2006/table";
const NOTES_SLIDE_RELATIONSHIP_SUFFIX = "/notesSlide";

// -- minimal preserveOrder XML node helpers (duplicated from docx.ts on --
// -- purpose — see that module's own note on why) --------------------------

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

function suffixOf(reference: string): string {
  const lastSlash = Math.max(
    reference.lastIndexOf("/"),
    reference.lastIndexOf("\\"),
  );
  const base = lastSlash >= 0 ? reference.slice(lastSlash + 1) : reference;
  const dotIndex = base.lastIndexOf(".");
  return dotIndex > 0 ? base.slice(dotIndex).toLowerCase() : "";
}

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

/** Loads the two optional OOXML deps ONLY when a `.pptx` is actually
 * about to be parsed — see docx.ts's own `loadOoxmlDeps` for the full
 * rationale (identical here, duplicated rather than shared so this
 * module never depends on docx.ts). */
async function loadOoxmlDeps(): Promise<OoxmlDeps> {
  try {
    const [{ unzipSync }, { XMLParser, XMLValidator }] = await Promise.all([
      import("fflate"),
      import("fast-xml-parser"),
    ]);
    return { unzipSync, XMLParser, XMLValidator };
  } catch (error) {
    throw new Error(
      "PptxConnector requires fflate and fast-xml-parser — install with " +
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

/** Resolves an OOXML relationship `Target` (as found in a `.rels` part)
 * against the directory containing the part that declared it — e.g.
 * `resolveOoxmlTarget("ppt/slides", "../notesSlides/notesSlide1.xml")`
 * `=== "ppt/notesSlides/notesSlide1.xml"`. */
function resolveOoxmlTarget(basePartDir: string, target: string): string {
  if (target.startsWith("/")) {
    return target.slice(1);
  }
  const parts = basePartDir.split("/").filter((segment) => segment.length > 0);
  for (const segment of target.split("/")) {
    if (segment === "..") {
      parts.pop();
    } else if (segment !== "." && segment !== "") {
      parts.push(segment);
    }
  }
  return parts.join("/");
}

interface OoxmlRelationship {
  type: string;
  target: string;
}

function parseRelationships(
  deps: OoxmlDeps,
  relsXmlText: string,
  partName: string,
): ReadonlyMap<string, OoxmlRelationship> {
  const map = new Map<string, OoxmlRelationship>();
  const root = parseXmlPart(deps, relsXmlText, partName);
  const relsNode = root.find((node) => xmlTagName(node) === "Relationships");
  if (!relsNode) {
    return map;
  }
  for (const rel of xmlChildrenNamed(relsNode, "Relationship")) {
    const attrs = xmlAttrs(rel);
    const id = attrs["@_Id"];
    const type = attrs["@_Type"];
    const target = attrs["@_Target"];
    if (id && type && target) {
      map.set(id, { type, target });
    }
  }
  return map;
}

function partRelsName(partName: string): string {
  const lastSlash = partName.lastIndexOf("/");
  const dir = partName.slice(0, lastSlash);
  const file = partName.slice(lastSlash + 1);
  return `${dir}/_rels/${file}.rels`;
}

// -- run/paragraph text --------------------------------------------------

/** An `a:p` node's own text: `a:r` runs' `a:t` text verbatim, `a:br` as
 * `"\v"` (a vertical tab — PowerPoint's own soft-carriage-return
 * encoding), `a:fld` (a slide-number/date field) as its own cached
 * `a:t` text — mirrors `python-pptx`'s `_Paragraph.text` exactly
 * (`content_children`: `a:r`/`a:br`/`a:fld`, direct children only). */
function paragraphText(paragraph: XmlNode): string {
  let text = "";
  for (const child of xmlChildren(paragraph)) {
    const tag = xmlTagName(child);
    if (tag === "a:r" || tag === "a:fld") {
      const tNode = xmlFirstChildNamed(child, "a:t");
      text += tNode ? xmlTextOf(tNode) : "";
    } else if (tag === "a:br") {
      text += "\v";
    }
  }
  return text;
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

// -- slide order (ppt/presentation.xml + its own .rels) --------------------

/** The ordered list of `ppt/slides/slideN.xml`-shaped part names, in the
 * order `p:sldIdLst/p:sldId` gives them (resolved through
 * `ppt/_rels/presentation.xml.rels`) — mirrors what `python-pptx`'s own
 * `presentation.slides` iterates, NOT simply the numeric order of
 * `slideN.xml` file names (which happen to coincide in every fixture this
 * port's own tests build, but are not guaranteed to in general). */
function slidePartOrder(
  deps: OoxmlDeps,
  entries: Record<string, Uint8Array>,
): string[] {
  const presentationBytes = entries["ppt/presentation.xml"];
  if (!presentationBytes) {
    throw new Error("missing required part ppt/presentation.xml");
  }
  const presRoot = parseXmlPart(
    deps,
    decodeUtf8(presentationBytes),
    "ppt/presentation.xml",
  );
  const presentationNode = presRoot.find(
    (node) => xmlTagName(node) === "p:presentation",
  );
  if (!presentationNode) {
    throw new Error("ppt/presentation.xml has no p:presentation root element");
  }
  const relsBytes = entries["ppt/_rels/presentation.xml.rels"];
  const rels = relsBytes
    ? parseRelationships(
        deps,
        decodeUtf8(relsBytes),
        "ppt/_rels/presentation.xml.rels",
      )
    : new Map<string, OoxmlRelationship>();

  const sldIdLst = xmlFirstChildNamed(presentationNode, "p:sldIdLst");
  const order: string[] = [];
  if (sldIdLst) {
    for (const sldId of xmlChildrenNamed(sldIdLst, "p:sldId")) {
      const rId = xmlAttrs(sldId)["@_r:id"];
      const rel = rId !== undefined ? rels.get(rId) : undefined;
      if (rel) {
        order.push(resolveOoxmlTarget("ppt", rel.target));
      }
    }
  }
  return order;
}

// -- title placeholder ----------------------------------------------------

/** The slide's title placeholder shape node, or `null` — mirrors
 * `python-pptx`'s `SlideShapes.title`: the first DIRECT `p:sp` child of
 * `spTree` (never descending into a `p:grpSp`) carrying a `p:ph` whose
 * `idx` is 0 — `idx` defaults to 0 when omitted (the OOXML default, and
 * the convention a real title placeholder relies on: it typically omits
 * `idx` entirely, only setting `@type="title"`/`"ctrTitle"`). */
function findTitleShapeNode(spTree: XmlNode): XmlNode | null {
  for (const child of xmlChildren(spTree)) {
    if (xmlTagName(child) !== "p:sp") {
      continue;
    }
    const nvSpPr = xmlFirstChildNamed(child, "p:nvSpPr");
    const nvPr = nvSpPr ? xmlFirstChildNamed(nvSpPr, "p:nvPr") : null;
    const ph = nvPr ? xmlFirstChildNamed(nvPr, "p:ph") : null;
    if (!ph) {
      continue;
    }
    const idxAttr = xmlAttrs(ph)["@_idx"];
    const idx = idxAttr !== undefined ? Number.parseInt(idxAttr, 10) : 0;
    if (idx === 0) {
      return child;
    }
  }
  return null;
}

/** The notes slide's own notes-placeholder shape node, or `null` —
 * mirrors `python-pptx`'s `NotesSlide.notes_placeholder`: the first
 * placeholder shape (anywhere in the notes slide's own `spTree`) whose
 * `p:ph/@type` is `"body"`. */
function findNotesPlaceholder(spTree: XmlNode): XmlNode | null {
  for (const child of xmlChildren(spTree)) {
    if (xmlTagName(child) !== "p:sp") {
      continue;
    }
    const nvSpPr = xmlFirstChildNamed(child, "p:nvSpPr");
    const nvPr = nvSpPr ? xmlFirstChildNamed(nvSpPr, "p:nvPr") : null;
    const ph = nvPr ? xmlFirstChildNamed(nvPr, "p:ph") : null;
    if (ph && xmlAttrs(ph)["@_type"] === "body") {
      return child;
    }
  }
  return null;
}

function graphicDataNode(graphicFrame: XmlNode): XmlNode | null {
  const graphic = xmlFirstChildNamed(graphicFrame, "a:graphic");
  if (!graphic) {
    return null;
  }
  return xmlFirstChildNamed(graphic, "a:graphicData");
}

function graphicFrameHasTable(graphicFrame: XmlNode): boolean {
  const data = graphicDataNode(graphicFrame);
  return data !== null && xmlAttrs(data)["@_uri"] === TABLE_GRAPHIC_DATA_URI;
}

function cellText(cell: XmlNode): string {
  const txBody = xmlFirstChildNamed(cell, "a:txBody");
  if (!txBody) {
    return "";
  }
  return xmlChildrenNamed(txBody, "a:p").map(paragraphText).join("\n");
}

/** A table (`p:graphicFrame` with a table `a:graphicData`) becomes one
 * paragraph, its rows joined with `\n`, cells with `" | "` — mirrors
 * `docx.ts`'s own `cellText`/table-row handling exactly, over `a:tbl`/
 * `a:tr`/`a:tc` in place of `w:tbl`/`w:tr`/`w:tc`. No `a:gridSpan`/
 * `a:hMerge`/`a:vMerge` resolution — same simplification, same "no test
 * exercises a merged cell" justification, as `docx.ts`'s own `cellText`. */
function handleTable(
  graphicFrame: XmlNode,
  commit: (text: string) => number | null,
): void {
  const data = graphicDataNode(graphicFrame);
  const tbl = data ? xmlFirstChildNamed(data, "a:tbl") : null;
  if (!tbl) {
    return;
  }
  const rowsText: string[] = [];
  for (const row of xmlChildrenNamed(tbl, "a:tr")) {
    const cells = xmlChildrenNamed(row, "a:tc").map(cellText);
    if (cells.some((cell) => cell.trim() !== "")) {
      rowsText.push(cells.join(" | "));
    }
  }
  if (rowsText.length > 0) {
    commit(rowsText.join("\n"));
  }
}

// -- body walk -------------------------------------------------------------

/** The one pass over every slide builds all three at once — mirrors
 * `docx.ts`'s own `BodyResult`, one paragraph-indexed pass producing
 * `paragraphs` plus paragraph-indexed `locators`/`sections` together, so
 * they can never drift out of sync with each other. */
interface BodyResult {
  paragraphs: string[];
  locators: LocatorEntry[];
  sections: SectionEntry[];
  firstTitle: string | null;
}

function buildSlide(
  deps: OoxmlDeps,
  entries: Record<string, Uint8Array>,
  slidePartName: string,
  slideNumber: number,
  result: BodyResult,
  options: {
    extractTitles: boolean;
    extractSpeakerNotes: boolean;
    extractTables: boolean;
  },
): void {
  const slideBytes = entries[slidePartName];
  if (!slideBytes) {
    throw new Error(`missing required part ${slidePartName}`);
  }
  const slideRoot = parseXmlPart(deps, decodeUtf8(slideBytes), slidePartName);
  const slideNode = slideRoot.find((node) => xmlTagName(node) === "p:sld");
  if (!slideNode) {
    throw new Error(`${slidePartName} has no p:sld root element`);
  }
  const cSld = xmlFirstChildNamed(slideNode, "p:cSld");
  const spTree = cSld ? xmlFirstChildNamed(cSld, "p:spTree") : null;

  const bodyLocator: Locator = { kind: "slide", value: String(slideNumber) };

  const commit = (text: string, locator: Locator): number | null => {
    const clean = sanitizeParagraphText(text);
    if (!clean) {
      return null;
    }
    const index = result.paragraphs.length;
    result.paragraphs.push(clean);
    result.locators.push(new LocatorEntry({ paragraph: index, locator }));
    return index;
  };

  const titleElement =
    options.extractTitles && spTree !== null
      ? findTitleShapeNode(spTree)
      : null;
  let titleIndex: number | null = null;
  let titleLabel: string | null = null;

  const handleShape = (shape: XmlNode): void => {
    const tag = xmlTagName(shape);
    if (tag === "p:grpSp") {
      for (const child of xmlChildren(shape)) {
        const childTag = xmlTagName(child);
        if (childTag !== null && SHAPE_TAGS.has(childTag)) {
          handleShape(child);
        }
      }
      return;
    }
    if (
      options.extractTables &&
      tag === "p:graphicFrame" &&
      graphicFrameHasTable(shape)
    ) {
      handleTable(shape, (text) => commit(text, bodyLocator));
      return;
    }
    if (tag !== "p:sp") {
      return; // only a p:sp shape can have a text frame
    }
    const isTitle = titleElement !== null && shape === titleElement;
    const txBody = xmlFirstChildNamed(shape, "p:txBody");
    if (!txBody) {
      return;
    }
    for (const paragraph of xmlChildrenNamed(txBody, "a:p")) {
      const index = commit(paragraphText(paragraph), bodyLocator);
      if (index === null) {
        continue;
      }
      if (isTitle && titleIndex === null) {
        // The committed (sanitized) paragraph text, not the raw
        // `paragraphText(paragraph)` — sanitizeParagraphText can collapse
        // an interior blank-line run, and the section label must match
        // what `result.paragraphs[index]` actually holds.
        const label = result.paragraphs[index]!;
        if (byteLen(label) <= MAX_SECTION_BYTES) {
          titleIndex = index;
          titleLabel = label;
        }
      }
    }
  };

  if (spTree !== null) {
    for (const shape of xmlChildren(spTree)) {
      const tag = xmlTagName(shape);
      if (tag !== null && SHAPE_TAGS.has(tag)) {
        handleShape(shape);
      }
    }
  }

  if (titleIndex !== null && titleLabel !== null) {
    result.sections.push(
      new SectionEntry({ paragraph: titleIndex, section: titleLabel }),
    );
    if (result.firstTitle === null) {
      result.firstTitle = titleLabel;
    }
  }

  if (!options.extractSpeakerNotes) {
    return;
  }

  // The slide's own relationship to a notes part (`ppt/slides/_rels/
  // slideN.xml.rels`) is resolved OUTSIDE the try/catch below: a
  // malformed `.rels` part is treated the same as any other malformed
  // OOXML part this connector eagerly reads — `corrupt` for the whole
  // document — mirroring `python-pptx`'s own eager, whole-package
  // relationship loading. Only the notes part's OWN content is allowed to
  // degrade silently to "no notes" (mirrors the narrow try/except
  // `python-pptx`-based `pptx.py` wraps around `slide.notes_slide.
  // notes_text_frame` specifically, never around `has_notes_slide`).
  const relsBytes = entries[partRelsName(slidePartName)];
  if (!relsBytes) {
    return;
  }
  const rels = parseRelationships(
    deps,
    decodeUtf8(relsBytes),
    partRelsName(slidePartName),
  );
  let notesTarget: string | null = null;
  for (const rel of rels.values()) {
    if (rel.type.endsWith(NOTES_SLIDE_RELATIONSHIP_SUFFIX)) {
      notesTarget = resolveOoxmlTarget("ppt/slides", rel.target);
      break;
    }
  }
  if (notesTarget === null) {
    return;
  }
  const notesBytes = entries[notesTarget];
  if (!notesBytes) {
    return; // a dangling relationship degrades to "no notes"
  }

  let notesTxBody: XmlNode | null = null;
  try {
    const notesRoot = parseXmlPart(deps, decodeUtf8(notesBytes), notesTarget);
    const notesNode = notesRoot.find((node) => xmlTagName(node) === "p:notes");
    const notesCSld = notesNode
      ? xmlFirstChildNamed(notesNode, "p:cSld")
      : null;
    const notesSpTree = notesCSld
      ? xmlFirstChildNamed(notesCSld, "p:spTree")
      : null;
    const notesPlaceholder = notesSpTree
      ? findNotesPlaceholder(notesSpTree)
      : null;
    notesTxBody = notesPlaceholder
      ? xmlFirstChildNamed(notesPlaceholder, "p:txBody")
      : null;
  } catch {
    notesTxBody = null; // a malformed notes part degrades to "no notes", never raised
  }
  if (notesTxBody === null) {
    return;
  }
  const notesLocator: Locator = {
    kind: "speaker_notes",
    value: String(slideNumber),
  };
  for (const paragraph of xmlChildrenNamed(notesTxBody, "a:p")) {
    commit(paragraphText(paragraph), notesLocator);
  }
}

function buildPresentation(
  deps: OoxmlDeps,
  entries: Record<string, Uint8Array>,
  options: {
    extractTitles: boolean;
    extractSpeakerNotes: boolean;
    extractTables: boolean;
  },
): BodyResult {
  const result: BodyResult = {
    paragraphs: [],
    locators: [],
    sections: [],
    firstTitle: null,
  };
  const order = slidePartOrder(deps, entries);
  order.forEach((slidePartName, index) => {
    buildSlide(deps, entries, slidePartName, index + 1, result, options);
  });
  return result;
}

function partialExtractionMessage(
  entries: Record<string, Uint8Array>,
): string | null {
  // Scanned slide by slide, never decoded-and-joined into one string —
  // the joined XML of a well-compressed deck can run to hundreds of MiB,
  // and every marker found ends its own search early. `kinds` is rebuilt
  // in PARTIAL_CONTENT_MARKERS declaration order so the message wording
  // is independent of slide order.
  const found = new Set<string>();
  for (const name of Object.keys(entries)) {
    if (!SLIDE_PART_RE.test(name)) {
      continue;
    }
    if (found.size === PARTIAL_CONTENT_MARKERS.length) {
      break;
    }
    const slideText = decodeUtf8(entries[name]!);
    for (const [marker, kind] of PARTIAL_CONTENT_MARKERS) {
      if (!found.has(kind) && slideText.includes(marker)) {
        found.add(kind);
      }
    }
  }
  const kinds = PARTIAL_CONTENT_MARKERS.filter(([, kind]) => found.has(kind)).map(
    ([, kind]) => kind,
  );
  if (kinds.length === 0) {
    return null;
  }
  return `${kinds.join(", ")} content is present but not extracted by this connector`;
}

// -- connector ---------------------------------------------------------------

export interface PptxConnectorOptions {
  /** Whether a slide's title placeholder becomes a {@link SectionEntry}
   * — the title's own text is always kept as an ordinary body paragraph
   * in `text` either way. Default `true`. */
  extractTitles?: boolean;
  /** Whether a slide's speaker notes are read at all; when `false`, notes
   * content is skipped entirely rather than folded into the slide's own
   * body. Default `true`. */
  extractSpeakerNotes?: boolean;
  /** Whether a table's row/cell text is extracted as its own paragraph;
   * when `false`, a table contributes nothing (never folded into a
   * neighboring shape's text). Default `true`. */
  extractTables?: boolean;
  /** Caps the raw file size checked before any parsing. Default 64 MiB. */
  maxFileBytes?: number;
}

/** Reference connector for `.pptx` files (issue #352). */
export class PptxConnector implements Connector {
  readonly parser: string = PARSER_NAME;
  readonly parserVersion: string = PARSER_VERSION;

  private readonly extractTitles: boolean;
  private readonly extractSpeakerNotes: boolean;
  private readonly extractTables: boolean;
  private readonly maxFileBytes: number;

  constructor(options: PptxConnectorOptions = {}) {
    this.extractTitles = options.extractTitles ?? true;
    this.extractSpeakerNotes = options.extractSpeakerNotes ?? true;
    this.extractTables = options.extractTables ?? true;
    this.maxFileBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
  }

  supports(reference: string): boolean {
    return SUPPORTED_SUFFIXES.has(suffixOf(reference));
  }

  private async optionsDigestValue(): Promise<string> {
    return optionsDigest({
      extract_titles: this.extractTitles,
      extract_speaker_notes: this.extractSpeakerNotes,
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
    // Loaded first, unconditionally — see docx.ts's own `read` for why.
    const deps = await loadOoxmlDeps();

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
      // affirmative evidence the file is NOT a PPTX — content_type stays
      // unclaimed (null) rather than asserting a MIME type this connector
      // has no basis for.
      return failure(
        "unsupported_format",
        `unsupported extension ${JSON.stringify(suffixOf(reference))} (only .pptx)`,
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
    const rawContentSha256 = await sha256HexBytes(raw);

    if (startsWithOle2Magic(raw)) {
      return failure(
        "encrypted",
        "the document requires a password this connector does not have",
        rawContentSha256,
      );
    }

    let body: BodyResult;
    let entries: Record<string, Uint8Array>;
    let coreTitle: string | null;
    try {
      entries = unzipBounded(deps.unzipSync, raw);
      body = buildPresentation(deps, entries, {
        extractTitles: this.extractTitles,
        extractSpeakerNotes: this.extractSpeakerNotes,
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
        "no extractable text (an image-only presentation, or a genuinely empty one)",
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
    const partialMessage = partialExtractionMessage(entries);
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
        title: coreTitle ?? body.firstTitle,
        contentType: CONTENT_TYPE,
      }),
      fingerprintInputs: await this.fingerprint(rawContentSha256),
      diagnostics,
    });
  }
}
