/**
 * Minimal `.pptx` byte-builders for `pptx-connector.test.ts` (issue
 * #352's TypeScript parity port, issue #415).
 *
 * Unlike the Python twin's own `tests/_pptx.py` — which builds every
 * fixture through `python-pptx`'s real `Presentation()`/`.save()` writer,
 * deliberately avoiding a hand-rolled zip the way `tests/_docx.py` does
 * for DOCX, precisely BECAUSE a `.pptx` package spans several genuinely
 * interdependent parts (slide master, slide layouts, theme, …) even for
 * one blank slide — this port has no TypeScript OOXML *writer* library to
 * lean on at all. `PptxConnector` itself, however, never reads a slide
 * layout, slide master, or theme part (see pptx.ts's own module
 * docstring: title detection is `p:ph/@idx == 0` read directly off the
 * SLIDE's own placeholder, never resolved through layout inheritance) —
 * so this module hand-assembles only the parts `PptxConnector` actually
 * parses (`[Content_Types].xml`, `_rels/.rels`, `ppt/presentation.xml` +
 * its own `.rels`, each `ppt/slides/slideN.xml` + (when it has notes) its
 * own `.rels`, each referenced `ppt/notesSlides/notesSlideN.xml`,
 * `docProps/core.xml`) via `fflate`'s `zipSync` — the same "hand-roll
 * exactly what the connector needs" convention `tests/_docx.py` already
 * uses for DOCX, applied here for the same underlying reason: no writer
 * library available. Every fixture's shape below (`p:ph type="title"`
 * with `idx` omitted, `p:ph type="body"` on a notes placeholder, `a:tbl`
 * table markup, `p:grpSp` group nesting, the chart/SmartArt/OLE-object
 * `a:graphicData` marker shapes) was confirmed against REAL
 * `python-pptx` writer output at TypeScript-port time (issue #415).
 *
 * The public API mirrors the Python twin's own function names
 * (camelCased) — `blankPresentation`, `addTitleSlide`, `addBody`,
 * `addTable`, `addGroup`, `addNotes`, `addTitleAndContentSlide`,
 * `addChart`, `setCoreTitle`, `saveBytes`/`pptxBytes`, `withSmartart`,
 * `withOleObject`, `corruptPptx`, `encryptedPptx` — so
 * `pptx-connector.test.ts` reads as the same test suite as
 * `test_pptx_connector.py`, just assembling its fixtures a different way
 * underneath.
 */

import { strToU8, unzipSync, zipSync } from "fflate";

const A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main";
const P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main";
const R_NS =
  "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships";
const SLIDE_REL_TYPE = `${R_NS}/slide`;
const NOTES_SLIDE_REL_TYPE = `${R_NS}/notesSlide`;
const TABLE_GRAPHIC_DATA_URI =
  "http://schemas.openxmlformats.org/drawingml/2006/table";
const CHART_GRAPHIC_DATA_URI =
  "http://schemas.openxmlformats.org/drawingml/2006/chart";

export interface SlideHandle {
  titleXml: string | null;
  shapesXml: string[];
  notesParagraphs: string[] | null;
}

export interface PresentationState {
  slides: SlideHandle[];
  coreTitle: string | null;
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function runXml(text: string): string {
  return `<a:r><a:t>${escapeXml(text)}</a:t></a:r>`;
}

function paragraphXml(text: string): string {
  return `<a:p>${text ? runXml(text) : ""}</a:p>`;
}

let nextShapeId = 2;

function freshShapeId(): number {
  nextShapeId += 1;
  return nextShapeId;
}

export function blankPresentation(): PresentationState {
  return { slides: [], coreTitle: null };
}

export function setCoreTitle(
  presentation: PresentationState,
  title: string,
): void {
  presentation.coreTitle = title;
}

/**
 * Adds one slide. `title === null` mirrors the Python twin's own `Blank`
 * layout (no title placeholder at all — `shapes.title` finds none); any
 * other value (including `""`) mirrors `Title Only` (a title placeholder
 * IS present; `""` leaves it with no text, the shape a locator-anchor
 * test needs to prove an empty title creates no `SectionEntry`).
 */
export function addTitleSlide(
  presentation: PresentationState,
  title: string | null,
): SlideHandle {
  const slide: SlideHandle = {
    titleXml:
      title === null
        ? null
        : `<p:sp><p:nvSpPr><p:cNvPr id="${freshShapeId()}" name="Title"/><p:cNvSpPr/>` +
          `<p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:spPr/>` +
          `<p:txBody>${paragraphXml(title)}</p:txBody></p:sp>`,
    shapesXml: [],
    notesParagraphs: null,
  };
  presentation.slides.push(slide);
  return slide;
}

/**
 * A slide built with a title placeholder AND a real content placeholder
 * (`p:ph idx="1"`, no `type` — the same shape a genuine `Title and
 * Content` layout's body placeholder carries, confirmed against real
 * `python-pptx` output) — never a manually added textbox, unlike
 * {@link addBody} — proving `PptxConnector` reads an ordinary content
 * placeholder exactly like the plain textboxes every other fixture here
 * uses.
 */
export function addTitleAndContentSlide(
  presentation: PresentationState,
  options: { title: string; body: string },
): SlideHandle {
  const slide = addTitleSlide(presentation, options.title);
  slide.shapesXml.push(
    `<p:sp><p:nvSpPr><p:cNvPr id="${freshShapeId()}" name="Content Placeholder"/><p:cNvSpPr/>` +
      `<p:nvPr><p:ph idx="1"/></p:nvPr></p:nvSpPr><p:spPr/>` +
      `<p:txBody>${paragraphXml(options.body)}</p:txBody></p:sp>`,
  );
  return slide;
}

function textboxXml(paragraphs: readonly string[]): string {
  return (
    `<p:sp><p:nvSpPr><p:cNvPr id="${freshShapeId()}" name="TextBox"/>` +
    `<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr/>` +
    `<p:txBody>${paragraphs.map(paragraphXml).join("")}</p:txBody></p:sp>`
  );
}

/** A plain textbox holding one paragraph per `paragraphs` element — the
 * general body-shape case every ordinary bullet/text box slide
 * exercises. */
export function addBody(
  slide: SlideHandle,
  paragraphs: readonly string[],
): void {
  slide.shapesXml.push(textboxXml(paragraphs));
}

export function addTable(
  slide: SlideHandle,
  rows: readonly (readonly string[])[],
): void {
  const rowsXml = rows
    .map(
      (row) =>
        `<a:tr h="1">${row
          .map(
            (cell) =>
              `<a:tc><a:txBody>${paragraphXml(cell)}</a:txBody><a:tcPr/></a:tc>`,
          )
          .join("")}</a:tr>`,
    )
    .join("");
  slide.shapesXml.push(
    `<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="${freshShapeId()}" name="Table"/>` +
      `<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>` +
      `<p:xfrm><a:off x="0" y="0"/><a:ext cx="1" cy="1"/></p:xfrm>` +
      `<a:graphic><a:graphicData uri="${TABLE_GRAPHIC_DATA_URI}">` +
      `<a:tbl><a:tblGrid/>${rowsXml}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>`,
  );
}

/** A single group shape holding one textbox per `texts` element —
 * exercises `PptxConnector`'s own recursion into a group's nested
 * shapes. */
export function addGroup(slide: SlideHandle, texts: readonly string[]): void {
  const children = texts.map((text) => textboxXml([text])).join("");
  slide.shapesXml.push(
    `<p:grpSp><p:nvGrpSpPr><p:cNvPr id="${freshShapeId()}" name="Group"/>` +
      `<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>${children}</p:grpSp>`,
  );
}

export function addNotes(
  slide: SlideHandle,
  paragraphs: readonly string[],
): void {
  slide.notesParagraphs = [...paragraphs];
}

/** A chart graphic frame — its own data lives in a separate `chartN.xml`
 * part `PptxConnector` never reads, so this exercises the `chart` case
 * of its `partial_extraction` marker scan; the `a:graphicData/@uri`
 * naming the chart namespace (not the table one) also means it correctly
 * contributes no table text of its own. */
export function addChart(slide: SlideHandle): void {
  slide.shapesXml.push(
    `<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="${freshShapeId()}" name="Chart"/>` +
      `<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>` +
      `<p:xfrm><a:off x="0" y="0"/><a:ext cx="1" cy="1"/></p:xfrm>` +
      `<a:graphic><a:graphicData uri="${CHART_GRAPHIC_DATA_URI}">` +
      `<c:chart xmlns:c="${CHART_GRAPHIC_DATA_URI}" r:id="rIdChart"/>` +
      `</a:graphicData></a:graphic></p:graphicFrame>`,
  );
}

function slideXml(slide: SlideHandle): string {
  const shapes = (slide.titleXml ?? "") + slide.shapesXml.join("");
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<p:sld xmlns:a="${A_NS}" xmlns:p="${P_NS}" xmlns:r="${R_NS}">` +
    `<p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/>${shapes}</p:spTree></p:cSld></p:sld>`
  );
}

function notesSlideXml(paragraphs: readonly string[]): string {
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<p:notes xmlns:a="${A_NS}" xmlns:p="${P_NS}" xmlns:r="${R_NS}">` +
    `<p:cSld><p:spTree><p:nvGrpSpPr/><p:grpSpPr/>` +
    `<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes Placeholder"/><p:cNvSpPr/>` +
    `<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/>` +
    `<p:txBody>${paragraphs.map(paragraphXml).join("")}</p:txBody></p:sp>` +
    `</p:spTree></p:cSld></p:notes>`
  );
}

function contentTypesXml(): string {
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">` +
    `<Default Extension="rels" ContentType=` +
    `"application/vnd.openxmlformats-package.relationships+xml"/>` +
    `<Default Extension="xml" ContentType="application/xml"/>` +
    `</Types>`
  );
}

function packageRelsXml(): string {
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<Relationships xmlns="${RELS_NS}">` +
    `<Relationship Id="rId1" Type=` +
    `"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"` +
    ` Target="ppt/presentation.xml"/>` +
    `<Relationship Id="rId2" Type=` +
    `"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"` +
    ` Target="docProps/core.xml"/>` +
    `</Relationships>`
  );
}

/** Assembles a complete `.pptx` from `presentation`'s accumulated slide
 * state — only the parts `PptxConnector` itself ever reads (see this
 * module's own doc comment for why slide layouts/masters/theme are
 * omitted entirely). */
export function saveBytes(presentation: PresentationState): Uint8Array {
  const files: Record<string, Uint8Array> = {
    "[Content_Types].xml": strToU8(contentTypesXml()),
    "_rels/.rels": strToU8(packageRelsXml()),
  };

  const sldIdEntries = presentation.slides
    .map(
      (_, index) =>
        `<p:sldId id="${256 + index}" r:id="rIdSlide${index + 1}"/>`,
    )
    .join("");
  files["ppt/presentation.xml"] = strToU8(
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<p:presentation xmlns:p="${P_NS}" xmlns:r="${R_NS}">` +
      `<p:sldIdLst>${sldIdEntries}</p:sldIdLst></p:presentation>`,
  );
  const presRelEntries = presentation.slides
    .map(
      (_, index) =>
        `<Relationship Id="rIdSlide${index + 1}" Type="${SLIDE_REL_TYPE}" ` +
        `Target="slides/slide${index + 1}.xml"/>`,
    )
    .join("");
  files["ppt/_rels/presentation.xml.rels"] = strToU8(
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<Relationships xmlns="${RELS_NS}">${presRelEntries}</Relationships>`,
  );

  presentation.slides.forEach((slide, index) => {
    const slideNumber = index + 1;
    files[`ppt/slides/slide${slideNumber}.xml`] = strToU8(slideXml(slide));
    if (slide.notesParagraphs !== null) {
      files[`ppt/slides/_rels/slide${slideNumber}.xml.rels`] = strToU8(
        `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
          `<Relationships xmlns="${RELS_NS}">` +
          `<Relationship Id="rId1" Type="${NOTES_SLIDE_REL_TYPE}" ` +
          `Target="../notesSlides/notesSlide${slideNumber}.xml"/></Relationships>`,
      );
      files[`ppt/notesSlides/notesSlide${slideNumber}.xml`] = strToU8(
        notesSlideXml(slide.notesParagraphs),
      );
    }
  });

  const titleXml = presentation.coreTitle
    ? `<dc:title>${escapeXml(presentation.coreTitle)}</dc:title>`
    : "";
  files["docProps/core.xml"] = strToU8(
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
      `<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/` +
      `metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">` +
      `${titleXml}</cp:coreProperties>`,
  );

  return zipSync(files);
}

/** One-slide convenience wrapper over the primitives above, for the
 * common case of a single slide exercising one or two features at once.
 * `title` defaults to `null` (no title placeholder at all — see
 * {@link addTitleSlide}), unlike the Python twin's own default of
 * `None` mapped the same way. */
export function pptxBytes(
  options: {
    title?: string | null;
    bodies?: readonly string[];
    table?: readonly (readonly string[])[];
    notes?: readonly string[];
    chart?: boolean;
  } = {},
): Uint8Array {
  const presentation = blankPresentation();
  const slide = addTitleSlide(presentation, options.title ?? null);
  if (options.bodies && options.bodies.length > 0) {
    addBody(slide, options.bodies);
  }
  if (options.table !== undefined) {
    addTable(slide, options.table);
  }
  if (options.notes && options.notes.length > 0) {
    addNotes(slide, options.notes);
  }
  if (options.chart) {
    addChart(slide);
  }
  return saveBytes(presentation);
}

// A SmartArt diagram, minimal enough to parse: a `<p:graphicFrame>` whose
// `graphicData` names the diagram namespace, containing the
// `<dgm:relIds>` reference `PptxConnector`'s own marker scan looks for —
// confirmed against real `python-pptx`-adjacent OOXML structure; needs no
// real diagram data part behind it, only the reference PowerPoint itself
// leaves in the slide.
const SMARTART_MARKER =
  '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="9001" name="Diagram"/>' +
  "<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>" +
  '<p:xfrm><a:off x="0" y="0"/><a:ext cx="1" cy="1"/></p:xfrm>' +
  '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/diagram">' +
  '<dgm:relIds xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" ' +
  'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ' +
  'r:dm="rIdX" r:lo="rIdX" r:qs="rIdX" r:cs="rIdX"/></a:graphicData></a:graphic>' +
  "</p:graphicFrame>";

// Likewise minimal for an embedded OLE object: a `<p:graphicFrame>` whose
// `graphicData` names the presentation-ole namespace, containing the
// `<p:oleObj>` reference.
const OLE_OBJECT_MARKER =
  '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="9002" name="Object"/>' +
  "<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>" +
  '<p:xfrm><a:off x="0" y="0"/><a:ext cx="1" cy="1"/></p:xfrm>' +
  '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/presentationml/2006/ole">' +
  '<p:oleObj name="Object" r:id="rIdX" imgW="1" imgH="1" progId="Package" ' +
  'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">' +
  "<p:embed/></p:oleObj></a:graphicData></a:graphic>" +
  "</p:graphicFrame>";

function injectMarker(
  raw: Uint8Array,
  markerXml: string,
  slideNumber: number,
): Uint8Array {
  const name = `ppt/slides/slide${slideNumber}.xml`;
  const entries = unzipSync(raw);
  const xml = entries[name];
  if (!xml) {
    throw new Error(`marker injection failed: ${name} not present in archive`);
  }
  const decoded = new TextDecoder("utf-8").decode(xml);
  const patched = decoded.replace("</p:spTree>", `${markerXml}</p:spTree>`);
  if (patched === decoded) {
    throw new Error(`marker injection failed: ${name} has no </p:spTree>`);
  }
  entries[name] = strToU8(patched);
  return zipSync(entries);
}

export function withSmartart(raw: Uint8Array, slideNumber = 1): Uint8Array {
  return injectMarker(raw, SMARTART_MARKER, slideNumber);
}

export function withOleObject(raw: Uint8Array, slideNumber = 1): Uint8Array {
  return injectMarker(raw, OLE_OBJECT_MARKER, slideNumber);
}

/** Three distinct corruption shapes `PptxConnector` must all report as
 * `corrupt`, never throw: `"not_zip"` (not a zip container at all),
 * `"missing_part"` (a valid zip missing `ppt/presentation.xml`
 * entirely), and `"malformed_xml"` (a slide part exists but its XML
 * doesn't parse). */
export function corruptPptx(
  kind: "not_zip" | "missing_part" | "malformed_xml" = "not_zip",
): Uint8Array {
  if (kind === "not_zip") {
    return strToU8("this is not a zip file at all, just plain bytes");
  }
  const baseline = pptxBytes({ bodies: ["placeholder"] });
  const entries = unzipSync(baseline);
  if (kind === "missing_part") {
    delete entries["ppt/presentation.xml"];
  } else if (kind === "malformed_xml") {
    entries["ppt/slides/slide1.xml"] = strToU8(
      "<p:sld><p:cSld><p:spTree not closed",
    );
  } else {
    throw new Error(`unknown corruptPptx kind: ${String(kind)}`);
  }
  return zipSync(entries);
}

/** The first 8 bytes any MS-OFFCRYPTO password-protected Office file
 * starts with (the OLE2/Compound File Binary container signature) —
 * identical to `tests/docx.ts`'s own `encryptedDocx`: the signature is
 * format-agnostic. */
export function encryptedPptx(): Uint8Array {
  const magic = Uint8Array.from([
    0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1,
  ]);
  const padded = new Uint8Array(magic.length + 504);
  padded.set(magic, 0);
  return padded;
}
