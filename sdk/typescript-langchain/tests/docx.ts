/**
 * Minimal OOXML (`.docx`) byte-builders for `docx-connector.test.ts`
 * (issue #350's TypeScript parity port, issue #415) — hand-assembled
 * rather than shipped as binary fixtures, the same "synthesize at test
 * time" convention the Python twin's `tests/_docx.py` uses.
 *
 * Assembled as a raw zip of exactly the parts `DocxConnector` needs
 * (`[Content_Types].xml`, `_rels/.rels`, `word/document.xml`,
 * `word/styles.xml`, `word/_rels/document.xml.rels`, `docProps/core.xml`)
 * via `fflate`'s `zipSync` — the mechanical TS mirror of the Python
 * twin's own `zipfile.ZipFile` assembly. There is no TypeScript
 * equivalent of `python-docx`'s writer, so unlike the Python twin (whose
 * `test_docx_connector.py` separately builds one document through
 * `python-docx`'s real `Document()`/`.save()`), this port has no second,
 * independently-produced fixture; `realWordStyles()` below instead
 * reproduces the one real-writer detail that actually matters to
 * `DocxConnector`'s own heading detection — a genuine Word install names
 * its Heading-N style `heading N`, lowercase — verified against real
 * `python-docx` output (confirmed at TypeScript-port time, issue #415).
 */

import { strToU8, zipSync } from "fflate";

const CONTENT_TYPES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels"
 ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType=
 "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType=
 "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType=
 "application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>`;

const PACKAGE_RELS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type=
 "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
 Target="word/document.xml"/>
<Relationship Id="rId2" Type=
 "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"
 Target="docProps/core.xml"/>
</Relationships>`;

const DOCUMENT_RELS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type=
 "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
 Target="styles.xml"/>
</Relationships>`;

// Heading1..Heading6, each named "Heading N" (capitalized) — a heading
// built with `heading()` below is recognized via `DocxConnector`'s
// case-insensitive style-name match, not because it copies real Word's
// own (lowercase) naming; `realWordStyles()` below covers that case
// separately. `MonTitre` is deliberately NOT a recognized heading name at
// all — it exists only so `outlineHeading()` can prove the
// `w:outlineLvl` fallback fires for a document using a custom/localized
// style.
function defaultStyles(): string {
  const headingStyles = Array.from(
    { length: 6 },
    (_, index) =>
      `<w:style w:type="paragraph" w:styleId="Heading${index + 1}">` +
      `<w:name w:val="Heading ${index + 1}"/><w:basedOn w:val="Normal"/></w:style>`,
  ).join("");
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">` +
    `<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>` +
    headingStyles +
    `<w:style w:type="paragraph" w:styleId="MonTitre">` +
    `<w:name w:val="MonTitre"/><w:basedOn w:val="Normal"/></w:style>` +
    `</w:styles>`
  );
}

/**
 * `word/styles.xml` bytes naming `Heading1` "heading 1" (lowercase, with
 * a space) — exactly how real `python-docx`-authored (and real
 * Word-authored) output names it, confirmed by inspecting real
 * `python-docx` writer output at port time. Passed to {@link docxBytes}'s
 * `stylesXml` override so a test can prove `DocxConnector`'s heading
 * detection recognizes the real naming convention, not only this
 * module's own capitalized dialect — the TS stand-in for the Python
 * twin's `test_real_python_docx_writer_output_round_trips` (there is no
 * TypeScript OOXML *writer* library to produce a genuine round-trip
 * fixture from, so this reproduces the one naming detail that test
 * actually exercised instead).
 */
export function realWordStyles(): Uint8Array {
  const xml =
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">` +
    `<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>` +
    `<w:style w:type="paragraph" w:styleId="Heading1">` +
    `<w:name w:val="heading 1"/><w:basedOn w:val="Normal"/></w:style>` +
    `</w:styles>`;
  return strToU8(xml);
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function run(text: string): string {
  return `<w:r><w:t xml:space="preserve">${escapeXml(text)}</w:t></w:r>`;
}

/** One plain (or styled) paragraph. */
export function para(text: string, options?: { style?: string }): string {
  const ppr = options?.style
    ? `<w:pPr><w:pStyle w:val="${options.style}"/></w:pPr>`
    : "";
  return `<w:p>${ppr}${run(text)}</w:p>`;
}

/** A paragraph styled `HeadingN` — the primary, style-name-based signal
 * `DocxConnector` looks for first. */
export function heading(text: string, level: number): string {
  return para(text, { style: `Heading${level}` });
}

/** A paragraph carrying only `w:outlineLvl` (0-based; `level` is 1-based)
 * and a style name (`MonTitre`) that isn't recognized as a heading at
 * all — exercises `DocxConnector`'s fallback for a document whose
 * heading styles aren't named the way the primary style-name match
 * expects. */
export function outlineHeading(text: string, level: number): string {
  return (
    `<w:p><w:pPr><w:pStyle w:val="MonTitre"/>` +
    `<w:outlineLvl w:val="${level - 1}"/></w:pPr>${run(text)}</w:p>`
  );
}

/** A plain table (no nested content) — `rows` is document order,
 * outer-to-inner cells left-to-right. */
export function table(rows: readonly (readonly string[])[]): string {
  const rowXml = rows
    .map(
      (row) =>
        `<w:tr>${row.map((cell) => `<w:tc><w:tcPr/><w:p>${run(cell)}</w:p></w:tc>`).join("")}</w:tr>`,
    )
    .join("");
  return `<w:tbl><w:tblPr/><w:tblGrid/>${rowXml}</w:tbl>`;
}

/** `table(rows)`, except cell `at` (0-based `[row, col]`) holds a nested
 * `table(nestedRows)` instead of its own plain text — a `w:tc` must
 * still end in a paragraph after a nested `w:tbl` (OOXML's own content
 * model), so an empty one follows it, exactly as real Word emits for
 * this shape. */
export function tableWithNestedCell(
  rows: readonly (readonly string[])[],
  options: {
    at: readonly [number, number];
    nestedRows: readonly (readonly string[])[];
  },
): string {
  return tableWithNestedCells(rows, {
    nested: new Map([[cellKey(options.at), options.nestedRows]]),
  });
}

function cellKey(at: readonly [number, number]): string {
  return `${at[0]},${at[1]}`;
}

/** `table(rows)`, except each cell keyed in `nested` (0-based
 * `[row, col]`) holds its own nested `table(...)` instead of plain text —
 * one parent table with a nested table in more than one of its own
 * cells, the shape a locator-numbering bug can only surface against
 * (issue #350 review: a nested table's ordinal must count across every
 * cell of its parent, not reset per cell). */
export function tableWithNestedCells(
  rows: readonly (readonly string[])[],
  options: { nested: ReadonlyMap<string, readonly (readonly string[])[]> },
): string {
  const rowXml = rows.map((row, rowIndex) => {
    const cells = row.map((cell, colIndex) => {
      const nestedRows = options.nested.get(cellKey([rowIndex, colIndex]));
      if (nestedRows !== undefined) {
        return `<w:tc><w:tcPr/>${table(nestedRows)}<w:p/></w:tc>`;
      }
      return `<w:tc><w:tcPr/><w:p>${run(cell)}</w:p></w:tc>`;
    });
    return `<w:tr>${cells.join("")}</w:tr>`;
  });
  return `<w:tbl><w:tblPr/><w:tblGrid/>${rowXml.join("")}</w:tbl>`;
}

/** A paragraph referencing footnote `noteId` — the reference alone,
 * without ever declaring a `word/footnotes.xml` part: enough for
 * `DocxConnector`'s own marker scan (it never reads the footnote text
 * itself, only detects that a reference exists in the body). */
export function footnoteReferencePara(text: string, noteId = 2): string {
  return `<w:p>${run(text)}<w:r><w:footnoteReference w:id="${noteId}"/></w:r></w:p>`;
}

export function endnoteReferencePara(text: string, noteId = 2): string {
  return `<w:p>${run(text)}<w:r><w:endnoteReference w:id="${noteId}"/></w:r></w:p>`;
}

export function commentReferencePara(text: string, commentId = 0): string {
  return `<w:p>${run(text)}<w:r><w:commentReference w:id="${commentId}"/></w:r></w:p>`;
}

/** A paragraph whose only content is a text box — real Word wraps this
 * in `w:drawing`/DrawingML shape markup; only the `w:txbxContent`
 * wrapper itself matters to `DocxConnector`'s marker scan, so the
 * surrounding shape XML is trimmed to the minimum well-formed markup. */
export function textboxPara(boxText: string): string {
  return (
    `<w:p><w:r><w:drawing><wps:txbx xmlns:wps=` +
    `"http://schemas.microsoft.com/office/word/2010/wordprocessingShape">` +
    `<w:txbxContent>${para(boxText)}</w:txbxContent>` +
    `</wps:txbx></w:drawing></w:r></w:p>`
  );
}

/** Assembles a complete, valid `.docx` whose `word/document.xml` body is
 * exactly `bodyXml` (already-built `para()`/`heading()`/`table()`
 * fragments, concatenated). `stylesXml` defaults to {@link defaultStyles}
 * (capitalized `Heading N` names); pass {@link realWordStyles} to prove
 * the lowercase real-Word naming convention is also recognized. */
export function docxBytes(
  bodyXml: string,
  options?: { title?: string; stylesXml?: Uint8Array },
): Uint8Array {
  const document =
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">` +
    `<w:body>${bodyXml}</w:body></w:document>`;
  const titleXml = options?.title
    ? `<dc:title>${escapeXml(options.title)}</dc:title>`
    : "";
  const core =
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/` +
    `metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">` +
    `${titleXml}</cp:coreProperties>`;
  return zipSync({
    "[Content_Types].xml": strToU8(CONTENT_TYPES),
    "_rels/.rels": strToU8(PACKAGE_RELS),
    "word/document.xml": strToU8(document),
    "word/styles.xml": options?.stylesXml ?? strToU8(defaultStyles()),
    "word/_rels/document.xml.rels": strToU8(DOCUMENT_RELS),
    "docProps/core.xml": strToU8(core),
  });
}

/** Three distinct corruption shapes `DocxConnector` must all report as
 * `corrupt`, never throw: `"not_zip"` (not a zip container at all),
 * `"missing_part"` (a valid zip missing `word/document.xml` entirely),
 * and `"malformed_xml"` (the part exists but its XML doesn't parse). */
export function corruptDocx(
  kind: "not_zip" | "missing_part" | "malformed_xml" = "not_zip",
): Uint8Array {
  // `docProps/core.xml` is included in every variant below (empty core
  // properties, no `<dc:title>`) for parity with the Python twin's own
  // fixture, even though `DocxConnector` itself only reads
  // `word/document.xml` eagerly — kept for byte-shape parity, not
  // because this port's own loader requires it.
  const core =
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/` +
    `metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"/>`;
  if (kind === "not_zip") {
    return strToU8("this is not a zip file at all, just plain bytes");
  }
  if (kind === "missing_part") {
    return zipSync({
      "[Content_Types].xml": strToU8(CONTENT_TYPES),
      "_rels/.rels": strToU8(PACKAGE_RELS),
      "docProps/core.xml": strToU8(core),
    });
  }
  if (kind === "malformed_xml") {
    return zipSync({
      "[Content_Types].xml": strToU8(CONTENT_TYPES),
      "_rels/.rels": strToU8(PACKAGE_RELS),
      "word/document.xml": strToU8("<w:document><w:body><w:p not closed"),
      "word/styles.xml": strToU8(defaultStyles()),
      "word/_rels/document.xml.rels": strToU8(DOCUMENT_RELS),
      "docProps/core.xml": strToU8(core),
    });
  }
  throw new Error(`unknown corruptDocx kind: ${String(kind)}`);
}

/** The first 8 bytes any MS-OFFCRYPTO password-protected Office file
 * starts with (the OLE2/Compound File Binary container signature) —
 * enough for `DocxConnector`'s own magic-byte check, which never parses
 * past this header. */
export function encryptedDocx(): Uint8Array {
  const magic = Uint8Array.from([
    0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1,
  ]);
  const padded = new Uint8Array(magic.length + 504);
  padded.set(magic, 0);
  return padded;
}
