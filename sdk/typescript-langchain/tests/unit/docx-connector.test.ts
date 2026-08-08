/** DocxConnector — the .docx connector (ADR 0007 §7/§8, issue #350;
 * TypeScript parity: issue #415). */

import { mkdtemp, open, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../../src/extract.js";
import { DocxConnector } from "../../src/ingest-connectors/docx.js";
import {
  commentReferencePara,
  corruptDocx,
  docxBytes,
  encryptedDocx,
  endnoteReferencePara,
  footnoteReferencePara,
  heading,
  outlineHeading,
  para,
  realWordStyles,
  table,
  tableWithNestedCell,
  tableWithNestedCells,
  textboxPara,
} from "../docx.js";

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "docx-connector-"));
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

async function write(name: string, data: Uint8Array): Promise<string> {
  const path = join(dir, name);
  await writeFile(path, data);
  return path;
}

test("headings and paragraphs are kept and sections are breadcrumbs", async () => {
  const body =
    heading("Title Heading", 1) +
    para("Body para one.") +
    heading("Sub Heading", 2) +
    para("Body para two.");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe(
    "Title Heading\n\nBody para one.\n\nSub Heading\n\nBody para two.",
  );
  expect(
    document.sections.map((entry) => [entry.paragraph, entry.section]),
  ).toEqual([
    [0, "Title Heading"],
    [2, "Title Heading > Sub Heading"],
  ]);
  // An ordinary heading/body paragraph never carries a locator — only a
  // table does (§7.2's one-locator-per-paragraph budget is spent there).
  expect(document.locators).toEqual([]);
});

test("outline level fallback recognizes a non-heading named style", async () => {
  const body = outlineHeading("Localized Heading", 2) + para("Body.");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(
    document.sections.map((entry) => [entry.paragraph, entry.section]),
  ).toEqual([[0, "Localized Heading"]]);
});

test("outline level 9 is body text, not a heading", async () => {
  // ECMA-376: `w:outlineLvl` value 9 explicitly means "no outline level".
  // `outlineHeading(text, 10)` emits `w:val="9"` (the helper is 1-based).
  const body = outlineHeading("Plain body paragraph", 10) + para("Body.");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.sections).toEqual([]);
});

test("table becomes one paragraph with a table locator", async () => {
  const body =
    para("Intro") +
    table([
      ["A1", "B1"],
      ["A2", "B2"],
    ]) +
    para("Outro");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Intro\n\nA1 | B1\nA2 | B2\n\nOutro");
  expect(
    document.locators.map((entry) => [entry.paragraph, entry.locator]),
  ).toEqual([[1, { kind: "table", value: "1" }]]);
});

test("nested table gets its own paragraph and dotted locator", async () => {
  const body =
    para("Intro") +
    tableWithNestedCell(
      [
        ["A1", "B1"],
        ["A2", "B2"],
      ],
      { at: [1, 0], nestedRows: [["N1", "N2"]] },
    ) +
    para("Outro");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(
    document.locators.map((entry) => [entry.paragraph, entry.locator]),
  ).toEqual([
    [1, { kind: "table", value: "1" }],
    [2, { kind: "table", value: "1.1" }],
  ]);
});

test("nested tables in different cells of the same parent get distinct locators", async () => {
  // Regression for a numbering bug: a nested table's ordinal must count
  // across every cell of its parent table, not reset to 1 for each cell —
  // otherwise two distinct nested tables can end up sharing one locator
  // value, which would let a citation resolve to the wrong table.
  const body =
    para("Intro") +
    tableWithNestedCells(
      [
        ["", "B1"],
        ["A2", ""],
      ],
      {
        nested: new Map([
          ["0,0", [["NA1"]]],
          ["1,1", [["NB1"]]],
        ]),
      },
    );
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  const values = document.locators.map((entry) => entry.locator.value);
  expect(new Set(values).size).toBe(values.length);
  expect(values).toEqual(["1", "1.1", "1.2"]);
});

test("multiple top-level tables are numbered in document order", async () => {
  const body = table([["A"]]) + para("Between") + table([["B"]]);
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.locators.map((entry) => entry.locator.value)).toEqual([
    "1",
    "2",
  ]);
});

test("built text resplits into exactly the paragraphs the locators and sections index", async () => {
  // The invariant this connector's correctness rests on: re-running the
  // server's own paragraph splitter over `text` must yield exactly as
  // many paragraphs as there are, with locator/section indices still
  // pointing at the right one.
  const body =
    heading("Title", 1) +
    para("Body one.") +
    table([
      ["A1", "B1"],
      ["A2", "B2"],
    ]) +
    para("Body two.\nsecond line.") +
    heading("Next", 2) +
    para("Body three.");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  const resplit = splitParagraphs(document.text);
  expect(resplit.join("\n\n")).toBe(document.text);
  for (const entry of document.locators) {
    expect(entry.paragraph).toBeGreaterThanOrEqual(0);
    expect(entry.paragraph).toBeLessThan(resplit.length);
  }
  for (const entry of document.sections) {
    expect(entry.paragraph).toBeGreaterThanOrEqual(0);
    expect(entry.paragraph).toBeLessThan(resplit.length);
  }
});

// Adapted from the Python twin's `test_real_python_docx_writer_output_
// round_trips`: there is no TypeScript OOXML *writer* library to build a
// genuine round-trip fixture from (see tests/docx.ts's own
// `realWordStyles` doc comment), so this proves the one real-Word detail
// that test actually exercised — the lowercase, space-separated
// `heading N` style-name convention a genuine Word/python-docx install
// writes — using a hand-built fixture instead of a real writer's output.
test("real Word style naming convention (lowercase 'heading N') is recognized", async () => {
  const body =
    heading("Real Word Heading", 1) +
    para("Real Word body paragraph.") +
    table([["X1", "Y1"]]);
  const path = await write(
    "real.docx",
    docxBytes(body, { stylesXml: realWordStyles() }),
  );
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toContain("Real Word Heading");
  expect(document.text).toContain("Real Word body paragraph.");
  expect(document.text).toContain("X1 | Y1");
  expect(
    document.sections.map((entry) => [entry.paragraph, entry.section]),
  ).toEqual([[0, "Real Word Heading"]]);
  expect(document.locators.map((entry) => entry.locator)).toEqual([
    { kind: "table", value: "1" },
  ]);
});

test.each([
  [() => footnoteReferencePara("See note."), "footnote"],
  [() => endnoteReferencePara("See note."), "endnote"],
  [() => commentReferencePara("Commented."), "comment"],
  [() => textboxPara("Box text"), "text box"],
])(
  "unreachable content is named partial_extraction (%#: %s)",
  async (builder, expectedKind) => {
    const body = para("Intro") + (builder as () => string)();
    const path = await write("doc.docx", docxBytes(body));
    const document = await new DocxConnector().read(path);

    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      "partial_extraction",
    ]);
    expect(document.diagnostics[0]!.message).toContain(expectedKind);
  },
);

test("corrupt docx variants are reported as corrupt", async () => {
  for (const kind of ["not_zip", "missing_part", "malformed_xml"] as const) {
    const path = await write(`broken-${kind}.docx`, corruptDocx(kind));
    const document = await new DocxConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      "corrupt",
    ]);
  }
});

test("encrypted docx is reported encrypted without being opened", async () => {
  const path = await write("enc.docx", encryptedDocx());
  const document = await new DocxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "encrypted",
  ]);
});

test("empty document is reported ocr_required", async () => {
  const path = await write("empty.docx", docxBytes(""));
  const document = await new DocxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "ocr_required",
  ]);
});

test("unsupported extension is reported without touching the filesystem", async () => {
  const path = await write("doc.doc", new TextEncoder().encode("whatever"));
  const document = await new DocxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unsupported_format",
  ]);
  // The extension mismatch is affirmative evidence the file isn't a DOCX
  // — content_type stays unclaimed rather than asserting a MIME type this
  // connector has no basis for.
  expect(document.metadata.contentType).toBeNull();
});

test("docm extension is also unsupported", async () => {
  const path = await write("doc.docm", docxBytes(para("x")));
  const document = await new DocxConnector().read(path);

  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unsupported_format",
  ]);
  expect(document.metadata.contentType).toBeNull();
});

test("other failure codes still claim the docx MIME type", async () => {
  // Unlike `unsupported_format`, every other failure below happens AFTER
  // the `.docx` extension already matched, so claiming the DOCX MIME type
  // is a reasonable inference from a trusted extension, not asserted
  // fact — kept unchanged by the `unsupported_format` case above.
  const document = await new DocxConnector().read(join(dir, "missing.docx"));

  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unreadable",
  ]);
  expect(document.metadata.contentType).toBe(
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  );
});

test("oversized file is reported without parsing", async () => {
  const path = join(dir, "big.docx");
  const handle = await open(path, "w");
  await handle.write(new Uint8Array([0]), 0, 1, 1024);
  await handle.close();
  const document = await new DocxConnector({ maxFileBytes: 1023 }).read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "content_too_large",
  ]);
});

test("oversized extracted text is reported content_too_large", async () => {
  const hugeParagraph = "x".repeat(MAX_PASSAGE_BYTES + 1);
  const path = await write("huge.docx", docxBytes(para(hugeParagraph)));
  const document = await new DocxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "content_too_large",
  ]);
});

test("oversized source id is reported without reading the file", async () => {
  const longReference = "x".repeat(1025) + ".docx";
  const document = await new DocxConnector().read(longReference);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "source_id_too_long",
  ]);
});

test("missing file is reported unreadable", async () => {
  const document = await new DocxConnector().read(join(dir, "missing.docx"));

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unreadable",
  ]);
});

test("supports matches only docx", () => {
  const connector = new DocxConnector();
  expect(connector.supports("a.docx")).toBe(true);
  expect(connector.supports("a.DOCX")).toBe(true);
  expect(connector.supports("a.doc")).toBe(false);
  expect(connector.supports("a.docm")).toBe(false);
  expect(connector.supports("a.txt")).toBe(false);
});

test("parser identity is stamped into the fingerprint", async () => {
  const path = await write("doc.docx", docxBytes(para("Body.")));
  const connector = new DocxConnector();
  const document = await connector.read(path);

  expect(document.fingerprintInputs.parser).toBe(connector.parser);
  expect(document.fingerprintInputs.parserVersion).toBe(
    connector.parserVersion,
  );
});

test("fingerprint hashes the raw bytes and the effective options", async () => {
  const data = docxBytes(para("Body."));
  const path = await write("doc.docx", data);

  const defaultDocument = await new DocxConnector().read(path);
  const expectedSha256 = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", data as BufferSource)),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
  expect(defaultDocument.fingerprintInputs.rawContentSha256).toBe(
    expectedSha256,
  );

  const otherDocument = await new DocxConnector({
    headingSeparator: " / ",
  }).read(path);
  expect(otherDocument.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
  // Changing an option never touches the raw-bytes hash — the two
  // fingerprint fields answer independent questions (ADR 0007 §5/§6.2).
  expect(otherDocument.fingerprintInputs.rawContentSha256).toBe(
    defaultDocument.fingerprintInputs.rawContentSha256,
  );
});

test("extractHeadings false keeps paragraph text but drops sections", async () => {
  const body = heading("Title", 1) + para("Body.");
  const path = await write("doc.docx", docxBytes(body));

  const defaultDocument = await new DocxConnector().read(path);
  const document = await new DocxConnector({ extractHeadings: false }).read(
    path,
  );

  expect(document.sections).toEqual([]);
  expect(document.text).toBe(defaultDocument.text);
  expect(document.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
});

test("extractTables false keeps table text but drops the locator", async () => {
  const body = para("Intro") + table([["A1", "B1"]]);
  const path = await write("doc.docx", docxBytes(body));

  const defaultDocument = await new DocxConnector().read(path);
  const document = await new DocxConnector({ extractTables: false }).read(path);

  expect(document.locators).toEqual([]);
  expect(document.text).toBe(defaultDocument.text);
  expect(document.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
});

test("metadata title prefers core properties over first heading", async () => {
  const body = heading("Heading Title", 1) + para("Body.");
  const path = await write(
    "doc.docx",
    docxBytes(body, { title: "Explicit Title" }),
  );
  const document = await new DocxConnector().read(path);

  expect(document.metadata.title).toBe("Explicit Title");
});

test("metadata title falls back to first heading", async () => {
  const body = heading("Heading Title", 1) + para("Body.");
  const path = await write("doc.docx", docxBytes(body));
  const document = await new DocxConnector().read(path);

  expect(document.metadata.title).toBe("Heading Title");
});

test("metadata content_type is the OOXML wordprocessing MIME type", async () => {
  const path = await write("doc.docx", docxBytes(para("Body.")));
  const document = await new DocxConnector().read(path);

  expect(document.metadata.contentType).toBe(
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  );
});

describe("missing OOXML dependencies", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doMock("fflate", () => {
      throw new Error("Cannot find module 'fflate'");
    });
  });

  afterEach(() => {
    vi.doUnmock("fflate");
    vi.resetModules();
  });

  test("reports a clear error instead of parsing", async () => {
    const { DocxConnector: IsolatedDocxConnector } =
      await import("../../src/ingest-connectors/docx.js");
    const path = await write("doc.docx", docxBytes(para("Body.")));

    await expect(new IsolatedDocxConnector().read(path)).rejects.toThrow(
      /fflate and fast-xml-parser/,
    );
  });
});
