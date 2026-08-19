/** PptxConnector — the .pptx connector (ADR 0007 §5/§7/§8/§10, issue
 * #352; TypeScript parity: issue #415). */

import { mkdtemp, open, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../../src/extract.js";
import { PptxConnector } from "../../src/ingest-connectors/pptx.js";
import {
  LocatorEntry,
  MAX_SECTION_BYTES,
} from "../../src/ingest-connectors/document.js";
import {
  addBody,
  addGroup,
  addNotes,
  addTable,
  addTitleAndContentSlide,
  addTitleSlide,
  blankPresentation,
  corruptPptx,
  encryptedPptx,
  pptxBytes,
  saveBytes,
  setCoreTitle,
  withOleObject,
  withSmartart,
} from "../pptx.js";

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "pptx-connector-"));
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

async function write(name: string, data: Uint8Array): Promise<string> {
  const path = join(dir, name);
  await writeFile(path, data);
  return path;
}

test("title and body paragraphs carry the same slide locator", async () => {
  const raw = pptxBytes({
    title: "Slide Title",
    bodies: ["Body one.", "Body two."],
  });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Slide Title\n\nBody one.\n\nBody two.");
  expect(
    document.sections.map((entry) => [entry.paragraph, entry.section]),
  ).toEqual([[0, "Slide Title"]]);
  // Unlike DocxConnector (whose one-locator-per-paragraph budget goes to
  // tables), every body paragraph on a slide — title included — carries
  // the SAME `slide` locator (ADR 0007 §7.2's budget spent on
  // distinguishing body from speaker notes instead).
  expect(document.locators.map((entry) => entry.locator)).toEqual([
    { kind: "slide", value: "1" },
    { kind: "slide", value: "1" },
    { kind: "slide", value: "1" },
  ]);
});

test("table becomes one paragraph with a slide locator", async () => {
  const raw = pptxBytes({
    bodies: ["Intro"],
    table: [
      ["A1", "B1"],
      ["A2", "B2"],
    ],
  });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Intro\n\nA1 | B1\nA2 | B2");
  expect(document.locators.map((entry) => entry.locator)).toEqual([
    { kind: "slide", value: "1" },
    { kind: "slide", value: "1" },
  ]);
});

test("speaker notes get their own locator after the slide body", async () => {
  const raw = pptxBytes({
    bodies: ["Body."],
    notes: ["Notes one.", "Notes two."],
  });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Body.\n\nNotes one.\n\nNotes two.");
  expect(
    document.locators.map((entry) => [entry.paragraph, entry.locator]),
  ).toEqual([
    [0, { kind: "slide", value: "1" }],
    [1, { kind: "speaker_notes", value: "1" }],
    [2, { kind: "speaker_notes", value: "1" }],
  ]);
});

test("multiple slides are numbered in document order", async () => {
  const presentation = blankPresentation();
  const slide1 = addTitleSlide(presentation, null);
  addBody(slide1, ["Slide one body."]);
  const slide2 = addTitleSlide(presentation, null);
  addBody(slide2, ["Slide two body."]);
  addNotes(slide2, ["Slide two notes."]);
  const path = await write("deck.pptx", saveBytes(presentation));
  const document = await new PptxConnector().read(path);

  expect(document.locators.map((entry) => entry.locator)).toEqual([
    { kind: "slide", value: "1" },
    { kind: "slide", value: "2" },
    { kind: "speaker_notes", value: "2" },
  ]);
});

test("group shape is walked recursively", async () => {
  const presentation = blankPresentation();
  const slide = addTitleSlide(presentation, null);
  addGroup(slide, ["Grouped one.", "Grouped two."]);
  const path = await write("deck.pptx", saveBytes(presentation));
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Grouped one.\n\nGrouped two.");
  expect(document.locators.map((entry) => entry.locator)).toEqual([
    { kind: "slide", value: "1" },
    { kind: "slide", value: "1" },
  ]);
});

test("empty title placeholder creates no section", async () => {
  const raw = pptxBytes({ title: "", bodies: ["Body."] });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.sections).toEqual([]);
  expect(document.text).toBe("Body.");
});

test("oversized title keeps the paragraph but creates no section", async () => {
  const hugeTitle = "x".repeat(MAX_SECTION_BYTES + 1);
  const raw = pptxBytes({ title: hugeTitle, bodies: ["Body."] });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.sections).toEqual([]);
  expect(document.text).toBe(`${hugeTitle}\n\nBody.`);
});

test("built text resplits into exactly the paragraphs the locators and sections index", async () => {
  // The invariant this connector's correctness rests on, mirroring
  // docx-connector.test.ts's own: re-running the server's own paragraph
  // splitter over `text` must yield exactly as many paragraphs as there
  // are, with locator/section indices still pointing at the right one.
  const presentation = blankPresentation();
  const slide1 = addTitleSlide(presentation, "First Slide");
  addBody(slide1, ["Body one.\nsecond line.", "Body two."]);
  addTable(slide1, [
    ["A1", "B1"],
    ["A2", "B2"],
  ]);
  addNotes(slide1, ["Notes."]);
  const slide2 = addTitleSlide(presentation, "Second Slide");
  addBody(slide2, ["More body."]);
  const path = await write("deck.pptx", saveBytes(presentation));
  const document = await new PptxConnector().read(path);

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

test("real content placeholder round trips", async () => {
  // Proves this connector reads an ordinary "Title and Content"
  // placeholder shape (real `p:ph idx="1"`), not only the plain textboxes
  // every other fixture in this module uses.
  const presentation = blankPresentation();
  addTitleAndContentSlide(presentation, {
    title: "Real Title",
    body: "Real body text.",
  });
  const path = await write("deck.pptx", saveBytes(presentation));
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Real Title\n\nReal body text.");
  expect(
    document.sections.map((entry) => [entry.paragraph, entry.section]),
  ).toEqual([[0, "Real Title"]]);
});

test.each([
  [(raw: Uint8Array) => withSmartart(raw), "SmartArt diagram"],
  [(raw: Uint8Array) => withOleObject(raw), "embedded object"],
])(
  "unreachable shape content is named partial_extraction (%#: %s)",
  async (build, expectedKind) => {
    let raw = pptxBytes({ bodies: ["Body."] });
    raw = build(raw);
    const path = await write("deck.pptx", raw);
    const document = await new PptxConnector().read(path);

    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      "partial_extraction",
    ]);
    expect(document.diagnostics[0]!.message).toContain(expectedKind);
  },
);

test("chart content is named partial_extraction", async () => {
  const raw = pptxBytes({ bodies: ["Body."], chart: true });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "partial_extraction",
  ]);
  expect(document.diagnostics[0]!.message).toContain("chart");
});

test("corrupt pptx variants are reported as corrupt", async () => {
  for (const kind of ["not_zip", "missing_part", "malformed_xml"] as const) {
    const path = await write(`broken-${kind}.pptx`, corruptPptx(kind));
    const document = await new PptxConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
      "corrupt",
    ]);
  }
});

test("encrypted pptx is reported encrypted without being opened", async () => {
  const path = await write("enc.pptx", encryptedPptx());
  const document = await new PptxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "encrypted",
  ]);
});

test("empty presentation is reported ocr_required", async () => {
  const raw = pptxBytes();
  const path = await write("empty.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "ocr_required",
  ]);
});

test("unsupported extension is reported without touching the filesystem", async () => {
  const path = await write("deck.ppt", new TextEncoder().encode("whatever"));
  const document = await new PptxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unsupported_format",
  ]);
  expect(document.metadata.contentType).toBeNull();
});

test("pptm extension is also unsupported", async () => {
  const path = await write("deck.pptm", pptxBytes({ bodies: ["x"] }));
  const document = await new PptxConnector().read(path);

  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unsupported_format",
  ]);
  expect(document.metadata.contentType).toBeNull();
});

test("other failure codes still claim the pptx MIME type", async () => {
  const document = await new PptxConnector().read(join(dir, "missing.pptx"));

  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unreadable",
  ]);
  expect(document.metadata.contentType).toBe(
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  );
});

test("oversized file is reported without parsing", async () => {
  const path = join(dir, "big.pptx");
  const handle = await open(path, "w");
  await handle.write(new Uint8Array([0]), 0, 1, 1024);
  await handle.close();
  const document = await new PptxConnector({ maxFileBytes: 1023 }).read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "content_too_large",
  ]);
});

test("oversized extracted text is reported content_too_large", async () => {
  const hugeParagraph = "x".repeat(MAX_PASSAGE_BYTES + 1);
  const raw = pptxBytes({ bodies: [hugeParagraph] });
  const path = await write("huge.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "content_too_large",
  ]);
});

test("oversized source id is reported without reading the file", async () => {
  const longReference = "x".repeat(1025) + ".pptx";
  const document = await new PptxConnector().read(longReference);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "source_id_too_long",
  ]);
});

test("missing file is reported unreadable", async () => {
  const document = await new PptxConnector().read(join(dir, "missing.pptx"));

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "unreadable",
  ]);
});

test("supports matches only pptx", () => {
  const connector = new PptxConnector();
  expect(connector.supports("a.pptx")).toBe(true);
  expect(connector.supports("a.PPTX")).toBe(true);
  expect(connector.supports("a.ppt")).toBe(false);
  expect(connector.supports("a.pptm")).toBe(false);
  expect(connector.supports("a.txt")).toBe(false);
});

test("parser identity is stamped into the fingerprint", async () => {
  const raw = pptxBytes({ bodies: ["Body."] });
  const path = await write("deck.pptx", raw);
  const connector = new PptxConnector();
  const document = await connector.read(path);

  expect(document.fingerprintInputs.parser).toBe(connector.parser);
  expect(document.fingerprintInputs.parserVersion).toBe(
    connector.parserVersion,
  );
});

test("fingerprint hashes the raw bytes and the effective options", async () => {
  const data = pptxBytes({ bodies: ["Body."] });
  const path = await write("deck.pptx", data);

  const defaultDocument = await new PptxConnector().read(path);
  const expectedSha256 = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", data as BufferSource)),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
  expect(defaultDocument.fingerprintInputs.rawContentSha256).toBe(
    expectedSha256,
  );

  const otherDocument = await new PptxConnector({ extractTitles: false }).read(
    path,
  );
  expect(otherDocument.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
  expect(otherDocument.fingerprintInputs.rawContentSha256).toBe(
    defaultDocument.fingerprintInputs.rawContentSha256,
  );
});

test("extractTitles false keeps paragraph text but drops sections", async () => {
  const raw = pptxBytes({ title: "Title.", bodies: ["Body."] });
  const path = await write("deck.pptx", raw);

  const defaultDocument = await new PptxConnector().read(path);
  const document = await new PptxConnector({ extractTitles: false }).read(path);

  expect(document.sections).toEqual([]);
  expect(document.text).toBe(defaultDocument.text);
  expect(document.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
});

test("extractSpeakerNotes false drops notes entirely", async () => {
  const raw = pptxBytes({ bodies: ["Body."], notes: ["Notes."] });
  const path = await write("deck.pptx", raw);

  const document = await new PptxConnector({ extractSpeakerNotes: false }).read(
    path,
  );

  expect(document.text).toBe("Body.");
  expect(document.locators).toEqual([
    new LocatorEntry({ paragraph: 0, locator: { kind: "slide", value: "1" } }),
  ]);
});

test("extractTables false drops the table entirely", async () => {
  // Unlike DocxConnector (whose extractTables=false keeps a table's text
  // but drops only its locator, since the text still needs *some*
  // paragraph to live in), a PPTX table has no locator of its own to
  // drop independently — dropping the locator would leave an ordinary
  // body paragraph indistinguishable from one, so this connector drops
  // the table's paragraph entirely instead.
  const raw = pptxBytes({ bodies: ["Intro"], table: [["A1", "B1"]] });
  const path = await write("deck.pptx", raw);

  const defaultDocument = await new PptxConnector().read(path);
  const document = await new PptxConnector({ extractTables: false }).read(path);

  expect(defaultDocument.text).toContain("A1 | B1");
  expect(document.text).toBe("Intro");
  expect(document.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
});

test("metadata title prefers core properties over first slide title", async () => {
  const presentation = blankPresentation();
  setCoreTitle(presentation, "Explicit Title");
  const slide = addTitleSlide(presentation, "Slide Title");
  addBody(slide, ["Body."]);
  const path = await write("deck.pptx", saveBytes(presentation));
  const document = await new PptxConnector().read(path);

  expect(document.metadata.title).toBe("Explicit Title");
});

test("metadata title falls back to first slide title", async () => {
  const raw = pptxBytes({ title: "Slide Title", bodies: ["Body."] });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.metadata.title).toBe("Slide Title");
});

test("metadata content_type is the OOXML presentation MIME type", async () => {
  const raw = pptxBytes({ bodies: ["Body."] });
  const path = await write("deck.pptx", raw);
  const document = await new PptxConnector().read(path);

  expect(document.metadata.contentType).toBe(
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
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
    const { PptxConnector: IsolatedPptxConnector } =
      await import("../../src/ingest-connectors/pptx.js");
    const path = await write("deck.pptx", pptxBytes({ bodies: ["Body."] }));

    await expect(new IsolatedPptxConnector().read(path)).rejects.toThrow(
      /fflate and fast-xml-parser/,
    );
  });
});

test("a real zip bomb is refused content_too_large by measured inflation (issue #737)", async () => {
  // The PPTX twin of the docx bomb test: same shared unzipWithinCap
  // wiring, pinned per connector since each maps the refusal to its own
  // content_too_large diagnostic.
  const { zipSync } = await import("fflate");
  const inflated = new Uint8Array(256 * 1024 * 1024 + 1024);
  const bomb = zipSync({ "ppt/slides/slide1.xml": inflated }, { level: 1 });
  const path = await write("bomb.pptx", bomb);
  const document = await new PptxConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual([
    "content_too_large",
  ]);
});
