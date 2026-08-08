/**
 * Ingest-connector round-trip tests against the real server binary — the
 * mechanical TS port of the Python suite's own connector tests in
 * `test_e2e.py` (issue #415 parity port, #347-#353). Kept as a sibling of
 * `e2e.test.ts` (rather than folded into it) since the retriever/ingester
 * suite there and this one exercise unrelated surfaces and each already
 * spawns/tears down its own server instance.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FakeListChatModel } from "@langchain/core/utils/testing";
import type { Locator } from "taguru";
import { Taguru } from "taguru";
import { serverBinary, spawnServer, type SpawnedServer } from "taguru/testing";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { FilesystemCheckpointStore } from "../../src/checkpoints.js";
import { ingestConnectorDocument } from "../../src/ingest-connectors/bridge.js";
import { ConnectorDocument, LocatorEntry } from "../../src/ingest-connectors/document.js";
import { DocxConnector } from "../../src/ingest-connectors/docx.js";
import { HtmlConnector } from "../../src/ingest-connectors/html.js";
import { openObjectStore } from "../../src/ingest-connectors/objectstore.js";
import { PdfConnector } from "../../src/ingest-connectors/pdf.js";
import { PptxConnector } from "../../src/ingest-connectors/pptx.js";
import { syncReferences } from "../../src/ingest-connectors/references.js";
import { syncObjectStorage } from "../../src/ingest-connectors/s3.js";
import { TextFileConnector } from "../../src/ingest-connectors/text.js";
import { TaguruIngester } from "../../src/ingest.js";
import { docxBytes, heading, para, table } from "../docx.js";
import { serve, type Route } from "../httpd.js";
import { textPdf } from "../pdfs.js";
import { addBody, addNotes, addTitleSlide, blankPresentation, saveBytes } from "../pptx.js";
import { writeBucket } from "../s3.js";

const TOKEN = "lc-ts-test-token-connectors";

let server: SpawnedServer;
let client: Taguru;
let workDir: string;

const EMPTY_LLM_ANSWER = JSON.stringify({ associations: [], aliases: [], questions: [] });

function emptyLlm(times = 1): FakeListChatModel {
  return new FakeListChatModel({ responses: new Array<string>(times).fill(EMPTY_LLM_ANSWER) });
}

/** Deletes `context` if it still exists — guaranteed cleanup even when an
 * assertion above throws, otherwise a failed run leaves the context behind
 * on the shared real server and the next run's own `create_context: true`
 * collides with it, turning one failure into a second, unrelated one. */
async function dropContext(context: string): Promise<void> {
  if (await client.contexts.exists(context)) {
    await client.contexts.delete(context);
  }
}

/** A fresh subdirectory of `workDir`, one per test, so sibling tests never
 * share fixture files or checkpoint stores. */
function tmpSubdir(name: string): string {
  return mkdtempSync(join(workDir, `${name}-`));
}

beforeAll(async () => {
  workDir = mkdtempSync(join(tmpdir(), "taguru-connectors-e2e-"));
  server = await spawnServer(serverBinary(), { TAGURU_API_TOKEN: TOKEN });
  client = new Taguru({ base_url: server.baseUrl, api_key: TOKEN });
  await client.waitUntilReady({ timeout: 30 });
});

afterAll(() => {
  server.stop();
  rmSync(workDir, { recursive: true, force: true });
});

describe("ingest connectors (real server)", () => {
  it("connector document round-trips sections and locators to citations (#347)", async () => {
    const dir = tmpSubdir("aizome");
    const path = join(dir, "aizome.md");
    const text = "# 藍染工房\n\n藍染めの技法を伝える工房である。\n\n代表作は暖簾である。\n";
    writeFileSync(path, text, "utf-8");

    const base = await new TextFileConnector().read(path);
    expect(base.diagnostics).toHaveLength(0);
    expect(base.sections).toHaveLength(1);
    expect(base.sections[0]!.paragraph).toBe(0);
    expect(base.sections[0]!.section).toBe("藍染工房");

    // TextFileConnector never produces a locator (no natural page/slide in
    // .md/.txt) — a synthetic one exercises the same wire path #348-#351's
    // real page/slide/sheet connectors will use.
    const document = new ConnectorDocument({
      source: base.source,
      text: base.text,
      sections: base.sections,
      locators: [new LocatorEntry({ paragraph: 2, locator: { kind: "page", value: "1" } })],
      metadata: base.metadata,
      fingerprintInputs: base.fingerprintInputs,
      diagnostics: base.diagnostics,
    });

    const ingester = new TaguruIngester({
      context: "aizome",
      llm: emptyLlm(),
      client,
      create_context: true,
      context_description: "connector round trip (issue #347)",
    });
    try {
      const outcome = await ingestConnectorDocument(ingester, document);
      expect(outcome.ok).toBe(true);
      expect(outcome.sections_stored).toBe(1);
      expect(outcome.locators_stored).toBe(1);

      const ctx = client.context("aizome");
      const headingCitation = await ctx.citePassage(document.source, 0);
      expect(headingCitation.section).toBe("藍染工房");

      const locatedCitation = await ctx.citePassage(document.source, 2);
      expect(locatedCitation.locator).toEqual({ kind: "page", value: "1" } satisfies Locator);
    } finally {
      await dropContext("aizome");
    }
  });

  it("PDF connector document round-trips page locators to citations (#348)", async () => {
    const dir = tmpSubdir("indigo");
    const path = join(dir, "workshop.pdf");

    // ASCII only: the hand-built test PDF (tests/pdfs.ts) uses the base14
    // Helvetica font with no embedded CJK glyphs/ToUnicode CMap, unlike
    // this suite's other fixtures.
    writeFileSync(
      path,
      textPdf([
        "Indigo Workshop\n\nThe workshop preserves indigo dyeing techniques.",
        "Its best known product is a noren curtain.",
      ]),
    );

    const document = await new PdfConnector().read(path);
    expect(document.diagnostics).toHaveLength(0);
    expect(document.locators).toHaveLength(3);
    expect(document.locators[2]!.locator).toEqual({ kind: "page", value: "2" } satisfies Locator);

    const ingester = new TaguruIngester({
      context: "indigo-workshop",
      llm: emptyLlm(),
      client,
      create_context: true,
      context_description: "PDF connector round trip (issue #348)",
    });
    try {
      const outcome = await ingestConnectorDocument(ingester, document);
      expect(outcome.ok).toBe(true);
      expect(outcome.locators_stored).toBe(document.locators.length);

      const ctx = client.context("indigo-workshop");
      const locatedCitation = await ctx.citePassage(document.source, 2);
      expect(locatedCitation.locator).toEqual({ kind: "page", value: "2" } satisfies Locator);
    } finally {
      await dropContext("indigo-workshop");
    }
  });

  it("HTML connector document round-trips fragment locators to citations (#349)", async () => {
    const body = Buffer.from(
      `<html><head><title>Weaving Studio</title></head><body><main>
      <h1 id="top">Weaving Studio</h1>
      <p>The studio preserves traditional loom weaving.</p>
      <h2 id="products">Products</h2>
      <p>Its best known product is an obi sash.</p>
      </main></body></html>`,
    );

    const route: Route = { body };
    const httpd = await serve({ "/studio": route });
    let document: ConnectorDocument;
    try {
      document = await new HtmlConnector({ allowPrivateNetworks: true }).read(`${httpd.baseUrl}/studio`);
    } finally {
      await httpd.close();
    }

    expect(document.diagnostics).toHaveLength(0);
    expect(document.sections.map((s) => [s.paragraph, s.section])).toEqual([
      [0, "Weaving Studio"],
      [2, "Weaving Studio > Products"],
    ]);
    expect(document.locators[2]!.locator).toEqual({ kind: "fragment", value: "products" } satisfies Locator);

    const ingester = new TaguruIngester({
      context: "weaving-studio",
      llm: emptyLlm(),
      client,
      create_context: true,
      context_description: "HTML connector round trip (issue #349)",
    });
    try {
      const outcome = await ingestConnectorDocument(ingester, document);
      expect(outcome.ok).toBe(true);
      expect(outcome.sections_stored).toBe(document.sections.length);
      expect(outcome.locators_stored).toBe(document.locators.length);

      const ctx = client.context("weaving-studio");
      const headingCitation = await ctx.citePassage(document.source, 2);
      expect(headingCitation.section).toBe("Weaving Studio > Products");
      expect(headingCitation.locator).toEqual({ kind: "fragment", value: "products" } satisfies Locator);
    } finally {
      await dropContext("weaving-studio");
    }
  });

  it("DOCX connector document round-trips table locators to citations (#350)", async () => {
    const dir = tmpSubdir("pottery");
    const path = join(dir, "pottery.docx");

    const bodyXml =
      heading("Pottery Studio", 1) +
      para("The studio preserves traditional raku firing.") +
      heading("Products", 2) +
      table([
        ["Item", "Price"],
        ["Tea bowl", "3000"],
      ]);
    writeFileSync(path, docxBytes(bodyXml));

    const document = await new DocxConnector().read(path);
    expect(document.diagnostics).toHaveLength(0);
    expect(document.sections.map((s) => [s.paragraph, s.section])).toEqual([
      [0, "Pottery Studio"],
      [2, "Pottery Studio > Products"],
    ]);
    expect(document.locators[0]!.locator).toEqual({ kind: "table", value: "1" } satisfies Locator);

    const ingester = new TaguruIngester({
      context: "pottery-studio",
      llm: emptyLlm(),
      client,
      create_context: true,
      context_description: "DOCX connector round trip (issue #350)",
    });
    try {
      const outcome = await ingestConnectorDocument(ingester, document);
      expect(outcome.ok).toBe(true);
      expect(outcome.sections_stored).toBe(document.sections.length);
      expect(outcome.locators_stored).toBe(document.locators.length);

      const ctx = client.context("pottery-studio");
      const tableParagraph = document.locators[0]!.paragraph;
      const tableCitation = await ctx.citePassage(document.source, tableParagraph);
      expect(tableCitation.section).toBe("Pottery Studio > Products");
      expect(tableCitation.locator).toEqual({ kind: "table", value: "1" } satisfies Locator);
    } finally {
      await dropContext("pottery-studio");
    }
  });

  it("PPTX connector document round-trips slide and notes locators to citations (#352)", async () => {
    const dir = tmpSubdir("glassblowing");
    const path = join(dir, "glassblowing.pptx");

    const presentation = blankPresentation();
    const slide = addTitleSlide(presentation, "Glassblowing Studio");
    addBody(slide, ["The studio preserves traditional glassblowing."]);
    addNotes(slide, ["Mention the kiln temperature on stage."]);
    writeFileSync(path, saveBytes(presentation));

    const document = await new PptxConnector().read(path);
    expect(document.diagnostics).toHaveLength(0);
    expect(document.sections.map((s) => [s.paragraph, s.section])).toEqual([[0, "Glassblowing Studio"]]);
    expect(document.locators.map((entry) => [entry.paragraph, entry.locator])).toEqual([
      [0, { kind: "slide", value: "1" }],
      [1, { kind: "slide", value: "1" }],
      [2, { kind: "speaker_notes", value: "1" }],
    ]);

    const ingester = new TaguruIngester({
      context: "glassblowing-studio",
      llm: emptyLlm(),
      client,
      create_context: true,
      context_description: "PPTX connector round trip (issue #352)",
    });
    try {
      const outcome = await ingestConnectorDocument(ingester, document);
      expect(outcome.ok).toBe(true);
      expect(outcome.sections_stored).toBe(document.sections.length);
      expect(outcome.locators_stored).toBe(document.locators.length);

      const ctx = client.context("glassblowing-studio");
      const bodyCitation = await ctx.citePassage(document.source, 1);
      expect(bodyCitation.section).toBe("Glassblowing Studio");
      expect(bodyCitation.locator).toEqual({ kind: "slide", value: "1" } satisfies Locator);

      const notesCitation = await ctx.citePassage(document.source, 2);
      expect(notesCitation.locator).toEqual({ kind: "speaker_notes", value: "1" } satisfies Locator);
    } finally {
      await dropContext("glassblowing-studio");
    }
  });

  it("S3 connector syncs a file bucket of PDF/HTML/DOCX/PPTX to citations (#351)", async () => {
    const dir = tmpSubdir("ceramics-s3");
    const bucket = join(dir, "bucket");
    mkdirSync(bucket);

    const pdfBytes = textPdf(["Ceramics Workshop\n\nThe workshop preserves raku firing techniques."]);
    const htmlBytes = Buffer.from(
      '<html><head><title>Ceramics</title></head><body><main>' +
        '<h1 id="top">Ceramics</h1><p>A page about ceramics.</p>' +
        "</main></body></html>",
    );
    const docxBytesValue = docxBytes(
      heading("Catalog", 1) +
        table([
          ["Item", "Price"],
          ["Bowl", "3000"],
        ]),
    );
    const pptxPresentation = blankPresentation();
    const pptxSlide = addTitleSlide(pptxPresentation, "Kiln Schedule");
    addBody(pptxSlide, ["Fire the raku kiln at 1000C."]);

    await writeBucket(bucket, {
      "report.pdf": pdfBytes,
      "page.html": htmlBytes,
      "catalog.docx": docxBytesValue,
      "schedule.pptx": saveBytes(pptxPresentation),
    });

    // Independently parsed from the SAME bytes, purely to learn the
    // locators/paragraph indices this run's own citations must match —
    // the actual documents synced below are produced inside S3Connector,
    // not these.
    const pdfDocument = await new PdfConnector().read(join(bucket, "report.pdf"));
    const htmlDocument = await new HtmlConnector().read(join(bucket, "page.html"));
    const docxDocument = await new DocxConnector().read(join(bucket, "catalog.docx"));
    const pptxDocument = await new PptxConnector().read(join(bucket, "schedule.pptx"));

    const [store, prefix] = await openObjectStore(`file://${bucket}`);
    const checkpoints = new FilesystemCheckpointStore(join(dir, "checkpoints"));
    const ingester = new TaguruIngester({
      context: "ceramics-s3",
      llm: emptyLlm(),
      client,
      create_context: true,
      context_description: "S3 connector round trip (issue #351)",
    });
    try {
      const report = await syncObjectStorage(store, prefix, { ingester, checkpoints });
      expect(report.imported).toBe(4);
      expect(report.failed).toBe(0);

      const ctx = client.context("ceramics-s3");
      const pdfSource = `${store.baseUri}/report.pdf`;
      const htmlSource = `${store.baseUri}/page.html`;
      const docxSource = `${store.baseUri}/catalog.docx`;
      const pptxSource = `${store.baseUri}/schedule.pptx`;

      const pdfCitation = await ctx.citePassage(pdfSource, pdfDocument.locators[0]!.paragraph);
      expect(pdfCitation.locator).toEqual(pdfDocument.locators[0]!.locator);

      const htmlCitation = await ctx.citePassage(htmlSource, htmlDocument.locators[0]!.paragraph);
      expect(htmlCitation.locator).toEqual(htmlDocument.locators[0]!.locator);

      const docxCitation = await ctx.citePassage(docxSource, docxDocument.locators[0]!.paragraph);
      expect(docxCitation.locator).toEqual(docxDocument.locators[0]!.locator);

      const pptxCitation = await ctx.citePassage(pptxSource, pptxDocument.locators[0]!.paragraph);
      expect(pptxCitation.locator).toEqual(pptxDocument.locators[0]!.locator);

      // A second pass with nothing changed on disk: both checkpoint
      // layers hit, so nothing is re-fetched or re-ingested.
      const second = await syncObjectStorage(store, prefix, { ingester, checkpoints });
      expect(second.unchanged).toBe(4);
      expect(second.imported).toBe(0);

      unlinkSync(join(bucket, "catalog.docx"));
      const third = await syncObjectStorage(store, prefix, {
        ingester,
        checkpoints,
        deletionPolicy: "retract",
      });
      expect(third.deletedDetected).toBe(1);
      expect(third.retracted).toBe(1);
      expect((await ctx.lookupPassages([docxSource])).missing).toContain(docxSource);
    } finally {
      await dropContext("ceramics-s3");
    }
  });

  it("sync_references end to end with an events sidecar (#353)", async () => {
    const dir = tmpSubdir("ceramics-references");
    const reference = join(dir, "manual.md");
    writeFileSync(reference, "The kiln reaches 1000C during raku firing.", "utf-8");
    const checkpoints = new FilesystemCheckpointStore(join(dir, "checkpoints"));
    const eventsPath = join(dir, "sync.jsonl");

    const ingester = new TaguruIngester({
      context: "ceramics-references",
      llm: emptyLlm(4),
      client,
      create_context: true,
      context_description: "sync_references round trip (issue #353)",
    });
    try {
      const report = await syncReferences([reference], { ingester, checkpoints, eventsOut: eventsPath });
      expect(report.imported).toBe(1);
      expect(report.failed).toBe(0);
      expect(report.eventsPath).toBe(eventsPath);

      const onDiskPhases = readFileSync(eventsPath, "utf-8")
        .trimEnd()
        .split("\n")
        .map((line) => (JSON.parse(line) as { phase: string }).phase);
      expect(onDiskPhases).toEqual(["discovered", "parsed", "extracted", "imported"]);

      const ctx = client.context("ceramics-references");
      expect(Object.keys((await ctx.lookupPassages([reference])).passages)).toContain(reference);

      const second = await syncReferences([reference], { ingester, checkpoints });
      expect(second.unchanged).toBe(1);
      expect(second.imported).toBe(0);
    } finally {
      await dropContext("ceramics-references");
    }
  });
});
