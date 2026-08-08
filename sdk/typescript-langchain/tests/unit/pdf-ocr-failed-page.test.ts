/**
 * A page whose `getTextContent()` REJECTS is exactly as unusable to the
 * PDF connector as one that decoded to nothing — a configured OcrAdapter
 * must be offered a chance to recover it too, not only pages that merely
 * decoded empty (the port of the Python twin's pypdf-monkeypatch tests).
 * Own file: the `vi.mock` below patches pdfjs-dist module-wide, which
 * must not leak into the ordinary pdf/ocr suites.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test, vi } from "vitest";

import {
  OcrRecoveredUnit,
  OcrResult,
  type OcrAdapter,
  type OcrRequest,
} from "../../src/ingest-connectors/ocr.js";
import { PdfConnector } from "../../src/ingest-connectors/pdf.js";
import { textPdf } from "../pdfs.js";

/** The `recovered`-map subset of ocr-adapter.test.ts's own fake. */
class FakeOcrAdapter implements OcrAdapter {
  readonly name = "fake-ocr";
  readonly version = "1.0.0";
  private readonly recovered: Map<string, string>;

  constructor(options: { recovered: Record<string, string> }) {
    this.recovered = new Map(Object.entries(options.recovered));
  }

  async recognize(request: OcrRequest): Promise<OcrResult> {
    const units = request.locators
      .filter((locator) => this.recovered.has(locator.value))
      .map(
        (locator) => new OcrRecoveredUnit({ locator, text: this.recovered.get(locator.value)! }),
      );
    return new OcrResult({ units, diagnostics: [] });
  }
}

vi.mock("pdfjs-dist/legacy/build/pdf.mjs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("pdfjs-dist/legacy/build/pdf.mjs")>();
  return {
    ...actual,
    getDocument: (params: Parameters<typeof actual.getDocument>[0]) => {
      const task = actual.getDocument(params);
      const promise = task.promise.then((doc) => {
        const originalGetPage = doc.getPage.bind(doc);
        (doc as { getPage: typeof doc.getPage }).getPage = async (pageNumber: number) => {
          const page = await originalGetPage(pageNumber);
          if (pageNumber === 2) {
            (page as { getTextContent: unknown }).getTextContent = async () => {
              throw new Error("simulated per-page decode failure");
            };
          }
          return page;
        };
        return doc;
      });
      return { promise, destroy: () => task.destroy() };
    },
  };
});

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "pdf-ocr-failed-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function write(name: string, bytes: Uint8Array): string {
  const path = join(dir, name);
  writeFileSync(path, bytes);
  return path;
}

const PAGES = [
  "Page one has plenty of readable text.",
  "Page two has plenty of readable text.",
];

test("an extraction-raising page is recovered when an adapter is configured", async () => {
  const path = write("doc.pdf", textPdf(PAGES));
  const adapter = new FakeOcrAdapter({
    recovered: { "2": "Recovered failed page, plenty of characters." },
  });

  const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toContain("Recovered failed page, plenty of characters.");
});

test("a raising page the adapter cannot recover stays partial_extraction", async () => {
  // An adapter's own failure to recover must leave the page reported
  // exactly as partial_extraction, as if no adapter had been configured.
  const path = write("doc.pdf", textPdf(PAGES));
  const adapter = new FakeOcrAdapter({ recovered: {} });

  const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

  expect(document.diagnostics.map((d) => d.code)).toEqual(["partial_extraction"]);
  expect(document.text).toContain("Page one has plenty of readable text.");
});
