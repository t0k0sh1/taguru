/**
 * PdfConnector — the .pdf connector (ADR 0007 §7/§8/§10, issue #348).
 * Mirrors the Python suite's test_pdf_connector.py case-for-case
 * (TypeScript parity: issue #415), with two adaptations noted at their own
 * test — pypdf-specific per-page monkeypatching has no direct pdfjs-dist
 * equivalent, so those two tests instead mock the dynamically-imported
 * `pdfjs-dist` module itself (`vi.doMock`) to inject the same per-page
 * failure, proving the identical CONTRACT (diagnostic code, surviving
 * locators).
 */

import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type * as PdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../../src/extract.js";
import { PdfConnector } from "../../src/ingest-connectors/pdf.js";
import { corruptPdf, encryptedPdf, scannedPdf, textPdf } from "../pdfs.js";

function write(dir: string, name: string, data: Uint8Array): string {
  const path = join(dir, name);
  writeFileSync(path, data);
  return path;
}

describe("PdfConnector", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "taguru-pdf-connector-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("page boundaries become one page locator per paragraph", async () => {
    const path = write(
      dir,
      "doc.pdf",
      textPdf(["First page, one paragraph.", "Second page.\n\nA second paragraph."]),
    );
    const document = await new PdfConnector().read(path);

    expect(document.diagnostics).toEqual([]);
    expect(document.locators.map((entry) => entry.paragraph)).toEqual([0, 1, 2]);
    expect(document.locators.map((entry) => entry.locator)).toEqual([
      { kind: "page", value: "1" },
      { kind: "page", value: "2" },
      { kind: "page", value: "2" },
    ]);
  });

  it("built text resplits into exactly the paragraphs the locators index", async () => {
    // The invariant the connector's correctness rests on: re-running the
    // server's own paragraph splitter over the `text` this connector
    // produces must yield exactly as many paragraphs as there are locator
    // entries, each at its own index — otherwise a locator silently points
    // at the wrong paragraph once the batch reaches `taguru import`.
    const path = write(
      dir,
      "doc.pdf",
      textPdf([
        "Page one heading.\n\nPage one body.",
        "Page two, single paragraph, two lines.\nsecond line.",
        "Page three.",
      ]),
    );
    const document = await new PdfConnector().read(path);

    const resplit = splitParagraphs(document.text);
    expect(resplit.length).toBe(document.locators.length);
    document.locators.forEach((entry, index) => {
      expect(entry.paragraph).toBe(index);
    });
    // The joined text must round-trip: no paragraph gained or lost a break.
    expect(resplit.join("\n\n")).toBe(document.text);
  });

  it("outline entries become sections at the matching paragraph", async () => {
    const path = write(
      dir,
      "doc.pdf",
      textPdf(
        [
          "Introduction text for the whole document.",
          "Chapter One body.\n\nMore chapter one content.",
        ],
        { outline: [["Introduction", 0], ["Chapter One", 1]] },
      ),
    );
    const document = await new PdfConnector().read(path);

    expect(document.sections.map((entry) => [entry.paragraph, entry.section])).toEqual([
      [0, "Introduction"],
      [1, "Chapter One"],
    ]);
    expect(document.metadata.title).toBe("Introduction");
  });

  it("extractOutline false disables sections and title", async () => {
    const path = write(
      dir,
      "doc.pdf",
      textPdf(["Introduction text for the whole document."], {
        outline: [["Introduction", 0]],
      }),
    );
    const document = await new PdfConnector({ extractOutline: false }).read(path);

    expect(document.sections).toEqual([]);
    expect(document.metadata.title).toBeNull();
  });

  it("outline entry on an unusable page falls forward to the next page", async () => {
    const path = write(
      dir,
      "doc.pdf",
      textPdf(["", "Real content starts here."], { outline: [["Front Matter", 0]] }),
    );
    const document = await new PdfConnector({ minCharsPerPage: 1 }).read(path);

    // Page 1 (index 0) is empty and diagnosed ocr_required; the bookmark
    // that names it lands on the first paragraph that actually exists.
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
    expect(document.sections.map((entry) => [entry.paragraph, entry.section])).toEqual([
      [0, "Front Matter"],
    ]);
  });

  it("scanned pdf is ocr_required with empty text", async () => {
    const path = write(dir, "scan.pdf", scannedPdf(2));
    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.locators).toEqual([]);
    expect(document.sections).toEqual([]);
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
    expect(document.diagnostics[0]!.message).toContain("1");
    expect(document.diagnostics[0]!.message).toContain("2");
  });

  it("partially scanned pdf keeps the readable pages and names the rest", async () => {
    const path = write(
      dir,
      "mixed.pdf",
      textPdf([
        "Readable page one with plenty of extractable text.",
        "",
        "Readable page three with plenty of extractable text.",
      ]),
    );
    const document = await new PdfConnector().read(path);

    expect(document.text).not.toBe("");
    expect(document.locators.map((entry) => entry.locator.value)).toEqual(["1", "3"]);
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
    expect(document.diagnostics[0]!.message.endsWith("page(s) 2")).toBe(true);
  });

  // -- adapted: pypdf-specific per-page monkeypatching (see module docstring) --

  describe("per-page extraction failure (adapted from pypdf monkeypatching)", () => {
    afterEach(() => {
      vi.doUnmock("pdfjs-dist/legacy/build/pdf.mjs");
    });

    /**
     * Wraps a real `PDFDocumentProxy` so that `getPage(n).getTextContent()`
     * rejects for every page number in `flakyPages`, while every other page
     * (and every other document/page method) is untouched real
     * pdfjs-dist behavior — the TS/vitest equivalent of the Python suite's
     * `monkeypatch.setattr(PageObject, "extract_text", flaky)`, since
     * pdfjs-dist's own `PDFPageProxy` is not part of its public export
     * surface and so cannot be monkeypatched the same way.
     */
    function wrapFlakyPages(
      doc: PdfjsLib.PDFDocumentProxy,
      flakyPages: ReadonlySet<number>,
    ): PdfjsLib.PDFDocumentProxy {
      return new Proxy(doc, {
        get(target, prop, receiver) {
          if (prop === "getPage") {
            return async (pageNumber: number) => {
              const page = await target.getPage(pageNumber);
              if (!flakyPages.has(pageNumber)) {
                return page;
              }
              return new Proxy(page, {
                get(pageTarget, pageProp) {
                  if (pageProp === "getTextContent") {
                    return () => Promise.reject(new Error("simulated per-page decode failure"));
                  }
                  const value = Reflect.get(pageTarget as object, pageProp);
                  return typeof value === "function" ? value.bind(pageTarget) : value;
                },
              });
            };
          }
          const value = Reflect.get(target as object, prop, receiver);
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
    }

    async function mockFlakyPages(flakyPages: ReadonlySet<number>): Promise<void> {
      vi.doMock("pdfjs-dist/legacy/build/pdf.mjs", async () => {
        const actual =
          await vi.importActual<typeof PdfjsLib>("pdfjs-dist/legacy/build/pdf.mjs");
        return {
          ...actual,
          getDocument: (params: Parameters<typeof actual.getDocument>[0]) => ({
            promise: actual
              .getDocument(params)
              .promise.then((doc) => wrapFlakyPages(doc, flakyPages)),
          }),
        };
      });
    }

    it("page extraction failure is reported as partial_extraction", async () => {
      await mockFlakyPages(new Set([2]));
      const path = write(
        dir,
        "doc.pdf",
        textPdf([
          "Page one has plenty of readable text.",
          "Page two has plenty of readable text.",
          "Page three has plenty of readable text.",
        ]),
      );

      const document = await new PdfConnector().read(path);

      expect(document.locators.map((entry) => entry.locator.value)).toEqual(["1", "3"]);
      expect(document.diagnostics.map((d) => d.code)).toEqual(["partial_extraction"]);
      expect(document.diagnostics[0]!.message.endsWith("page(s) 2")).toBe(true);
    });

    it("all pages failing to extract is reported as corrupt, not ocr_required", async () => {
      await mockFlakyPages(new Set([1, 2]));
      const path = write(dir, "doc.pdf", textPdf(["Page one.", "Page two."]));

      const document = await new PdfConnector().read(path);

      expect(document.text).toBe("");
      expect(document.diagnostics.map((d) => d.code)).toEqual(["corrupt"]);
    });
  });

  it("encrypted pdf with a user password is reported as encrypted", async () => {
    const path = write(dir, "enc.pdf", encryptedPdf(["Secret text."], { userPassword: "secret" }));
    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["encrypted"]);
  });

  it("owner password only pdf still extracts", async () => {
    const path = write(
      dir,
      "restricted.pdf",
      encryptedPdf(["Readable despite owner restrictions."], {
        userPassword: "",
        ownerPassword: "owner-secret",
      }),
    );
    const document = await new PdfConnector().read(path);

    expect(document.diagnostics).toEqual([]);
    expect(document.text).toContain("Readable despite owner restrictions.");
  });

  it("corrupt pdf is reported as corrupt", async () => {
    const path = write(dir, "broken.pdf", corruptPdf());
    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["corrupt"]);
  });

  it("empty pdf with zero pages is reported as corrupt", async () => {
    const path = write(dir, "empty.pdf", textPdf([]));
    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["corrupt"]);
  });

  it("unsupported extension is reported without touching the filesystem", async () => {
    const path = write(dir, "doc.txt", new TextEncoder().encode("whatever"));
    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
  });

  it("oversized file is reported without parsing", async () => {
    const path = join(dir, "big.pdf");
    writeFileSync(path, new Uint8Array(1024));
    const document = await new PdfConnector({ maxFileBytes: 1023 }).read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["content_too_large"]);
  });

  it("oversized extracted text is reported content_too_large", async () => {
    const hugePage = "x".repeat(MAX_PASSAGE_BYTES + 1);
    const path = write(dir, "huge.pdf", textPdf([hugePage]));
    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["content_too_large"]);
  });

  it("oversized source id is reported without reading the file", async () => {
    const longReference = "x".repeat(1025) + ".pdf";
    const document = await new PdfConnector().read(longReference);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["source_id_too_long"]);
  });

  it("missing file is reported unreadable", async () => {
    const document = await new PdfConnector().read(join(dir, "missing.pdf"));

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });

  it("supports matches only pdf", () => {
    const connector = new PdfConnector();
    expect(connector.supports("a.pdf")).toBe(true);
    expect(connector.supports("a.PDF")).toBe(true);
    expect(connector.supports("a.txt")).toBe(false);
  });

  it("parser identity is stamped into the fingerprint", async () => {
    const path = write(dir, "doc.pdf", textPdf(["Body."]));
    const connector = new PdfConnector();
    const document = await connector.read(path);

    expect(document.fingerprintInputs.parser).toBe(connector.parser);
    expect(document.fingerprintInputs.parserVersion).toBe(connector.parserVersion);
  });

  it("fingerprint hashes the raw bytes and the effective options", async () => {
    const data = textPdf(["Body."]);
    const path = write(dir, "doc.pdf", data);

    const defaultDocument = await new PdfConnector().read(path);
    expect(defaultDocument.fingerprintInputs.rawContentSha256).toBe(
      createHash("sha256").update(data).digest("hex"),
    );

    const otherThresholdDocument = await new PdfConnector({ minCharsPerPage: 5 }).read(path);
    expect(otherThresholdDocument.fingerprintInputs.parseOptionsDigest).not.toBe(
      defaultDocument.fingerprintInputs.parseOptionsDigest,
    );
    // Changing an option never touches the raw-bytes hash — the two
    // fingerprint fields answer independent questions (ADR 0007 §5/§6.2).
    expect(otherThresholdDocument.fingerprintInputs.rawContentSha256).toBe(
      defaultDocument.fingerprintInputs.rawContentSha256,
    );
  });

  // Adapted (issue #415): the Python twin raises `ImportError` synchronously
  // at `PdfConnector` construction, since `import pypdf` happens at module
  // load. A dynamic `import()` cannot be awaited inside a TS constructor, so
  // `PdfConnector` defers the equivalent check to the first call to `read()`
  // instead (see pdf.ts's module docstring) — this test mocks the
  // dynamically-imported module itself to simulate "not installed" rather
  // than actually uninstalling the devDependency.
  it("missing pdfjs-dist raises a clear error on first use", async () => {
    vi.doMock("pdfjs-dist/legacy/build/pdf.mjs", () => {
      throw new Error("Cannot find package 'pdfjs-dist'");
    });
    try {
      const path = write(dir, "doc.pdf", textPdf(["Body."]));
      await expect(new PdfConnector().read(path)).rejects.toThrow(/pdfjs-dist/);
    } finally {
      vi.doUnmock("pdfjs-dist/legacy/build/pdf.mjs");
    }
  });
});
