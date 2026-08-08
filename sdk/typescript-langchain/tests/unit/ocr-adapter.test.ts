/**
 * The external OCR adapter boundary (ADR 0007 §10, issue #352) — exercised
 * through `PdfConnector` (pdf.ts), the one connector that calls out to a
 * configured adapter today. A fake `OcrAdapter` (ocr.ts) stands in for a
 * real OCR engine throughout, per §10's own "no OCR engine ships" posture —
 * these tests prove the CALL-OUT contract, never a real recognition
 * result. Mirrors the Python suite's test_ocr_adapter.py case-for-case
 * (TypeScript parity: issue #415).
 */

import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { Diagnostic } from "../../src/ingest-connectors/document.js";
import { OcrRecoveredUnit, OcrRequest, OcrResult, type OcrAdapter } from "../../src/ingest-connectors/ocr.js";
import { PdfConnector } from "../../src/ingest-connectors/pdf.js";
import { scannedPdf, textPdf } from "../pdfs.js";

/**
 * A minimal, in-memory stand-in for a real OCR engine. `recovered` maps a
 * requested locator's own `value` (e.g. a page number, as a string) to the
 * text to hand back for it — any requested locator with no entry here is
 * simply not recovered, the same "adapter recovered nothing for this page"
 * case a real engine can hit. `fixedResult`, when set, bypasses `recovered`
 * entirely and is returned as-is regardless of what was requested — for
 * exercising `PdfConnector`'s own defense against a misbehaving adapter
 * (returning a locator nobody asked for, or text too thin to count).
 * `raiseError`, when set, is thrown instead of returning anything, standing
 * in for the adapter's own failure.
 */
class FakeOcrAdapter implements OcrAdapter {
  readonly name: string;
  readonly version: string;
  readonly receivedRequests: OcrRequest[] = [];
  private readonly recovered: Map<string, string>;
  private readonly diagnostics: readonly Diagnostic[];
  private readonly fixedResult: OcrResult | null;
  private readonly raiseError: unknown;

  constructor(options?: {
    name?: string;
    version?: string;
    recovered?: Record<string, string>;
    diagnostics?: readonly Diagnostic[];
    fixedResult?: OcrResult | null;
    raiseError?: unknown;
  }) {
    this.name = options?.name ?? "fake-ocr";
    this.version = options?.version ?? "1.0.0";
    this.recovered = new Map(Object.entries(options?.recovered ?? {}));
    this.diagnostics = options?.diagnostics ?? [];
    this.fixedResult = options?.fixedResult ?? null;
    this.raiseError = options?.raiseError;
  }

  async recognize(request: OcrRequest): Promise<OcrResult> {
    this.receivedRequests.push(request);
    if (this.raiseError !== undefined) {
      throw this.raiseError;
    }
    if (this.fixedResult !== null) {
      return this.fixedResult;
    }
    const units = request.locators
      .filter((locator) => this.recovered.has(locator.value))
      .map((locator) => new OcrRecoveredUnit({ locator, text: this.recovered.get(locator.value)! }));
    return new OcrResult({ units, diagnostics: this.diagnostics });
  }
}

describe("OcrAdapter call-out (PdfConnector)", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "taguru-ocr-adapter-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function write(name: string, data: Uint8Array): string {
    const path = join(dir, name);
    writeFileSync(path, data);
    return path;
  }

  it("adapter not invoked when no page is unusable", async () => {
    // A configured adapter is called out to only when there is something
    // to recover — never on every parse regardless of need.
    const adapter = new FakeOcrAdapter({ raiseError: new Error("should not have been called") });
    const path = write("text.pdf", textPdf(["Plenty of real, extractable page text."]));

    const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

    expect(document.diagnostics).toEqual([]);
    expect(adapter.receivedRequests).toEqual([]);
  });

  it("absent adapter leaves ocr_required exactly as before", async () => {
    // Regression: no ocrAdapter (the default) must behave identically to
    // PdfConnector before it could accept one at all.
    const path = write("scan.pdf", scannedPdf(1));

    const document = await new PdfConnector().read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
  });

  it("full recovery clears ocr_required and uses page locators", async () => {
    const adapter = new FakeOcrAdapter({
      recovered: {
        "1": "Recovered page one text, plenty of characters.",
        "2": "Recovered page two text, plenty of characters.",
      },
    });
    const path = write("scan.pdf", scannedPdf(2));

    const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

    expect(document.diagnostics).toEqual([]);
    expect(document.text).toBe(
      "Recovered page one text, plenty of characters.\n\n" +
        "Recovered page two text, plenty of characters.",
    );
    expect(document.locators.map((entry) => [entry.paragraph, entry.locator])).toEqual([
      [0, { kind: "page", value: "1" }],
      [1, { kind: "page", value: "2" }],
    ]);
    // The adapter was offered exactly the unusable pages, the raw bytes,
    // and this document's own source id/content type — never asked to
    // recover a page this connector already had usable text for.
    expect(adapter.receivedRequests.length).toBe(1);
    const request = adapter.receivedRequests[0]!;
    expect(request.locators).toEqual([
      { kind: "page", value: "1" },
      { kind: "page", value: "2" },
    ]);
    expect(request.content).toEqual(scannedPdf(2));
    expect(request.contentType).toBe("application/pdf");
    expect(request.source).toBe(document.source);
  });

  it("partial recovery leaves the remaining pages named", async () => {
    const adapter = new FakeOcrAdapter({
      recovered: { "2": "Recovered middle page, plenty of characters." },
    });
    const path = write("scan.pdf", scannedPdf(3));

    const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

    expect(document.text).toBe("Recovered middle page, plenty of characters.");
    expect(document.locators.map((entry) => entry.locator)).toEqual([
      { kind: "page", value: "2" },
    ]);
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
    expect(document.diagnostics[0]!.message).toContain("page(s) 1, 3");
  });

  it("adapter exception leaves the page ocr_required and names the failure", async () => {
    const adapter = new FakeOcrAdapter({ raiseError: new Error("engine unavailable") });
    const path = write("scan.pdf", scannedPdf(1));

    const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
    const message = document.diagnostics[0]!.message;
    expect(message).toContain("page(s) 1");
    expect(message).toContain("OCR adapter failed: engine unavailable");
  });

  it("adapter returning an unrequested or too-thin unit is discarded", async () => {
    // A misbehaving (or merely imperfect) adapter must not corrupt the
    // document: a locator nobody asked about, and text too thin to clear
    // the same minCharsPerPage bar every other page's text must clear, are
    // both silently dropped rather than spliced in.
    const fixed = new OcrResult({
      units: [
        new OcrRecoveredUnit({
          locator: { kind: "page", value: "99" },
          text: "Text for a page nobody asked about, plenty long enough.",
        }),
        new OcrRecoveredUnit({ locator: { kind: "page", value: "1" }, text: "x" }),
      ],
    });
    const adapter = new FakeOcrAdapter({ fixedResult: fixed });
    const path = write("scan.pdf", scannedPdf(1));

    const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
    expect(document.diagnostics[0]!.message).not.toContain("OCR adapter failed");
  });

  it("OcrResult diagnostics propagate to the document", async () => {
    const path = write("scan.pdf", scannedPdf(1));
    const source = path;
    const adapter = new FakeOcrAdapter({
      recovered: { "1": "Recovered page text, plenty of characters." },
      diagnostics: [
        new Diagnostic({ code: "partial_extraction", message: "adapter skipped a region", source }),
      ],
    });

    const document = await new PdfConnector({ ocrAdapter: adapter }).read(path);

    expect(document.text).toBe("Recovered page text, plenty of characters.");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["partial_extraction"]);
    expect(document.diagnostics[0]!.message).toBe("adapter skipped a region");
  });

  it("configuring an adapter changes the digest but not the raw hash", async () => {
    const data = textPdf(["Some ordinary page text."]);
    const path = write("doc.pdf", data);

    const defaultDocument = await new PdfConnector().read(path);
    const adapterDocument = await new PdfConnector({ ocrAdapter: new FakeOcrAdapter() }).read(path);

    expect(adapterDocument.fingerprintInputs.parseOptionsDigest).not.toBe(
      defaultDocument.fingerprintInputs.parseOptionsDigest,
    );
    const expectedHash = createHash("sha256").update(data).digest("hex");
    expect(adapterDocument.fingerprintInputs.rawContentSha256).toBe(
      defaultDocument.fingerprintInputs.rawContentSha256,
    );
    expect(defaultDocument.fingerprintInputs.rawContentSha256).toBe(expectedHash);
  });

  it("swapping adapter identity changes the digest", async () => {
    // Not just "an adapter is configured," but the adapter's own declared
    // identity — swapping engines (or versions) must invalidate a §6.3
    // checkpoint's prior skip decision just as surely as configuring one
    // for the first time does.
    const data = textPdf(["Some ordinary page text."]);
    const path = write("doc.pdf", data);

    const first = await new PdfConnector({ ocrAdapter: new FakeOcrAdapter({ name: "engine-a" }) }).read(
      path,
    );
    const second = await new PdfConnector({ ocrAdapter: new FakeOcrAdapter({ name: "engine-b" }) }).read(
      path,
    );

    expect(first.fingerprintInputs.parseOptionsDigest).not.toBe(
      second.fingerprintInputs.parseOptionsDigest,
    );
  });
});
