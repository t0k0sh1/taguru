/**
 * TextFileConnector — the reference .md/.txt connector (ADR 0007, issue
 * #347). Mirrors the Python suite's test_text_connector.py case-for-case
 * (TypeScript parity: issue #415).
 */

import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../../src/extract.js";
import { TextFileConnector } from "../../src/ingest-connectors/text.js";

function write(dir: string, name: string, content: string, encoding: BufferEncoding = "utf-8"): string {
  const path = join(dir, name);
  writeFileSync(path, Buffer.from(content, encoding));
  return path;
}

describe("TextFileConnector", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "taguru-text-connector-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("markdown headings become sections at splitParagraphs indices", async () => {
    const text = "# Title\n\nIntro paragraph.\n\n## Section Two\n\nMore text.\n";
    const path = write(dir, "doc.md", text);
    const document = await new TextFileConnector().read(path);

    expect(document.text).toBe(text);
    const paragraphs = splitParagraphs(document.text);
    expect(paragraphs[0]).toBe("# Title");
    expect(paragraphs[2]).toBe("## Section Two");
    expect(document.sections.map((entry) => [entry.paragraph, entry.section])).toEqual([
      [0, "Title"],
      [2, "Section Two"],
    ]);
    expect(document.metadata.title).toBe("Title");
    expect(document.metadata.contentType).toBe("text/markdown");
    expect(document.diagnostics).toEqual([]);
  });

  it("txt never carries sections even with heading-like lines", async () => {
    const path = write(dir, "doc.txt", "# Not a heading\n\nJust text.\n");
    const document = await new TextFileConnector().read(path);
    expect(document.sections).toEqual([]);
    expect(document.metadata.title).toBeNull();
    expect(document.metadata.contentType).toBe("text/plain");
  });

  it("extractHeadings false disables section extraction for markdown", async () => {
    const path = write(dir, "doc.md", "# Title\n\nBody.\n");
    const document = await new TextFileConnector({ extractHeadings: false }).read(path);
    expect(document.sections).toEqual([]);
    expect(document.metadata.title).toBeNull();
  });

  it("a heading sharing a paragraph with body text is not recognized", async () => {
    // No blank line separates the heading from the next line, so the two
    // fold into one multi-line paragraph — the connector never guesses
    // where a heading ends without a blank-line boundary to trust (a
    // documented limitation, not a bug).
    const path = write(dir, "doc.md", "# Title\nSame paragraph body.\n");
    const document = await new TextFileConnector().read(path);
    expect(document.sections).toEqual([]);
  });

  it("closing hash sequence is stripped only when preceded by space", async () => {
    const path = write(dir, "doc.md", "## Heading ##\n\n# C#\n");
    const document = await new TextFileConnector().read(path);
    const labels = document.sections.map((entry) => entry.section);
    expect(labels).toContain("Heading");
    expect(labels).toContain("C#"); // trailing '#' with no preceding space is literal text
  });

  it("leading BOM is stripped and does not shift paragraph indices", async () => {
    const path = write(dir, "doc.md", "﻿# Title\n\nBody.\n");
    const document = await new TextFileConnector().read(path);
    expect(document.text.startsWith("# Title")).toBe(true);
    expect(document.sections[0]?.paragraph).toBe(0);
    expect(document.sections[0]?.section).toBe("Title");
  });

  it("CRLF line endings still yield a single-line heading paragraph", async () => {
    const path = write(dir, "doc.md", "# Title\r\n\r\nBody line.\r\n");
    const document = await new TextFileConnector().read(path);
    expect(document.sections.length).toBe(1);
    expect(document.sections[0]?.paragraph).toBe(0);
    expect(document.sections[0]?.section).toBe("Title");
  });

  it("raw_content_sha256 is deterministic and independent of extension", async () => {
    const content = "# Title\n\nBody.\n";
    const mdPath = write(dir, "doc.md", content);
    const txtPath = write(dir, "doc.txt", content);
    const mdDocument = await new TextFileConnector().read(mdPath);
    const txtDocument = await new TextFileConnector().read(txtPath);
    const expected = createHash("sha256").update(content, "utf-8").digest("hex");
    expect(mdDocument.fingerprintInputs.rawContentSha256).toBe(expected);
    expect(txtDocument.fingerprintInputs.rawContentSha256).toBe(expected);
  });

  it("missing file is reported unreadable", async () => {
    const document = await new TextFileConnector().read(join(dir, "missing.md"));
    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });

  it("unsupported extension is reported without touching the filesystem", async () => {
    const path = write(dir, "doc.pdf", "whatever");
    const document = await new TextFileConnector().read(path);
    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
  });

  it("oversized file is reported content_too_large", async () => {
    const path = join(dir, "big.txt");
    writeFileSync(path, Buffer.alloc(MAX_PASSAGE_BYTES + 1));
    const document = await new TextFileConnector().read(path);
    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["content_too_large"]);
  });

  it("non-UTF-8 bytes are reported corrupt", async () => {
    const path = join(dir, "bad.txt");
    writeFileSync(path, Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from(" not utf-8")]));
    const document = await new TextFileConnector().read(path);
    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["corrupt"]);
  });

  it("oversized source id is reported without reading the file", async () => {
    // The length check runs before any filesystem access, so a fabricated
    // path that is never actually created still exercises it cleanly.
    const longReference = "x".repeat(1025) + ".txt";
    const document = await new TextFileConnector().read(longReference);
    expect(document.text).toBe("");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["source_id_too_long"]);
  });

  it("supports matches only .md and .txt", () => {
    const connector = new TextFileConnector();
    expect(connector.supports("a.md")).toBe(true);
    expect(connector.supports("a.TXT")).toBe(true);
    expect(connector.supports("a.pdf")).toBe(false);
  });

  it("parser identity is stamped into the fingerprint", async () => {
    const path = write(dir, "doc.txt", "body");
    const connector = new TextFileConnector();
    const document = await connector.read(path);
    expect(document.fingerprintInputs.parser).toBe(connector.parser);
    expect(document.fingerprintInputs.parserVersion).toBe(connector.parserVersion);
  });
});
