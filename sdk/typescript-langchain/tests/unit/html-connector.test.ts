/**
 * HtmlConnector — the local-file half of the `.html`/`.htm`/`.xhtml`
 * connector (ADR 0007 §7/§8, issue #349). URL-fetch behavior is
 * `html-connector-fetch.test.ts`'s own file. TypeScript parity: issue
 * #415.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { MAX_PASSAGE_BYTES, splitParagraphs } from "../../src/extract.js";
import { HtmlConnector } from "../../src/ingest-connectors/html.js";
import { MAX_SECTION_BYTES } from "../../src/ingest-connectors/document.js";

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "html-connector-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function write(name: string, html: string | Buffer): string {
  const path = join(dir, name);
  if (typeof html === "string") {
    writeFileSync(path, html, "utf-8");
  } else {
    writeFileSync(path, html);
  }
  return path;
}

async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

test("boilerplate is removed and main is preferred", async () => {
  const path = write(
    "doc.html",
    `<html><head><title>T</title></head><body>
        <nav>Site nav</nav>
        <header>Site banner</header>
        <main><p>Real content.</p></main>
        <aside>Related links</aside>
        <footer>Site footer</footer>
        <script>doStuff();</script>
        <style>.x{color:red}</style>
        </body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toBe("Real content.");
});

test("headings become breadcrumb sections and fragment locators", async () => {
  const path = write(
    "doc.html",
    `<html><body><main>
        <h1 id="top">Guide</h1>
        <p>Intro.</p>
        <h2 id="install">Installation</h2>
        <p>Step one.</p>
        <p>Step two.</p>
        </main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  expect(document.sections.map((s) => [s.paragraph, s.section])).toEqual([
    [0, "Guide"],
    [2, "Guide > Installation"],
  ]);
  expect(document.locators.map((entry) => [entry.paragraph, entry.locator])).toEqual([
    [0, { kind: "fragment", value: "top" }],
    [1, { kind: "fragment", value: "top" }],
    [2, { kind: "fragment", value: "install" }],
    [3, { kind: "fragment", value: "install" }],
    [4, { kind: "fragment", value: "install" }],
  ]);
});

test("heading without id falls back to the nearest enclosing id", async () => {
  const path = write(
    "doc.html",
    `<html><body><main>
        <section id="sec-1"><h2>No id of its own</h2><p>Body text.</p></section>
        </main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  expect(new Set(document.locators.map((entry) => entry.locator.value))).toEqual(new Set(["sec-1"]));
});

test("extractHeadings false disables sections and locators", async () => {
  const path = write(
    "doc.html",
    `<html><body><main><h1 id="top">Guide</h1><p>Body.</p></main></body></html>`,
  );
  const document = await new HtmlConnector({ extractHeadings: false }).read(path);

  expect(document.sections).toEqual([]);
  expect(document.locators).toEqual([]);
  expect(document.text).toBe("Guide\n\nBody.");
});

test("title falls back to first h1 when no title tag", async () => {
  const path = write("doc.html", `<html><body><main><h1>Fallback Title</h1></main></body></html>`);
  const document = await new HtmlConnector().read(path);

  expect(document.metadata.title).toBe("Fallback Title");
});

test("canonical link is honored only when absolute", async () => {
  const absolute = write(
    "abs.html",
    `<html><head><link rel="canonical" href="https://example.com/guide">
        </head><body><main><p>Text.</p></main></body></html>`,
  );
  const relative = write(
    "rel.html",
    `<html><head><link rel="canonical" href="/guide">
        </head><body><main><p>Text.</p></main></body></html>`,
  );

  expect((await new HtmlConnector().read(absolute)).metadata.canonicalUrl).toBe(
    "https://example.com/guide",
  );
  expect((await new HtmlConnector().read(relative)).metadata.canonicalUrl).toBeNull();
});

test("br is an interior line break not a paragraph boundary", async () => {
  const path = write("doc.html", `<html><body><p>Line one<br>Line two</p></body></html>`);
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("Line one\nLine two");
  expect(splitParagraphs(document.text)).toEqual(["Line one\nLine two"]);
});

test("double br never creates a false paragraph boundary", async () => {
  const path = write("doc.html", `<html><body><p>Before<br><br>After</p></body></html>`);
  const document = await new HtmlConnector().read(path);

  // A literal blank line here would silently register as a second
  // paragraph on resplit, offsetting every locator/section index after it
  // — the exact hazard `sanitizeParagraphText` exists to rule out.
  expect(splitParagraphs(document.text)).toEqual([document.text]);
});

test("pre preserves whitespace but collapses an internal blank line", async () => {
  const path = write("doc.html", "<html><body><pre>line one\n\n\nline two</pre></body></html>");
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("line one\nline two");
});

test("table becomes one paragraph with rows and cells joined", async () => {
  const path = write(
    "doc.html",
    `<html><body><main><table>
        <tr><th>Name</th><th>Value</th></tr>
        <tr><td>a</td><td>1</td></tr>
        </table></main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("Name | Value\na | 1");
});

test("nested table content is dropped not glued into the outer cell", async () => {
  const path = write(
    "doc.html",
    `<html><body><main><table>
        <tr><td>outer<table><tr><td>inner</td></tr></table></td></tr>
        </table></main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  // Nested tables are unsupported: the inner table's cell never leaks
  // into the outer cell's own text, and is never independently emitted
  // either.
  expect(document.text).toBe("outer");
});

test("resplit invariant holds for a mixed document", async () => {
  // The invariant this connector's correctness rests on: re-running the
  // server's own paragraph splitter over `text` must reproduce exactly
  // the paragraphs `locators`/`sections` index — otherwise a citation
  // silently points at the wrong paragraph once the batch reaches
  // `taguru import`.
  const path = write(
    "doc.html",
    `<html><body><main>
        <h1 id="top">Title</h1>
        <p>First<br>paragraph.</p>
        <pre>pre\n\nblock</pre>
        <table><tr><td>c1</td><td>c2</td></tr></table>
        <h2 id="next">Next</h2>
        <p>Last paragraph.</p>
        </main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  const resplit = splitParagraphs(document.text);
  expect(resplit.join("\n\n")).toBe(document.text);
  for (const entry of document.locators) {
    expect(entry.paragraph).toBeGreaterThanOrEqual(0);
    expect(entry.paragraph).toBeLessThan(resplit.length);
  }
  for (const section of document.sections) {
    expect(section.paragraph).toBeGreaterThanOrEqual(0);
    expect(section.paragraph).toBeLessThan(resplit.length);
  }
});

test("hidden aria-hidden and denylisted role elements are dropped", async () => {
  const path = write(
    "doc.html",
    `<html><body><main>
        <p>Visible.</p>
        <div hidden>Hidden attr.</div>
        <div aria-hidden="true">Aria hidden.</div>
        <div role="navigation">Role nav.</div>
        <div role="search">Role search.</div>
        </main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("Visible.");
});

test("header and footer are kept inside article but dropped at body level", async () => {
  const withArticle = write(
    "article.html",
    `<html><body><article><header><h1>Byline header</h1></header>
        <p>Body.</p></article></body></html>`,
  );
  const withoutScope = write(
    "plain.html",
    `<html><body><header>Site banner</header><p>Body.</p></body></html>`,
  );

  expect((await new HtmlConnector().read(withArticle)).text).toContain("Byline header");
  expect((await new HtmlConnector().read(withoutScope)).text).toBe("Body.");
});

test("malformed markup degrades quietly instead of raising", async () => {
  // The stray unmatched </div> is ignored rather than raised.
  const path = write("doc.html", "<html><body><p>Hello</div><p>World</p></body></html>");
  const document = await new HtmlConnector().read(path);

  expect(document.diagnostics).toEqual([]);
  expect(document.text).toContain("Hello");
  expect(document.text).toContain("World");
});

test("iframe content is excluded and reported as partial_extraction", async () => {
  const path = write(
    "doc.html",
    `<html><body><main><p>Real text.</p>
        <iframe src="https://example.com/embed"></iframe>
        </main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("Real text.");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["partial_extraction"]);
});

test("empty body after boilerplate removal is ocr_required", async () => {
  const path = write("doc.html", `<html><body><nav>Nav only</nav><script>x()</script></body></html>`);
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["ocr_required"]);
});

test("BOM is stripped from the first paragraph", async () => {
  const path = join(dir, "doc.html");
  writeFileSync(path, Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from("<html><body><p>Text.</p></body></html>", "utf-8")]));
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("Text.");
});

test("meta charset is honored for non-utf8 content", async () => {
  const path = join(dir, "doc.html");
  const html = '<html><head><meta charset="iso-8859-1"></head><body><p>café</p></body></html>';
  writeFileSync(path, Buffer.from(html, "latin1"));
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("café");
});

test("undecodable bytes under the declared charset are reported corrupt", async () => {
  const path = join(dir, "doc.html");
  const header = Buffer.from('<html><head><meta charset="utf-8"></head><body><p>', "utf-8");
  const tail = Buffer.from("</p></body></html>", "utf-8");
  writeFileSync(path, Buffer.concat([header, Buffer.from([0xff, 0xfe]), tail]));
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["corrupt"]);
});

test("section breadcrumb over cap drops the outermost crumbs first", async () => {
  // At the cap alone (fits as the h1's own single-crumb section); combined
  // with "Child" via the separator it would exceed MAX_SECTION_BYTES,
  // forcing the breadcrumb builder to drop this outermost crumb first.
  const longAncestor = "A".repeat(MAX_SECTION_BYTES);
  const path = write(
    "doc.html",
    `<html><body><main>
        <h1 id="a">${longAncestor}</h1>
        <h2 id="b">Child</h2>
        <p>Body.</p>
        </main></body></html>`,
  );
  const document = await new HtmlConnector().read(path);

  const childSection = document.sections.find((s) => s.section.endsWith("Child"))!;
  expect(childSection.section).toBe("Child"); // the long ancestor crumb was dropped, not truncated
  expect(new TextEncoder().encode(childSection.section).length).toBeLessThanOrEqual(MAX_SECTION_BYTES);
});

test("unsupported extension is reported without touching the filesystem", async () => {
  const path = write("doc.txt", "whatever");
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
});

test("oversized file is reported without parsing", async () => {
  const path = join(dir, "big.html");
  const buffer = Buffer.alloc(1025);
  writeFileSync(path, buffer);
  const document = await new HtmlConnector({ maxFileBytes: 1023 }).read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["content_too_large"]);
});

test("oversized extracted text is reported content_too_large", async () => {
  const huge = "x".repeat(MAX_PASSAGE_BYTES + 1);
  const path = write("huge.html", `<html><body><p>${huge}</p></body></html>`);
  const document = await new HtmlConnector().read(path);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["content_too_large"]);
});

test("oversized source id is reported without reading the file", async () => {
  const longReference = "x".repeat(1025) + ".html";
  const document = await new HtmlConnector().read(longReference);

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["source_id_too_long"]);
});

test("missing file is reported unreadable", async () => {
  const document = await new HtmlConnector().read(join(dir, "missing.html"));

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
});

test("supports matches html suffixes and http urls only", () => {
  const connector = new HtmlConnector();
  expect(connector.supports("a.html")).toBe(true);
  expect(connector.supports("a.HTM")).toBe(true);
  expect(connector.supports("a.xhtml")).toBe(true);
  expect(connector.supports("a.txt")).toBe(false);
  expect(connector.supports("https://example.com/guide")).toBe(true);
  expect(connector.supports("http://example.com/guide")).toBe(true);
  expect(connector.supports("ftp://example.com/guide.html")).toBe(false);
});

test("unsupported scheme is reported without touching the filesystem", async () => {
  const document = await new HtmlConnector().read("ftp://example.com/doc.html");

  expect(document.text).toBe("");
  expect(document.diagnostics.map((d) => d.code)).toEqual(["unsupported_format"]);
});

describe("Windows drive-letter path is treated as local, not a URL scheme", () => {
  test("supports() accepts both slash styles", () => {
    // `new URL(...)`/a scheme-prefix regex would read `C:\docs\a.html`'s
    // scheme as `"c"` — indistinguishable from a real (if unusual)
    // single-letter URL scheme by scheme alone, so this must be
    // special-cased before scheme dispatch, not inferred from scheme
    // length.
    const connector = new HtmlConnector();
    expect(connector.supports("C:\\docs\\a.html")).toBe(true);
    expect(connector.supports("C:/docs/a.html")).toBe(true);
  });

  test("a nonexistent drive path reads as unreadable, not unsupported_format", async () => {
    const connector = new HtmlConnector();
    const document = await connector.read("C:\\does\\not\\exist.html");
    expect(document.diagnostics.map((d) => d.code)).toEqual(["unreadable"]);
  });
});

test("parser identity is stamped into the fingerprint", async () => {
  const path = write("doc.html", "<html><body><p>Body.</p></body></html>");
  const connector = new HtmlConnector();
  const document = await connector.read(path);

  expect(document.fingerprintInputs.parser).toBe(connector.parser);
  expect(document.fingerprintInputs.parserVersion).toBe(connector.parserVersion);
});

test("fingerprint hashes the raw bytes and the effective options", async () => {
  const data = "<html><body><p>Body.</p></body></html>";
  const path = write("doc.html", data);

  const defaultDocument = await new HtmlConnector().read(path);
  expect(defaultDocument.fingerprintInputs.rawContentSha256).toBe(await sha256Hex(data));

  const otherDocument = await new HtmlConnector({ headingSeparator: " / " }).read(path);
  expect(otherDocument.fingerprintInputs.parseOptionsDigest).not.toBe(
    defaultDocument.fingerprintInputs.parseOptionsDigest,
  );
  // Changing an option never touches the raw-bytes hash — the two
  // fingerprint fields answer independent questions (ADR 0007 §5/§6.2).
  expect(otherDocument.fingerprintInputs.rawContentSha256).toBe(
    defaultDocument.fingerprintInputs.rawContentSha256,
  );
});

test("timeout and userAgent are excluded from the options digest", async () => {
  const path = write("doc.html", "<html><body><p>Body.</p></body></html>");

  const a = await new HtmlConnector({ timeout: 5.0, userAgent: "a" }).read(path);
  const b = await new HtmlConnector({ timeout: 99.0, userAgent: "b" }).read(path);

  expect(a.fingerprintInputs.parseOptionsDigest).toBe(b.fingerprintInputs.parseOptionsDigest);
});
