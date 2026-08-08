/**
 * Minimal PDF byte-builders for `pdf-connector.test.ts` and
 * `ocr-adapter.test.ts` (issue #348's TypeScript parity port, issue #415)
 * — hand-assembled rather than shipped as binary fixtures, the same
 * "synthesize at test time" convention every other connector test in this
 * repo already follows. The mechanical (though NOT byte-for-byte) mirror
 * of the Python twin's `tests/_pdfs.py`.
 *
 * Unlike the Python twin — which embeds a whole page's text, `\n`/`\n\n`
 * included, as ONE literal-string `Tj` operand (legal for pypdf, whose
 * `extract_text()` reads a literal string's embedded control characters
 * back verbatim) — pdfjs-dist's own `getTextContent()` does not: a `\n`
 * byte inside a PDF content-stream string has no glyph in a base font's
 * encoding and is silently dropped rather than round-tripped (confirmed
 * empirically against pdfjs-dist 6.2.108 at port time). So each PAGE here
 * is instead built as one `Tj` operator PER LINE, each positioned via its
 * own `Td` at an explicit vertical offset: consecutive lines advance by
 * one line height, and a blank line (an empty string between two `\n`s in
 * the input) advances by an EXTRA line height with no `Tj` of its own —
 * exactly the vertical-gap shape `assemblePageText` (`../src/ingest-
 * connectors/pdf.ts`) reconstructs `\n` (single gap) and `\n\n`
 * (double-or-more gap) from. This is a deliberate deviation from the
 * Python builder's layout (task per issue #415: "port the builder so it
 * produces PDFs pdfjs-dist can parse ... mirroring what `_pdfs.py`
 * builds") — the two builders differ in HOW a page's content stream is
 * laid out, but `textPdf(pages)` still produces exactly `pages[i]` back
 * out of `PdfConnector` for page `i`, the same contract `_pdfs.py`'s own
 * docstring promises.
 */

import { createHash, randomBytes } from "node:crypto";

const LINE_HEIGHT = 14;
const FONT_SIZE = 12;

// Every page uses this (deliberately oversized, well past any realistic
// print page) `/MediaBox` — see MAX_LINE_CHARS's own comment below for why:
// pdfjs-dist's `getTextContent()` clips a `Tj` run that would render
// outside the page's box, in BOTH axes, so a page sized for actual paper
// would silently drop test content that runs past its edge. Nothing here
// is ever rendered, so an oversized box costs nothing.
const PAGE_WIDTH = 20000;
const PAGE_HEIGHT = 100000;
const START_Y = PAGE_HEIGHT - 72;

function latin1(text: string): Uint8Array {
  const bytes = new Uint8Array(text.length);
  for (let index = 0; index < text.length; index += 1) {
    bytes[index] = text.charCodeAt(index) & 0xff;
  }
  return bytes;
}

function concatBytes(...parts: readonly Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function escapeLiteral(text: string): string {
  return text.replaceAll("\\", "\\\\").replaceAll("(", "\\(").replaceAll(")", "\\)");
}

/**
 * The smallest correct PDF container: sequential objects, each recorded at
 * its own byte offset, followed by a plain (uncompressed) xref table and a
 * one-entry trailer — enough for pdfjs-dist to open, and nothing
 * pdfjs-dist-specific.
 */
class PdfBuilder {
  private readonly objects = new Map<number, Uint8Array>();
  private next = 1;

  alloc(): number {
    const number = this.next;
    this.next += 1;
    return number;
  }

  add(number: number, body: Uint8Array): void {
    this.objects.set(number, body);
  }

  render(root: number, extraTrailer = ""): Uint8Array {
    const header = latin1("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");
    const chunks: Uint8Array[] = [header];
    const offsets = new Map<number, number>();
    let length = header.length;
    const numbers = [...this.objects.keys()].sort((a, b) => a - b);
    for (const number of numbers) {
      offsets.set(number, length);
      const objHeader = latin1(`${number} 0 obj\n`);
      const body = this.objects.get(number)!;
      const objFooter = latin1("\nendobj\n");
      chunks.push(objHeader, body, objFooter);
      length += objHeader.length + body.length + objFooter.length;
    }
    const xrefOffset = length;
    const highest = numbers.length > 0 ? Math.max(...numbers) : 0;
    let xref = `xref\n0 ${highest + 1}\n0000000000 65535 f \n`;
    for (let number = 1; number <= highest; number += 1) {
      const offset = offsets.get(number);
      xref +=
        offset !== undefined
          ? `${String(offset).padStart(10, "0")} 00000 n \n`
          : "0000000000 00000 f \n";
    }
    xref += `trailer\n<< /Size ${highest + 1} /Root ${root} 0 R${extraTrailer} >>\n`;
    xref += `startxref\n${xrefOffset}\n%%EOF`;
    chunks.push(latin1(xref));
    return concatBytes(...chunks);
  }
}

function streamObject(body: Uint8Array): Uint8Array {
  return concatBytes(
    latin1(`<< /Length ${body.length} >>\nstream\n`),
    body,
    latin1("\nendstream"),
  );
}

/**
 * One page's content stream: one `BT ... Tj ET` per non-blank line of
 * `pageText.split("\n")`, each positioned `LINE_HEIGHT` below the previous
 * — a blank line advances the position an EXTRA `LINE_HEIGHT` with no `Tj`
 * of its own. See the module docstring for why this (rather than one `Tj`
 * per page) is how a paragraph break survives the round trip through
 * pdfjs-dist's `getTextContent()`.
 */
// A logical line longer than this is wrapped across multiple
// same-paragraph physical `Tj` lines (a single, not double, `LINE_HEIGHT`
// gap between the pieces — see `assemblePageText`, pdf.ts) so it stays
// within `PAGE_WIDTH` regardless of how long the ORIGINAL (pre-wrap) line
// was — confirmed empirically at port time that pdfjs-dist's own
// `getTextContent()` clips a `Tj` run past the page's horizontal bound.
// `PAGE_HEIGHT` is sized to comfortably fit every wrapped piece of the
// longest line any test here builds (the `content_too_large` fixture's
// multi-megabyte single line) without ALSO running past the page's
// vertical bound, which pdfjs-dist clips identically. Tests that care
// about EXACT text content never use a line long enough to wrap, so the
// extra line breaks this introduces are never observed.
const MAX_LINE_CHARS = 1500;

function wrapLine(line: string, maxChars: number): string[] {
  if (line.length <= maxChars) {
    return [line];
  }
  const chunks: string[] = [];
  for (let offset = 0; offset < line.length; offset += maxChars) {
    chunks.push(line.slice(offset, offset + maxChars));
  }
  return chunks;
}

function pageContentStream(pageText: string): Uint8Array {
  const lines = pageText.split("\n");
  let y = START_Y;
  let firstLine = true;
  const ops: string[] = [];
  for (const line of lines) {
    if (line === "") {
      y -= LINE_HEIGHT;
      continue;
    }
    for (const chunk of wrapLine(line, MAX_LINE_CHARS)) {
      if (!firstLine) {
        y -= LINE_HEIGHT;
      }
      firstLine = false;
      ops.push(`BT /F1 ${FONT_SIZE} Tf 72 ${y} Td (${escapeLiteral(chunk)}) Tj ET`);
    }
  }
  return latin1(ops.join("\n"));
}

/**
 * Builds the shared catalog/pages/font/content/outline object graph both
 * `textPdf` and `encryptedPdf` need, without adding the content streams'
 * bodies yet — `encryptedPdf` needs each one's PLAINTEXT bytes (to compute
 * its own per-object RC4 key and encrypt it) before it is added to the
 * builder; `textPdf` adds it unencrypted as-is.
 */
function buildStructure(
  pages: readonly string[],
  outline: ReadonlyArray<readonly [string, number]>,
): { builder: PdfBuilder; catalog: number; contents: Array<{ number: number; plaintext: Uint8Array }> } {
  const builder = new PdfBuilder();
  const catalog = builder.alloc();
  const pagesNum = builder.alloc();
  const font = builder.alloc();
  builder.add(font, latin1("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"));

  const pageNums: number[] = [];
  const contents: Array<{ number: number; plaintext: Uint8Array }> = [];
  for (const pageText of pages) {
    const pageNum = builder.alloc();
    const contentNum = builder.alloc();
    pageNums.push(pageNum);
    contents.push({ number: contentNum, plaintext: pageContentStream(pageText) });
    builder.add(
      pageNum,
      latin1(
        `<< /Type /Page /Parent ${pagesNum} 0 R /Resources << /Font << /F1 ${font} 0 R >> >> ` +
          `/MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}] /Contents ${contentNum} 0 R >>`,
      ),
    );
  }
  const kids = pageNums.map((number) => `${number} 0 R`).join(" ");
  builder.add(pagesNum, latin1(`<< /Type /Pages /Kids [${kids}] /Count ${pageNums.length} >>`));

  let outlinesRef = "";
  if (outline.length > 0) {
    const outlinesNum = builder.alloc();
    const itemNums = outline.map(() => builder.alloc());
    outline.forEach(([title, pageIndex], index) => {
      const parts = [
        `/Title (${escapeLiteral(title)})`,
        `/Parent ${outlinesNum} 0 R`,
        `/Dest [${pageNums[pageIndex]} 0 R /Fit]`,
      ];
      if (index > 0) {
        parts.push(`/Prev ${itemNums[index - 1]} 0 R`);
      }
      if (index + 1 < itemNums.length) {
        parts.push(`/Next ${itemNums[index + 1]} 0 R`);
      }
      builder.add(itemNums[index]!, latin1(`<< ${parts.join(" ")} >>`));
    });
    builder.add(
      outlinesNum,
      latin1(
        `<< /Type /Outlines /First ${itemNums[0]} 0 R /Last ${itemNums[itemNums.length - 1]} 0 R ` +
          `/Count ${itemNums.length} >>`,
      ),
    );
    outlinesRef = ` /Outlines ${outlinesNum} 0 R`;
  }
  builder.add(catalog, latin1(`<< /Type /Catalog /Pages ${pagesNum} 0 R${outlinesRef} >>`));

  return { builder, catalog, contents };
}

/**
 * A PDF whose page `i` extracts to exactly `pages[i]` through
 * `PdfConnector` (see the module docstring for how, given pdfjs-dist's own
 * text-extraction differences from pypdf). `outline` is a flat list of
 * `[title, pageIndex]` bookmarks, each pointing at `pages[pageIndex]` via
 * an explicit `/Dest`.
 */
export function textPdf(
  pages: readonly string[],
  options?: { outline?: ReadonlyArray<readonly [string, number]> },
): Uint8Array {
  const { builder, catalog, contents } = buildStructure(pages, options?.outline ?? []);
  for (const { number, plaintext } of contents) {
    builder.add(number, streamObject(plaintext));
  }
  return builder.render(catalog);
}

/**
 * A structurally valid PDF whose every page has an empty content stream —
 * the "photographed page, no text layer" shape `ocr_required` (ADR 0007
 * §10) exists to name.
 */
export function scannedPdf(pageCount: number): Uint8Array {
  return textPdf(new Array<string>(pageCount).fill(""));
}

// -- RC4 / standard security handler (PDF 32000-1:2008 §7.6.3, revision 2) --
//
// Implemented from scratch here (rather than via a PDF-writing library —
// none of this SDK's dependencies ship one) to produce a PDF pdfjs-dist's
// OWN standard-security-handler reader (pdf.worker.mjs's
// `CipherTransformFactory`) can open: Algorithm 2 (compute the file
// encryption key), Algorithm 3 (compute /O), and Algorithm 4 (compute /U,
// revision 2) — the same three algorithms, and the same 32-byte password
// padding string, `CipherTransformFactory` itself decodes against
// (verified empirically against pdfjs-dist 6.2.108 at port time). Revision
// 2 (40-bit RC4, 5-byte key) throughout: the simplest standard security
// handler variant, and sufficient to exercise `PdfConnector`'s two
// encrypted-PDF branches (a real user password, and an owner-password-only
// document that still opens with an empty one).

const PASSWORD_PADDING = new Uint8Array([
  0x28, 0xbf, 0x4e, 0x5e, 0x4e, 0x75, 0x8a, 0x41, 0x64, 0x00, 0x4e, 0x56, 0xff, 0xfa, 0x01, 0x08,
  0x2e, 0x2e, 0x00, 0xb6, 0xd0, 0x68, 0x3e, 0x80, 0x2f, 0x0c, 0xa9, 0xfe, 0x64, 0x53, 0x69, 0x7a,
]);

function md5(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(createHash("md5").update(Buffer.from(bytes)).digest());
}

function rc4(key: Uint8Array, data: Uint8Array): Uint8Array {
  const s = new Uint8Array(256);
  for (let index = 0; index < 256; index += 1) {
    s[index] = index;
  }
  let j = 0;
  for (let index = 0; index < 256; index += 1) {
    j = (j + s[index]! + key[index % key.length]!) % 256;
    const tmp = s[index]!;
    s[index] = s[j]!;
    s[j] = tmp;
  }
  const out = new Uint8Array(data.length);
  let i = 0;
  j = 0;
  for (let k = 0; k < data.length; k += 1) {
    i = (i + 1) % 256;
    j = (j + s[i]!) % 256;
    const tmp = s[i]!;
    s[i] = s[j]!;
    s[j] = tmp;
    out[k] = data[k]! ^ s[(s[i]! + s[j]!) % 256]!;
  }
  return out;
}

function padPassword(password: string): Uint8Array {
  const bytes = latin1(password);
  const out = new Uint8Array(32);
  const n = Math.min(32, bytes.length);
  out.set(bytes.subarray(0, n), 0);
  out.set(PASSWORD_PADDING.subarray(0, 32 - n), n);
  return out;
}

function permissionBytesLE(flags: number): Uint8Array {
  return new Uint8Array([
    flags & 0xff,
    (flags >> 8) & 0xff,
    (flags >> 16) & 0xff,
    (flags >>> 24) & 0xff,
  ]);
}

function computeOwnerValue(
  ownerPasswordPadded: Uint8Array,
  userPasswordPadded: Uint8Array,
  keyLengthBytes: number,
): Uint8Array {
  const ownerKey = md5(ownerPasswordPadded).subarray(0, keyLengthBytes);
  return rc4(ownerKey, userPasswordPadded);
}

function computeEncryptionKey(
  userPasswordPadded: Uint8Array,
  ownerValue: Uint8Array,
  permissions: number,
  fileId: Uint8Array,
  keyLengthBytes: number,
): Uint8Array {
  const input = concatBytes(userPasswordPadded, ownerValue, permissionBytesLE(permissions), fileId);
  return md5(input).subarray(0, keyLengthBytes);
}

function computeUserValue(fileEncryptionKey: Uint8Array): Uint8Array {
  return rc4(fileEncryptionKey, PASSWORD_PADDING);
}

function buildObjectKey(
  fileEncryptionKey: Uint8Array,
  objectNumber: number,
  generation: number,
): Uint8Array {
  const n = fileEncryptionKey.length;
  const input = concatBytes(
    fileEncryptionKey,
    new Uint8Array([
      objectNumber & 0xff,
      (objectNumber >> 8) & 0xff,
      (objectNumber >> 16) & 0xff,
      generation & 0xff,
      (generation >> 8) & 0xff,
    ]),
  );
  return md5(input).subarray(0, Math.min(n + 5, 16));
}

function toHex(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("hex");
}

/**
 * `textPdf(pages)`, re-encrypted with the from-scratch RC4 revision-2
 * standard security handler above — pdfjs-dist has no writer surface (it's
 * a reader only), so unlike the Python twin (which delegates to
 * `pypdf.PdfWriter.encrypt`), this implements the handful of PDF-spec
 * algorithms directly. A blank `userPassword` (the default an "owner
 * restrictions only" PDF uses) still opens without a password —
 * `ownerPassword` defaults to `userPassword` when not given, mirroring
 * `pypdf.PdfWriter.encrypt`'s own default.
 */
export function encryptedPdf(
  pages: readonly string[] = ["Secret text."],
  options?: { userPassword?: string; ownerPassword?: string | null },
): Uint8Array {
  const userPassword = options?.userPassword ?? "secret";
  const ownerPassword = options?.ownerPassword ?? userPassword;
  const keyLengthBytes = 5;
  const fileId = new Uint8Array(randomBytes(16));

  const userPasswordPadded = padPassword(userPassword);
  const ownerPasswordPadded = padPassword(ownerPassword);
  const ownerValue = computeOwnerValue(ownerPasswordPadded, userPasswordPadded, keyLengthBytes);
  const permissions = -1;
  const fileEncryptionKey = computeEncryptionKey(
    userPasswordPadded,
    ownerValue,
    permissions,
    fileId,
    keyLengthBytes,
  );
  const userValue = computeUserValue(fileEncryptionKey);

  const { builder, catalog, contents } = buildStructure(pages, []);
  for (const { number, plaintext } of contents) {
    const objectKey = buildObjectKey(fileEncryptionKey, number, 0);
    builder.add(number, streamObject(rc4(objectKey, plaintext)));
  }

  const idHex = toHex(fileId);
  const encryptDict =
    `<< /Filter /Standard /V 1 /R 2 /O <${toHex(ownerValue)}> /U <${toHex(userValue)}> ` +
    `/P ${permissions} >>`;
  const extraTrailer = ` /Encrypt ${encryptDict} /ID [<${idHex}> <${idHex}>]`;
  return builder.render(catalog, extraTrailer);
}

/**
 * A truncated PDF: a real xref/trailer starts to parse and then runs out
 * of bytes — pdfjs-dist rejects `getDocument(...).promise` with an
 * `InvalidPDFException` on this (structurally distinct from a
 * `PasswordException`), exercising `PdfConnector`'s generic `corrupt`
 * branch.
 */
export function corruptPdf(): Uint8Array {
  const data = textPdf(["This document will be cut off before its xref table."]);
  return data.subarray(0, Math.floor(data.length / 2));
}
