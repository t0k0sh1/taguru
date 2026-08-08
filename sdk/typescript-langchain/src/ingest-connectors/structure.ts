/**
 * Small text/heading helpers shared by more than one connector — first
 * extracted out of the HTML connector (issue #349) when the DOCX
 * connector (issue #350) needed the same paragraph-sanitization and
 * breadcrumb-fitting behavior for its own heading hierarchy. Not a
 * connector itself, and not part of the protocol (protocol.ts) — purely
 * internal plumbing. The mechanical mirror of the Python
 * `taguru_langchain.ingest_connectors._structure` module (issue #415).
 */

// Collapses an interior blank-line run to a single `\n` — mirrors exactly
// what a connector's paragraph splitting must never let slip through: a
// stray blank line INSIDE what the connector considers one paragraph (e.g.
// two consecutive `<br><br>`, or a `<w:br/>` pair) would otherwise
// silently register as an extra paragraph boundary once `splitParagraphs`
// (extract.ts) re-derives paragraphs from the final `"\n\n".join(...)`
// text, offsetting every locator/section paragraph index after it.
const BLANK_RUN_RE = /\n\s*\n+/g;

export function byteLen(text: string): number {
  return new TextEncoder().encode(text).length;
}

export function sanitizeParagraphText(text: string): string {
  return text.replace(BLANK_RUN_RE, "\n").trim();
}

/**
 * A zip package whose entries decompress past the caller's cap — the
 * decompression-bomb refusal `unzipWithinCap` throws, mapped by the
 * DOCX/PPTX connectors to a `content_too_large` diagnostic.
 */
export class DecompressedSizeExceededError extends Error {}

/** The fflate pieces `unzipWithinCap` needs, injected by the connectors'
 * own optional-dependency loading (`OoxmlDeps`). */
export interface BoundedUnzipDeps {
  Unzip: typeof import("fflate").Unzip;
  UnzipInflate: typeof import("fflate").UnzipInflate;
}

/**
 * `unzipSync`, but refusing any package whose entries decompress to more
 * than `cap` bytes in total — measured by ACTUALLY inflating every entry
 * through fflate's streaming `Unzip` rather than trusting the central
 * directory's declared `originalSize`, which an attacker can forge to a
 * small value while the deflate stream still expands to gigabytes. The
 * stream aborts at the first byte past `cap`, so a bomb never
 * materializes more than `cap` bytes; a package within the cap is
 * assembled from this same single decompression pass. Mirrors the Python
 * twin's `_structure.decompressed_size_within`. Any other failure (a
 * malformed zip, an unsupported compression method) propagates for the
 * caller's own `corrupt` path, exactly as a plain `unzipSync` call would.
 */
export function unzipWithinCap(
  deps: BoundedUnzipDeps,
  raw: Uint8Array,
  cap: number,
): Record<string, Uint8Array> {
  // fflate's streaming Unzip silently yields zero entries for a non-zip
  // byte stream (it just never finds a local-header signature), where
  // `unzipSync` would throw — check the PK magic up front so a non-zip
  // still surfaces as an error for the caller's `corrupt` path.
  if (raw.length < 4 || raw[0] !== 0x50 || raw[1] !== 0x4b) {
    throw new Error("not a zip archive (no PK signature)");
  }
  let total = 0;
  const collected = new Map<string, Uint8Array[]>();
  const unzip = new deps.Unzip((file) => {
    if (file.name.endsWith("/")) {
      // A directory entry carries no bytes worth keeping (unzipSync
      // surfaces them as empty entries; nothing here reads them).
      return;
    }
    const parts: Uint8Array[] = [];
    collected.set(file.name, parts);
    file.ondata = (error, data, _final) => {
      if (error) {
        throw error;
      }
      total += data.length;
      if (total > cap) {
        throw new DecompressedSizeExceededError(
          `the package decompresses to more than ${cap} bytes`,
        );
      }
      parts.push(data);
    };
    file.start();
  });
  unzip.register(deps.UnzipInflate);
  unzip.push(raw, true);
  const entries: Record<string, Uint8Array> = {};
  for (const [name, parts] of collected) {
    const merged = new Uint8Array(parts.reduce((size, part) => size + part.length, 0));
    let offset = 0;
    for (const part of parts) {
      merged.set(part, offset);
      offset += part.length;
    }
    entries[name] = merged;
  }
  return entries;
}

/**
 * Joins `crumbs` (outermost first) with `separator`, dropping the
 * outermost (least specific) ancestor first until the result fits within
 * `maxBytes` — `null` if even the innermost crumb alone doesn't fit (or
 * `crumbs` is empty).
 */
export function fitBreadcrumb(
  crumbs: readonly string[],
  options: { separator: string; maxBytes: number },
): string | null {
  let remaining = [...crumbs];
  while (remaining.length > 0) {
    const candidate = remaining.join(options.separator);
    if (byteLen(candidate) <= options.maxBytes) {
      return candidate;
    }
    remaining = remaining.slice(1);
  }
  return null;
}
