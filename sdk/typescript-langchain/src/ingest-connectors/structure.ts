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
