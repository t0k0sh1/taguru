/**
 * The HTML connector (ADR 0007 §7/§8, issue #349): `.html`/`.htm`/`.xhtml`
 * files and `http(s)://` URLs into `ConnectorDocument` (document.ts). The
 * mechanical mirror of the Python `taguru_langchain.ingest_connectors.html`
 * module (issue #415).
 *
 * Boilerplate (script/style/nav/aside/hidden regions, and a page's own
 * header/footer when no `<main>`/`<article>` scopes the content) is
 * stripped before `text` is built; the heading hierarchy survives as a
 * breadcrumb `section` per paragraph (`"Guide > Installation"` — `sections`
 * is flat, so a multi-level heading path is flattened into one label
 * rather than dropped); and each heading's own `id` (or its nearest
 * `id`-bearing ancestor's) becomes a `{"kind": "fragment", "value": ...}`
 * locator on every paragraph from that heading up to the next one —
 * combined with `metadata.canonicalUrl`, a citation can point at a real
 * in-page deep link.
 *
 * Parsing has no Node built-in or third-party HTML parser to build on
 * (unlike Python's stdlib `html.parser.HTMLParser`), so this module
 * hand-rolls a small, tolerant HTML tokenizer/tree builder
 * (`tokenizeHtml`/`TreeBuilder` below) that reproduces the same "minimal
 * reference implementation, not a full parser" posture the Python module
 * takes: start/end/self-closing tags with attributes, `html.unescape`-style
 * named/numeric entity decoding, comments, and script/style raw-text
 * skipping, all case-insensitive on tag names, tolerant of malformed
 * markup. A page whose markup this parser cannot make sense of, or whose
 * body carries no extractable text once boilerplate is stripped (an
 * image-only page, an unrendered JS-shell SPA), degrades to a diagnostic
 * (`corrupt`/`ocr_required`) with empty `text` — never a thrown error,
 * never a silently thin passage.
 *
 * URL fetches use the global `fetch` (Node >= 20) with `redirect: "manual"`
 * and a hand-rolled redirect loop (`fetchFollowingRedirects` below) —
 * chosen over `redirect: "follow"` specifically so each hop's destination
 * can be validated BEFORE that hop's request is made, mirroring the
 * Python module's `httpx` `request` event hook, which fires once per
 * redirect hop too. The source id is the *final*, fragment-stripped,
 * canonicalized URL (ADR 0007 §6.1) — a page's own
 * `<link rel="canonical">`, when present, is recorded in
 * `metadata.canonicalUrl` but never substituted for the fetch URL as the
 * source id, since a page can claim any canonical and two distinct pages
 * claiming the same one would otherwise collide.
 *
 * Fetching an arbitrary `http(s)://` URL is only ever safe when the
 * caller controls (or trusts) the URL — `HtmlConnector` is not a defense
 * against a malicious *caller* feeding it a URL chosen by an untrusted
 * third party (user-submitted, scraped from an external feed, produced by
 * an LLM). By default, though, it still refuses a destination that
 * resolves to a private, loopback, link-local, multicast, or otherwise
 * non-global IP address — including one reached only after following a
 * redirect — so an *otherwise-trusted* URL cannot be turned into a probe
 * of `localhost`, the caller's own internal network, or a cloud metadata
 * endpoint (`169.254.169.254`) by a redirect the origin server controls.
 * This check resolves DNS itself (`node:dns/promises`) and cannot fully
 * close a same-instant DNS-rebinding race against a hostile resolver; a
 * deployment that fetches URLs supplied by a genuinely untrusted party
 * should not rely on it alone. Pass `allowPrivateNetworks: true` to
 * disable the check entirely — this connector's own tests need it to
 * reach their loopback test server. Unlike Python's `ipaddress` module,
 * Node has no stdlib IP-range classifier, so `isBlockedIp` below
 * hand-implements a deliberately non-exhaustive (but test-covering) set
 * of private/loopback/link-local/multicast/reserved ranges for IPv4 and
 * IPv6 — a documented, scoped deviation, not a claim of parity with
 * Python's `ipaddress.ip_address(...).is_global`.
 *
 * Out of scope, deliberately: ADR 0007 §7.3's `{"kind": "table", ...}`
 * locator. At most one locator is stored per paragraph (§7.2), and this
 * connector already spends that one slot on `fragment`; a `<table>` still
 * becomes its own paragraph (never folded into a neighbour, per §7.3) and
 * inherits whichever `fragment` locator its surrounding heading set, but
 * gets no locator of its own kind.
 */

import type { Connector } from "./protocol.js";
import {
  ConnectorDocument,
  ConnectorMetadata,
  Diagnostic,
  type DiagnosticCode,
  FingerprintInputs,
  LocatorEntry,
  MAX_SECTION_BYTES,
  SectionEntry,
  optionsDigest,
} from "./document.js";
import { canonicalizeUrl, checkSourceId, fileSourceId } from "./sources.js";
import { byteLen, fitBreadcrumb, sanitizeParagraphText } from "./structure.js";
import { MAX_PASSAGE_BYTES } from "../extract.js";

export const PARSER_NAME = "taguru-html-connector";
export const PARSER_VERSION = "1.0.0";

const SUPPORTED_SUFFIXES: ReadonlySet<string> = new Set([".html", ".htm", ".xhtml"]);
const SUPPORTED_URL_SCHEMES: ReadonlySet<string> = new Set(["http", "https"]);
const ALLOWED_CONTENT_TYPES: ReadonlySet<string> = new Set(["text/html", "application/xhtml+xml"]);

// A Windows drive-letter path (`C:\docs\a.html`, `C:/docs/a.html`) parses
// under a bare scheme-prefix regex as scheme `"c"` — indistinguishable,
// by scheme alone, from a genuine (if unusual) single-letter URL scheme.
// Checked before any scheme dispatch so a Windows absolute path is never
// misrouted into the URL-fetch path or refused as `unsupported_format`.
const WINDOWS_DRIVE_RE = /^[A-Za-z]:[\\/]/;

// The raw-file/fetch cap, checked before (local) or during (URL,
// streamed) reading — independent of MAX_PASSAGE_BYTES (checked again
// below, against the EXTRACTED text): markup overhead means a
// large-but-sparse page can stay well under the text cap, and a
// heavily-templated small page's raw bytes can still exceed a naive
// per-file cap.
const DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024;
// Each fetch (initial or a followed redirect hop) gets its own
// `AbortSignal.timeout` bound to this many seconds — an approximation of
// httpx's per-phase (connect/write/read) timeout, since the Fetch API has
// no equivalent per-phase primitive; `maxTotalSeconds` (below) is the
// separate ceiling on the fetch's total wall-clock time, enforced by hand
// in `readUrl` via `Date.now()`, catching a server that trickles data
// slower than `timeout` but without ever going fully idle.
const DEFAULT_TIMEOUT = 30.0;
const DEFAULT_MAX_TOTAL_SECONDS = 60.0;
const MAX_REDIRECTS = 20;
const REDIRECT_STATUSES: ReadonlySet<number> = new Set([301, 302, 303, 307, 308]);

// Dropped entirely, with every descendant: never rendered content, never
// navigation chrome. `header`/`footer` are handled separately (below)
// since whether they count as boilerplate depends on whether a
// `<main>`/`<article>` already scopes the content.
const DROP_TAGS: ReadonlySet<string> = new Set([
  "script",
  "style",
  "noscript",
  "template",
  "svg",
  "form",
  "button",
  "select",
  "iframe",
  "frame",
  "object",
  "embed",
  "canvas",
  "nav",
  "aside",
]);
// Dropped only when no `<main>`/`<article>` was found to scope the
// content — inside one, a `<header>`/`<footer>` is the article's own
// (often holding its `<h1>` or a byline), not site chrome.
const CONDITIONAL_DROP_TAGS: ReadonlySet<string> = new Set(["header", "footer"]);
// ARIA landmark roles that are always chrome, regardless of tag name.
const ROLE_DENYLIST: ReadonlySet<string> = new Set([
  "navigation",
  "banner",
  "contentinfo",
  "complementary",
  "search",
]);
// Elements with no closing tag in ordinary (non-XHTML) HTML — never
// pushed onto the tree-builder's open-element stack.
const VOID_TAGS: ReadonlySet<string> = new Set([
  "area",
  "base",
  "br",
  "col",
  "embed",
  "hr",
  "img",
  "input",
  "link",
  "meta",
  "param",
  "source",
  "track",
  "wbr",
]);
// Raw-text elements: their content is never tokenized as markup (a stray
// `<`/`>` in inline JS/CSS must not be mistaken for a tag), and is never
// entity-decoded either — mirrors `html.parser`'s CDATA content-element
// handling for exactly these two tags.
const RAW_TEXT_TAGS: ReadonlySet<string> = new Set(["script", "style"]);
const HEADING_TAGS: ReadonlySet<string> = new Set(["h1", "h2", "h3", "h4", "h5", "h6"]);
// Block-level: each one flushes whatever inline text preceded it into its
// own paragraph, and is itself followed by a flush — so nesting
// (`<div><p>...`) never bleeds one block's text into a sibling's. `tr` is
// included as a fallback for a stray `<tr>` outside any `<table>`
// (malformed markup); ordinarily table rows are consumed whole by the
// table handler below and never reach this generic dispatch.
const BLOCK_TAGS: ReadonlySet<string> = new Set([
  "p",
  "div",
  "section",
  "article",
  "li",
  "blockquote",
  "figcaption",
  "dt",
  "dd",
  "tr",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
]);

const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

const WS_RE = /\s+/g;
const CHARSET_RE = /charset\s*=\s*["']?\s*([\w-]+)/i;
const HEADER_CHARSET_RE = /charset\s*=\s*["']?\s*([\w-]+)/i;

function stripFragment(url: string): string {
  const hashIndex = url.indexOf("#");
  return hashIndex === -1 ? url : url.slice(0, hashIndex);
}

function urlDisplayName(url: string): string {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  const segments = parsed.pathname.split("/").filter((segment) => segment !== "");
  if (segments.length > 0) {
    return segments[segments.length - 1]!;
  }
  return parsed.host || url;
}

function isWindowsDrivePath(reference: string): boolean {
  return WINDOWS_DRIVE_RE.test(reference);
}

function basenameOf(pathLike: string): string {
  const segments = pathLike.split(/[\\/]/);
  return segments[segments.length - 1] ?? pathLike;
}

function suffixOf(pathLike: string): string {
  const base = basenameOf(pathLike);
  const dotIndex = base.lastIndexOf(".");
  return dotIndex <= 0 ? "" : base.slice(dotIndex);
}

const SCHEME_RE = /^([A-Za-z][A-Za-z0-9+.-]*):/;

function urlScheme(reference: string): string {
  const match = SCHEME_RE.exec(reference);
  return match === null ? "" : match[1]!.toLowerCase();
}

function resolveUrl(base: string, href: string): string {
  try {
    return new URL(href, base).toString();
  } catch {
    return href;
  }
}

/**
 * The Fetch API (unlike Python's httpx) refuses to construct a `Request`
 * from a URL that embeds `userinfo` at all — `TypeError: Request cannot
 * be constructed from a URL that includes credentials` — rather than
 * silently dropping or forwarding it. `fetchFollowingRedirects` calls this
 * to strip `userinfo` from the URL actually handed to `fetch`, moving it
 * into a standard `Authorization: Basic` header instead (mirroring the
 * same auto-conversion httpx does), so a URL like
 * `http://user:pass@host/path` still reaches the server — the SOURCE ID
 * (derived from the ORIGINAL, credentialed reference via
 * `canonicalizeUrl`, ADR 0007 §6.1) always strips `userinfo` outright
 * regardless, independent of this.
 */
function prepareFetchUrl(url: string): { url: string; authorizationHeader: string | null } {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { url, authorizationHeader: null };
  }
  if (!parsed.username && !parsed.password) {
    return { url, authorizationHeader: null };
  }
  const credentials = `${decodeURIComponent(parsed.username)}:${decodeURIComponent(parsed.password)}`;
  const authorizationHeader = `Basic ${Buffer.from(credentials, "utf-8").toString("base64")}`;
  parsed.username = "";
  parsed.password = "";
  return { url: parsed.toString(), authorizationHeader };
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function sha256HexBytes(data: Uint8Array): Promise<string> {
  // `Buffer` (a `Uint8Array<ArrayBufferLike>`, which also admits a
  // `SharedArrayBuffer`-backed view) is not directly assignable to
  // `BufferSource` under this package's `lib.dom.d.ts` — `.slice()`
  // copies into a fresh, definitely-`ArrayBuffer`-backed `Uint8Array`.
  const digest = await crypto.subtle.digest("SHA-256", data.slice());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

// -- SSRF mitigation: resolve-then-check before every fetch/redirect hop -----

/** Raised from `assertPublicDestination` to abort a fetch whose
 * destination resolved to a non-global IP address — caught alongside any
 * other fetch error in `readUrl` and mapped to the same `unreadable`
 * diagnostic every other fetch failure gets. */
class BlockedDestinationError extends Error {}

function ipv4ToInt(ip: string): number | null {
  const parts = ip.split(".");
  if (parts.length !== 4) {
    return null;
  }
  let value = 0;
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) {
      return null;
    }
    const n = Number(part);
    if (n > 255) {
      return null;
    }
    value = value * 256 + n;
  }
  return value;
}

function inIpv4Range(value: number, base: string, prefixLength: number): boolean {
  const baseValue = ipv4ToInt(base)!;
  const shift = 32 - prefixLength;
  const mask = shift === 32 ? 0 : (0xffffffff << shift) >>> 0;
  return ((value & mask) >>> 0) === ((baseValue & mask) >>> 0);
}

// A deliberately non-exhaustive set of IPv4 ranges that are never a sane
// public HTTP destination — this-host, private (RFC 1918), carrier-grade
// NAT, loopback, link-local (including the cloud metadata endpoint
// 169.254.169.254), the IETF protocol assignments/NAT64/documentation/
// benchmarking ranges, multicast, and reserved/broadcast.
const IPV4_BLOCKED_RANGES: ReadonlyArray<readonly [string, number]> = [
  ["0.0.0.0", 8],
  ["10.0.0.0", 8],
  ["100.64.0.0", 10],
  ["127.0.0.0", 8],
  ["169.254.0.0", 16],
  ["172.16.0.0", 12],
  ["192.0.0.0", 24],
  ["192.0.2.0", 24],
  ["192.168.0.0", 16],
  ["198.18.0.0", 15],
  ["198.51.100.0", 24],
  ["203.0.113.0", 24],
  ["224.0.0.0", 4], // multicast
  ["240.0.0.0", 4], // reserved
  ["255.255.255.255", 32],
];

function isBlockedIpv4(ip: string): boolean {
  const value = ipv4ToInt(ip);
  if (value === null) {
    return false;
  }
  return IPV4_BLOCKED_RANGES.some(([base, prefix]) => inIpv4Range(value, base, prefix));
}

function isBlockedIpv4Int(value: number): boolean {
  return IPV4_BLOCKED_RANGES.some(([base, prefix]) => inIpv4Range(value, base, prefix));
}

/** Expands a textual IPv6 address (with optional `::` compression and an
 * optional embedded-IPv4 tail, e.g. `::ffff:127.0.0.1`) into 8 16-bit
 * groups, or `null` if it cannot be parsed. */
function expandIpv6(address: string): number[] | null {
  let addr = address;
  const lastColon = addr.lastIndexOf(":");
  const tail = addr.slice(lastColon + 1);
  if (tail.includes(".")) {
    const v4 = ipv4ToInt(tail);
    if (v4 === null) {
      return null;
    }
    const hi = (v4 >>> 16) & 0xffff;
    const lo = v4 & 0xffff;
    addr = `${addr.slice(0, lastColon + 1)}${hi.toString(16)}:${lo.toString(16)}`;
  }
  const parts = addr.split("::");
  if (parts.length > 2) {
    return null;
  }
  const head = parts[0] ? parts[0]!.split(":").filter((s) => s !== "") : [];
  const tail2 = parts.length === 2 && parts[1] ? parts[1]!.split(":").filter((s) => s !== "") : [];
  const headNums = head.map((h) => Number.parseInt(h, 16));
  const tailNums = tail2.map((h) => Number.parseInt(h, 16));
  if (headNums.some(Number.isNaN) || tailNums.some(Number.isNaN)) {
    return null;
  }
  let full: number[];
  if (parts.length === 2) {
    const missing = 8 - headNums.length - tailNums.length;
    if (missing < 0) {
      return null;
    }
    full = [...headNums, ...new Array<number>(missing).fill(0), ...tailNums];
  } else {
    full = headNums;
  }
  return full.length === 8 ? full : null;
}

function isBlockedIpv6(address: string): boolean {
  const hextets = expandIpv6(address);
  if (hextets === null) {
    return false;
  }
  if (hextets.every((h) => h === 0)) {
    return true; // :: (unspecified)
  }
  if (hextets.slice(0, 7).every((h) => h === 0) && hextets[7] === 1) {
    return true; // ::1 (loopback)
  }
  const first = hextets[0]!;
  if ((first & 0xffc0) === 0xfe80) {
    return true; // fe80::/10 (link-local)
  }
  if ((first & 0xfe00) === 0xfc00) {
    return true; // fc00::/7 (unique local)
  }
  if ((first & 0xff00) === 0xff00) {
    return true; // ff00::/8 (multicast)
  }
  if (
    hextets[0] === 0 &&
    hextets[1] === 0 &&
    hextets[2] === 0 &&
    hextets[3] === 0 &&
    hextets[4] === 0 &&
    hextets[5] === 0xffff
  ) {
    // ::ffff:0:0/96 (IPv4-mapped) — judge the embedded IPv4 address.
    const v4 = ((hextets[6]! << 16) | hextets[7]!) >>> 0;
    return isBlockedIpv4Int(v4);
  }
  return false;
}

function isBlockedIp(rawAddress: string): boolean {
  const address = rawAddress.split("%")[0]!; // drop an IPv6 zone id
  return address.includes(":") ? isBlockedIpv6(address) : isBlockedIpv4(address);
}

/**
 * Fires before the initial request AND before every followed redirect hop
 * (`fetchFollowingRedirects` calls this once per hop) — mirrors the
 * Python module's httpx `request` event hook, which fires per hop too. A
 * hostname that fails to resolve is left alone here — the connection
 * attempt that follows will fail on its own, through the ordinary fetch
 * error path, so this only has an opinion once resolution actually names
 * a destination.
 */
async function assertPublicDestination(urlString: string): Promise<void> {
  const hostname = new URL(urlString).hostname;
  const { lookup } = await import("node:dns/promises");
  let addresses: Array<{ address: string; family: number }>;
  try {
    addresses = await lookup(hostname, { all: true, verbatim: true });
  } catch {
    return;
  }
  for (const { address } of addresses) {
    if (isBlockedIp(address)) {
      throw new BlockedDestinationError(
        `refusing to fetch '${hostname}': resolves to a private/internal address`,
      );
    }
  }
}

// -- charset resolution --------------------------------------------------------

/**
 * A byte-order mark first (it is definitive, and the UTF-16 ones make
 * every later text-level scan meaningless), then `Content-Type` header
 * `charset=`, else an ASCII-safe scan of a `<meta charset=...>`/`<meta
 * http-equiv="Content-Type" content="... charset=...">` tag in the first
 * 2 KiB (where HTML5 requires one to live), else UTF-8 — after one last
 * look for the interleaved-NUL shape of BOM-less UTF-16. That last case
 * matters because it decodes as UTF-8 WITHOUT error: every ASCII-range
 * character in UTF-16 pairs its code with a NUL byte, so the "text"
 * comes out with NULs between the markup's characters, the parser sees
 * no tags at all, and `<script>` bodies (dropped from text on every
 * healthy path) would ride into the body text undiagnosed. A UTF-8 BOM
 * still surfaces as U+FEFF after decoding and is stripped in
 * `parseDocument`, the same convention text.ts uses.
 */
function resolveCharset(raw: Buffer, contentTypeHeader: string | null): string {
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    // TextDecoder consumes a matching BOM itself for these labels.
    return "utf-16le";
  }
  if (raw.length >= 2 && raw[0] === 0xfe && raw[1] === 0xff) {
    return "utf-16be";
  }
  if (contentTypeHeader) {
    const headerMatch = HEADER_CHARSET_RE.exec(contentTypeHeader);
    if (headerMatch) {
      return headerMatch[1]!;
    }
  }
  const sample = raw.subarray(0, 2048);
  const prefix = sample.toString("latin1");
  const metaMatch = CHARSET_RE.exec(prefix);
  if (metaMatch) {
    return metaMatch[1]!;
  }
  if (sample.length > 0) {
    // BOM-less UTF-16 heuristic: markup is ASCII, so one byte of every
    // pair is NUL — concentrated on odd offsets for LE, even for BE.
    let oddNuls = 0;
    let evenNuls = 0;
    for (let i = 0; i < sample.length; i += 1) {
      if (sample[i] === 0) {
        if (i % 2 === 1) {
          oddNuls += 1;
        } else {
          evenNuls += 1;
        }
      }
    }
    const quarter = Math.floor(sample.length / 4);
    if (oddNuls > quarter && oddNuls > 2 * evenNuls) {
      return "utf-16le";
    }
    if (evenNuls > quarter && evenNuls > 2 * oddNuls) {
      return "utf-16be";
    }
  }
  return "utf-8";
}

function decodeBuffer(raw: Buffer, charset: string): string {
  const decoder = new TextDecoder(charset, { fatal: true });
  return decoder.decode(raw);
}

// -- entity decoding (mirrors Python's `html.unescape` for the named/ ----------
// -- numeric entities this connector's own tests exercise) ---------------------

// A deliberately bounded named-entity table — the XML built-ins plus a
// common HTML4/5 subset — not the full ~2000-entry HTML5 named character
// reference table Python's `html.unescape` draws on. No test in this
// connector's suite exercises an entity reference at all; this table
// exists for the connector's general robustness against real-world markup
// rather than to satisfy a specific assertion.
const NAMED_ENTITIES: Readonly<Record<string, string>> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: "\u00a0",
  copy: "\u00a9",
  reg: "\u00ae",
  trade: "\u2122",
  mdash: "\u2014",
  ndash: "\u2013",
  hellip: "\u2026",
  lsquo: "\u2018",
  rsquo: "\u2019",
  ldquo: "\u201c",
  rdquo: "\u201d",
  laquo: "\u00ab",
  raquo: "\u00bb",
  eacute: "\u00e9",
  egrave: "\u00e8",
  agrave: "\u00e0",
  ccedil: "\u00e7",
  uuml: "\u00fc",
  ouml: "\u00f6",
  auml: "\u00e4",
  szlig: "\u00df",
  euro: "\u20ac",
  cent: "\u00a2",
  pound: "\u00a3",
  yen: "\u00a5",
  sect: "\u00a7",
  para: "\u00b6",
  middot: "\u00b7",
  deg: "\u00b0",
  plusmn: "\u00b1",
  times: "\u00d7",
  divide: "\u00f7",
  frac12: "\u00bd",
  frac14: "\u00bc",
  frac34: "\u00be",
  bull: "\u2022",
  dagger: "\u2020",
  Dagger: "\u2021",
};

const ENTITY_RE = /&(#x[0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]*);/g;

function decodeEntities(text: string): string {
  if (!text.includes("&")) {
    return text;
  }
  return text.replace(ENTITY_RE, (match, ref: string) => {
    if (ref.startsWith("#")) {
      const isHex = ref[1] === "x" || ref[1] === "X";
      const code = Number.parseInt(isHex ? ref.slice(2) : ref.slice(1), isHex ? 16 : 10);
      if (Number.isNaN(code)) {
        return match;
      }
      try {
        return String.fromCodePoint(code);
      } catch {
        return match;
      }
    }
    return NAMED_ENTITIES[ref] ?? match;
  });
}

// -- hand-rolled HTML tokenizer / tree builder ----------------------------------

/** One element in the tree `TreeBuilder` assembles — named `HtmlNode`
 * rather than `Node` to avoid shadowing the DOM lib's global `Node`
 * interface (this package's tsconfig includes the `DOM` lib for other
 * modules' sake). Attribute names are already lowercased by the
 * tokenizer; values are compared case-insensitively where HTML itself is
 * case-insensitive (`rel`, `role`, `aria-hidden`). */
class HtmlNode {
  tag: string;
  attrs: Record<string, string>;
  children: Array<HtmlNode | string> = [];

  constructor(tag: string, attrs: Record<string, string>) {
    this.tag = tag;
    this.attrs = attrs;
  }
}

interface TokenizerCallbacks {
  onStartTag(tag: string, attrs: Record<string, string>, selfClosing: boolean): void;
  onEndTag(tag: string): void;
  onData(data: string): void;
}

const TAG_NAME_RE = /^[A-Za-z][A-Za-z0-9:_-]*/;
const END_TAG_NAME_RE = /^\/\s*([A-Za-z][A-Za-z0-9:_-]*)/;

/**
 * Parses attributes starting right after a tag's name, up to (and past)
 * its closing `>`. Returns `end: -1` when the tag was never closed (ran
 * off the end of the document) — the caller then treats the rest of the
 * document as consumed, the same "tolerant, never throws" posture as
 * every other malformed-markup case this tokenizer handles.
 */
function parseAttributes(
  html: string,
  start: number,
): { attrs: Record<string, string>; selfClosing: boolean; end: number } {
  const attrs: Record<string, string> = {};
  const length = html.length;
  let i = start;
  const selfClosing = false;
  while (i < length) {
    while (i < length && /\s/.test(html[i]!)) {
      i += 1;
    }
    if (i >= length) {
      return { attrs, selfClosing, end: -1 };
    }
    if (html[i] === ">") {
      return { attrs, selfClosing, end: i + 1 };
    }
    if (html[i] === "/") {
      if (html[i + 1] === ">") {
        return { attrs, selfClosing: true, end: i + 2 };
      }
      i += 1;
      continue;
    }
    const nameStart = i;
    while (i < length && !/[\s=/>]/.test(html[i]!)) {
      i += 1;
    }
    const name = html.slice(nameStart, i).toLowerCase();
    if (!name) {
      i += 1;
      continue;
    }
    while (i < length && /\s/.test(html[i]!)) {
      i += 1;
    }
    let value = "";
    if (html[i] === "=") {
      i += 1;
      while (i < length && /\s/.test(html[i]!)) {
        i += 1;
      }
      if (html[i] === '"' || html[i] === "'") {
        const quote = html[i]!;
        i += 1;
        const valueStart = i;
        const closeIndex = html.indexOf(quote, i);
        const valueEnd = closeIndex === -1 ? length : closeIndex;
        value = decodeEntities(html.slice(valueStart, valueEnd));
        i = closeIndex === -1 ? length : closeIndex + 1;
      } else {
        const valueStart = i;
        while (i < length && !/[\s>]/.test(html[i]!)) {
          i += 1;
        }
        value = decodeEntities(html.slice(valueStart, i));
      }
    }
    attrs[name] = value;
  }
  return { attrs, selfClosing, end: -1 };
}

/**
 * A small, tolerant HTML tokenizer over the whole document string —
 * everything `TreeBuilder` needs from a real parser: start/end/
 * self-closing tags with attributes, comments/doctypes/processing
 * instructions/CDATA sections skipped without emitting anything, raw-text
 * handling for `<script>`/`<style>`, and entity decoding in text/attribute
 * content (never inside raw text, mirroring `html.parser`'s CDATA mode).
 * An unrecognized `<` (not the start of a tag, comment, or declaration) is
 * emitted as a literal `<` character — malformed markup degrades quietly
 * rather than raising, the same posture the Python module's own
 * `HTMLParser` subclass documents for itself.
 */
function tokenizeHtml(html: string, cb: TokenizerCallbacks): void {
  const length = html.length;
  let pos = 0;
  while (pos < length) {
    const lt = html.indexOf("<", pos);
    if (lt === -1) {
      const text = html.slice(pos);
      if (text) {
        cb.onData(decodeEntities(text));
      }
      break;
    }
    if (lt > pos) {
      cb.onData(decodeEntities(html.slice(pos, lt)));
    }
    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt + 4);
      pos = end === -1 ? length : end + 3;
      continue;
    }
    if (html.startsWith("<![CDATA[", lt)) {
      const end = html.indexOf("]]>", lt + 9);
      pos = end === -1 ? length : end + 3;
      continue;
    }
    if (html.startsWith("<!", lt) || html.startsWith("<?", lt)) {
      const end = html.indexOf(">", lt + 2);
      pos = end === -1 ? length : end + 1;
      continue;
    }
    if (html.startsWith("</", lt)) {
      const nameMatch = END_TAG_NAME_RE.exec(html.slice(lt + 1));
      const end = html.indexOf(">", lt + 2);
      pos = end === -1 ? length : end + 1;
      if (nameMatch !== null) {
        cb.onEndTag(nameMatch[1]!.toLowerCase());
      }
      // No matching tag-name shape (e.g. "</>", "</1foo>"): a bogus
      // construct, skipped like a comment — nothing emitted.
      continue;
    }
    const nameMatch = TAG_NAME_RE.exec(html.slice(lt + 1));
    if (nameMatch === null) {
      // '<' is not the start of any recognized construct: literal text.
      cb.onData("<");
      pos = lt + 1;
      continue;
    }
    const tag = nameMatch[0].toLowerCase();
    const { attrs, selfClosing, end } = parseAttributes(html, lt + 1 + nameMatch[0].length);
    pos = end === -1 ? length : end;
    cb.onStartTag(tag, attrs, selfClosing);
    if (RAW_TEXT_TAGS.has(tag) && !selfClosing) {
      const closeRe = new RegExp(`</${tag}\\s*>`, "i");
      const rest = html.slice(pos);
      const match = closeRe.exec(rest);
      if (match !== null) {
        if (match.index > 0) {
          cb.onData(rest.slice(0, match.index));
        }
        pos += match.index + match[0].length;
        cb.onEndTag(tag);
      } else {
        if (rest) {
          cb.onData(rest);
        }
        pos = length;
      }
    }
  }
}

/**
 * Builds a minimal DOM from HTML text via `tokenizeHtml`, tolerant of
 * malformed markup: an unmatched close tag is ignored rather than raised
 * — real, hand-authored HTML on the web is routinely not well-formed, and
 * a connector that raised on every such page would defeat its own
 * purpose.
 */
class TreeBuilder {
  readonly root: HtmlNode;
  private readonly stack: HtmlNode[];

  constructor() {
    this.root = new HtmlNode("#document", {});
    this.stack = [this.root];
  }

  feed(text: string): void {
    tokenizeHtml(text, {
      onStartTag: (tag, attrs, selfClosing) => this.handleStartTag(tag, attrs, selfClosing),
      onEndTag: (tag) => this.handleEndTag(tag),
      onData: (data) => this.handleData(data),
    });
  }

  private handleStartTag(tag: string, attrs: Record<string, string>, selfClosing: boolean): void {
    const node = new HtmlNode(tag, attrs);
    this.stack[this.stack.length - 1]!.children.push(node);
    if (!VOID_TAGS.has(tag) && !selfClosing) {
      this.stack.push(node);
    }
  }

  private handleEndTag(tag: string): void {
    for (let index = this.stack.length - 1; index > 0; index -= 1) {
      if (this.stack[index]!.tag === tag) {
        this.stack.length = index;
        return;
      }
    }
    // No matching open tag on the stack: malformed markup, ignored.
  }

  private handleData(data: string): void {
    if (data) {
      this.stack[this.stack.length - 1]!.children.push(data);
    }
  }
}

function findFirst(node: HtmlNode, tag: string): HtmlNode | null {
  for (const child of node.children) {
    if (typeof child === "string") {
      continue;
    }
    if (child.tag === tag) {
      return child;
    }
    const found = findFirst(child, tag);
    if (found !== null) {
      return found;
    }
  }
  return null;
}

function findAll(
  node: HtmlNode,
  tag: string,
  stopTags: ReadonlySet<string> = new Set(),
): HtmlNode[] {
  const results: HtmlNode[] = [];
  for (const child of node.children) {
    if (typeof child === "string" || stopTags.has(child.tag)) {
      continue;
    }
    if (child.tag === tag) {
      results.push(child);
    }
    results.push(...findAll(child, tag, stopTags));
  }
  return results;
}

/**
 * Whitespace-collapsed concatenation of every descendant text node,
 * skipping `DROP_TAGS` (so e.g. a `<title>` never absorbs stray
 * `<script>` content some templating leaves inside it) and, when passed,
 * never descending into `stopTags` — used for a table cell's own text
 * (`stopTags: {"table"}`) so a nested table's cells are never glued into
 * the outer cell's text.
 */
function elementText(node: HtmlNode, stopTags: ReadonlySet<string> = new Set()): string {
  const parts: string[] = [];
  const visit = (current: HtmlNode): void => {
    for (const child of current.children) {
      if (typeof child === "string") {
        parts.push(child);
      } else if (!DROP_TAGS.has(child.tag) && !stopTags.has(child.tag)) {
        visit(child);
      }
    }
  };
  visit(node);
  return parts.join("").replace(WS_RE, " ").trim();
}

function extractTitle(tree: HtmlNode): string | null {
  const titleNode = findFirst(tree, "title");
  if (titleNode !== null) {
    const text = elementText(titleNode);
    if (text) {
      return text;
    }
  }
  const h1 = findFirst(tree, "h1");
  if (h1 !== null) {
    const text = elementText(h1);
    if (text) {
      return text;
    }
  }
  return null;
}

function extractCanonicalHref(tree: HtmlNode): string | null {
  for (const link of findAll(tree, "link")) {
    const relValues = new Set(
      (link.attrs["rel"] ?? "")
        .split(/\s+/)
        .filter((token) => token !== "")
        .map((token) => token.toLowerCase()),
    );
    if (relValues.has("canonical")) {
      const href = link.attrs["href"];
      if (href) {
        return href;
      }
    }
  }
  return null;
}

// -- one pass over the content root: builds paragraph-joined `text` plus -------
// -- paragraph-indexed `locators`/`sections` ------------------------------------

/**
 * One pass over the content root: builds paragraph-joined `text` plus
 * paragraph-indexed `locators`/`sections`, mirroring exactly the shape
 * `splitParagraphs` (extract.ts, mirroring src/paragraph.rs) will
 * re-derive from the final `"\n\n".join(paragraphs)` text — never the
 * reverse (this connector builds paragraphs FROM the DOM's own structure,
 * then never calls the splitter itself; `html-connector.test.ts`'s
 * resplit invariant is what keeps the two in lockstep).
 */
class Extractor {
  paragraphs: string[] = [];
  locators: LocatorEntry[] = [];
  sections: SectionEntry[] = [];
  hadDroppedFrame = false;

  private readonly extractHeadings: boolean;
  private readonly headingSeparator: string;
  private readonly dropHeaderFooter: boolean;
  private readonly headingStack: Array<[number, string]> = [];
  private currentAnchor: string | null = null;
  private buffer: string[] = [];

  constructor(options: {
    extractHeadings: boolean;
    headingSeparator: string;
    dropHeaderFooter: boolean;
  }) {
    this.extractHeadings = options.extractHeadings;
    this.headingSeparator = options.headingSeparator;
    this.dropHeaderFooter = options.dropHeaderFooter;
  }

  run(root: HtmlNode): string {
    this.visit(root, root.attrs["id"] ?? null);
    this.flushBuffer();
    return this.paragraphs.join("\n\n");
  }

  private hiddenByAttrs(node: HtmlNode): boolean {
    if (Object.hasOwn(node.attrs, "hidden")) {
      return true;
    }
    if ((node.attrs["aria-hidden"] ?? "").trim().toLowerCase() === "true") {
      return true;
    }
    if (ROLE_DENYLIST.has((node.attrs["role"] ?? "").trim().toLowerCase())) {
      return true;
    }
    return false;
  }

  private shouldDrop(node: HtmlNode): boolean {
    if (DROP_TAGS.has(node.tag)) {
      return true;
    }
    if (CONDITIONAL_DROP_TAGS.has(node.tag) && this.dropHeaderFooter) {
      return true;
    }
    return this.hiddenByAttrs(node);
  }

  /** Appends `text` as a new paragraph (attaching the current section's
   * `fragment` locator, if any) and returns its index — or `null` if
   * `text` sanitized to nothing, in which case no paragraph was
   * appended. */
  private commitParagraph(text: string): number | null {
    const sanitized = sanitizeParagraphText(text);
    if (!sanitized) {
      return null;
    }
    const index = this.paragraphs.length;
    this.paragraphs.push(sanitized);
    if (this.currentAnchor !== null) {
      this.locators.push(
        new LocatorEntry({ paragraph: index, locator: { kind: "fragment", value: this.currentAnchor } }),
      );
    }
    return index;
  }

  private flushBuffer(): void {
    if (this.buffer.length === 0) {
      return;
    }
    const text = this.buffer.join("");
    this.buffer = [];
    this.commitParagraph(text);
  }

  private visit(node: HtmlNode, ambientId: string | null): void {
    if (this.shouldDrop(node)) {
      // Flag only a frame a reader would have seen. One hidden by its
      // own attributes contributes no visible content, and one inside an
      // already-dropped subtree (nav/aside/…) never reaches this visit
      // at all — the diagnostic means "visible embedded content was not
      // fetched", not "a frame tag existed somewhere".
      if ((node.tag === "iframe" || node.tag === "frame") && !this.hiddenByAttrs(node)) {
        this.hadDroppedFrame = true;
      }
      return;
    }
    const ownId = node.attrs["id"];
    const effectiveId = ownId ? ownId : ambientId;
    const tag = node.tag;
    if (tag === "br") {
      this.buffer.push("\n");
      return;
    }
    if (HEADING_TAGS.has(tag)) {
      this.flushBuffer();
      this.handleHeading(node, Number(tag[1]), effectiveId);
      return;
    }
    if (tag === "table") {
      this.flushBuffer();
      this.handleTable(node);
      return;
    }
    if (tag === "pre") {
      this.flushBuffer();
      this.commitParagraph(this.preText(node));
      return;
    }
    if (BLOCK_TAGS.has(tag)) {
      this.flushBuffer();
      for (const child of node.children) {
        this.dispatchChild(child, effectiveId);
      }
      this.flushBuffer();
      return;
    }
    // Inline (or unrecognized) tag: no paragraph boundary of its own.
    for (const child of node.children) {
      this.dispatchChild(child, effectiveId);
    }
  }

  private dispatchChild(child: HtmlNode | string, ambientId: string | null): void {
    if (typeof child === "string") {
      this.buffer.push(child.replace(WS_RE, " "));
    } else {
      this.visit(child, ambientId);
    }
  }

  /** Raw text, whitespace UNCOLLAPSED (unlike every other paragraph kind)
   * — a `<pre>` block's own line breaks and indentation are content, not
   * formatting. Still passed through `sanitizeParagraphText` like every
   * paragraph (via `commitParagraph`), so an internal blank line
   * collapses to a single `\n` rather than covertly splitting into two
   * paragraphs on resplit. */
  private preText(node: HtmlNode): string {
    const parts: string[] = [];
    const visit = (current: HtmlNode): void => {
      for (const child of current.children) {
        if (typeof child === "string") {
          parts.push(child);
        } else if (!this.shouldDrop(child)) {
          if (child.tag === "br") {
            parts.push("\n");
          } else {
            visit(child);
          }
        }
      }
    };
    visit(node);
    return parts.join("");
  }

  private handleHeading(node: HtmlNode, level: number, anchor: string | null): void {
    const label = elementText(node);
    if (!label) {
      return;
    }
    if (this.extractHeadings) {
      this.currentAnchor = anchor;
    }
    const index = this.commitParagraph(label);
    if (index === null || !this.extractHeadings) {
      return;
    }
    while (
      this.headingStack.length > 0 &&
      this.headingStack[this.headingStack.length - 1]![0] >= level
    ) {
      this.headingStack.pop();
    }
    this.headingStack.push([level, label]);
    const breadcrumb = fitBreadcrumb(
      this.headingStack.map(([, crumb]) => crumb),
      { separator: this.headingSeparator, maxBytes: MAX_SECTION_BYTES },
    );
    if (breadcrumb !== null) {
      this.sections.push(new SectionEntry({ paragraph: index, section: breadcrumb }));
    }
  }

  private handleTable(node: HtmlNode): void {
    const rows: string[] = [];
    for (const row of findAll(node, "tr", new Set(["table"]))) {
      const cells = row.children
        .filter(
          (cell): cell is HtmlNode =>
            typeof cell !== "string" && (cell.tag === "td" || cell.tag === "th"),
        )
        .map((cell) => elementText(cell, new Set(["table"])));
      if (cells.some((cell) => cell !== "")) {
        rows.push(cells.join(" | "));
      }
    }
    if (rows.length > 0) {
      this.commitParagraph(rows.join("\n"));
    }
  }
}

function extractBody(
  tree: HtmlNode,
  options: { extractHeadings: boolean; headingSeparator: string },
): { text: string; locators: LocatorEntry[]; sections: SectionEntry[]; hadDroppedFrame: boolean } {
  let root = findFirst(tree, "main") ?? findFirst(tree, "article");
  const dropHeaderFooter = root === null;
  if (root === null) {
    root = findFirst(tree, "body") ?? tree;
  }
  const extractor = new Extractor({
    extractHeadings: options.extractHeadings,
    headingSeparator: options.headingSeparator,
    dropHeaderFooter,
  });
  const text = extractor.run(root);
  return {
    text,
    locators: extractor.locators,
    sections: extractor.sections,
    hadDroppedFrame: extractor.hadDroppedFrame,
  };
}

// -- HtmlConnector ----------------------------------------------------------------

export interface HtmlConnectorOptions {
  /** Whether headings become breadcrumb `sections` and `fragment`
   * locators at all — `false` yields a plain body-text passage with
   * neither (headings still form their own paragraph either way).
   * Default `true`. */
  extractHeadings?: boolean;
  /** Joins a heading's ancestor chain into one `section` label. Default
   * `" > "`. */
  headingSeparator?: string;
  /** Caps the raw HTML checked before (local file) or during (URL,
   * streamed) reading. Default 16 MiB. */
  maxFileBytes?: number;
  /** Per-fetch (initial or redirect-hop) timeout, in seconds — see this
   * module's own header doc for how this approximates httpx's per-phase
   * timeout. Default 30. */
  timeout?: number;
  /** The separate ceiling on a URL fetch's total wall-clock time, in
   * seconds. Default 60. */
  maxTotalSeconds?: number;
  /** Names this connector to the server. Default
   * `taguru-html-connector/{PARSER_VERSION}`. */
  userAgent?: string;
  /** Disables the destination check documented in this module's own
   * header doc — set `true` only for a caller (or a test) that
   * intentionally fetches from a private/loopback address. Default
   * `false`. */
  allowPrivateNetworks?: boolean;
}

/**
 * Reference connector for `.html`/`.htm`/`.xhtml` files and `http(s)://`
 * URLs (issue #349). None of `timeout`/`maxTotalSeconds`/`userAgent`/
 * `allowPrivateNetworks` change a document's own content, so all four are
 * excluded from `fingerprintInputs.parseOptionsDigest`.
 */
export class HtmlConnector implements Connector {
  private readonly extractHeadings: boolean;
  private readonly headingSeparator: string;
  private readonly maxFileBytes: number;
  private readonly timeout: number;
  private readonly maxTotalSeconds: number;
  private readonly userAgent: string;
  private readonly allowPrivateNetworks: boolean;

  constructor(options: HtmlConnectorOptions = {}) {
    this.extractHeadings = options.extractHeadings ?? true;
    this.headingSeparator = options.headingSeparator ?? " > ";
    this.maxFileBytes = options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES;
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.maxTotalSeconds = options.maxTotalSeconds ?? DEFAULT_MAX_TOTAL_SECONDS;
    this.userAgent = options.userAgent ?? `taguru-html-connector/${PARSER_VERSION}`;
    this.allowPrivateNetworks = options.allowPrivateNetworks ?? false;
  }

  get parser(): string {
    return PARSER_NAME;
  }

  get parserVersion(): string {
    return PARSER_VERSION;
  }

  /**
   * The digest `read` would stamp into `fingerprintInputs.
   * parseOptionsDigest` for THIS instance's effective config, computable
   * without reading anything (ADR 0007 §6.3).
   */
  async parseOptionsDigest(): Promise<string> {
    return this.computeOptionsDigest();
  }

  supports(reference: string): boolean {
    if (isWindowsDrivePath(reference)) {
      return SUPPORTED_SUFFIXES.has(suffixOf(reference).toLowerCase());
    }
    const scheme = urlScheme(reference);
    if (SUPPORTED_URL_SCHEMES.has(scheme)) {
      return true;
    }
    if (scheme) {
      return false;
    }
    return SUPPORTED_SUFFIXES.has(suffixOf(reference).toLowerCase());
  }

  private async computeOptionsDigest(): Promise<string> {
    return optionsDigest({
      extract_headings: this.extractHeadings,
      heading_separator: this.headingSeparator,
      max_file_bytes: this.maxFileBytes,
    });
  }

  private async fingerprint(rawContentSha256: string): Promise<FingerprintInputs> {
    return new FingerprintInputs({
      rawContentSha256,
      parser: this.parser,
      parserVersion: this.parserVersion,
      parseOptionsDigest: await this.computeOptionsDigest(),
    });
  }

  private async failure(params: {
    source: string;
    displayName: string;
    code: DiagnosticCode;
    message: string;
    rawContentSha256: string;
  }): Promise<ConnectorDocument> {
    return new ConnectorDocument({
      source: params.source,
      text: "",
      metadata: new ConnectorMetadata({
        originUri: params.source,
        displayName: params.displayName,
        contentType: null,
      }),
      fingerprintInputs: await this.fingerprint(params.rawContentSha256),
      diagnostics: [new Diagnostic({ code: params.code, message: params.message, source: params.source })],
    });
  }

  async read(reference: string): Promise<ConnectorDocument> {
    if (isWindowsDrivePath(reference)) {
      return this.readFile(reference);
    }
    const scheme = urlScheme(reference);
    if (SUPPORTED_URL_SCHEMES.has(scheme)) {
      return this.readUrl(reference);
    }
    if (scheme) {
      return this.failure({
        source: reference,
        displayName: reference,
        code: "unsupported_format",
        message: `unsupported scheme '${scheme}' (only http/https URLs or a local .html/.htm/.xhtml path)`,
        rawContentSha256: EMPTY_SHA256,
      });
    }
    return this.readFile(reference);
  }

  private async readFile(reference: string): Promise<ConnectorDocument> {
    // `reference` itself is the source id, never a normalized rewrite of
    // it (ADR 0007 §6.1).
    const source = fileSourceId(reference);
    const displayName = basenameOf(reference);

    const failure = (code: DiagnosticCode, message: string, rawContentSha256 = EMPTY_SHA256) =>
      this.failure({ source, displayName, code, message, rawContentSha256 });

    const sourceDiagnostic = checkSourceId(source);
    if (sourceDiagnostic !== null) {
      return failure(sourceDiagnostic.code, sourceDiagnostic.message);
    }

    const suffix = suffixOf(reference).toLowerCase();
    if (!SUPPORTED_SUFFIXES.has(suffix)) {
      return failure(
        "unsupported_format",
        `unsupported extension '${suffix}' (only .html/.htm/.xhtml)`,
      );
    }

    const { stat, readFile } = await import("node:fs/promises");

    let size: number;
    try {
      size = (await stat(reference)).size;
    } catch (error) {
      return failure("unreadable", errorMessage(error));
    }
    if (size > this.maxFileBytes) {
      return failure(
        "content_too_large",
        `${size} bytes exceeds the ${this.maxFileBytes}-byte file cap`,
      );
    }

    let raw: Buffer;
    try {
      raw = await readFile(reference);
    } catch (error) {
      return failure("unreadable", errorMessage(error));
    }
    const rawContentSha256 = await sha256HexBytes(raw);

    return this.parseDocument({
      source,
      displayName,
      raw,
      rawContentSha256,
      contentTypeHeader: null,
      canonicalBase: null,
    });
  }

  private async readUrl(reference: string): Promise<ConnectorDocument> {
    const preSource = canonicalizeUrl(stripFragment(reference));

    const earlyFailure = (code: DiagnosticCode, message: string) =>
      this.failure({
        source: preSource,
        displayName: preSource,
        code,
        message,
        rawContentSha256: EMPTY_SHA256,
      });

    const preCheck = checkSourceId(preSource);
    if (preCheck !== null) {
      return earlyFailure(preCheck.code, preCheck.message);
    }

    const deadline = Date.now() + this.maxTotalSeconds * 1000;

    let response: Response;
    let finalUrl: string;
    try {
      const result = await this.fetchFollowingRedirects(reference, deadline);
      response = result.response;
      finalUrl = result.finalUrl;
    } catch (error) {
      return earlyFailure("unreadable", errorMessage(error));
    }

    // Resolved before ANY response-driven exit: after a redirect chain,
    // every failure from here on must name the URL that actually
    // answered, exactly like the success path and the size/deadline
    // failures below — "the source id is the final URL" is this module's
    // contract, and observability-side identity (e.g. duplicate_source
    // folding) depends on it.
    const responseFailure = (code: DiagnosticCode, message: string) =>
      this.failure({
        source: canonicalizeUrl(stripFragment(finalUrl)),
        displayName: urlDisplayName(finalUrl),
        code,
        message,
        rawContentSha256: EMPTY_SHA256,
      });

    if (response.status >= 400) {
      if (response.body) {
        await response.body.cancel().catch(() => {});
      }
      return responseFailure("unreadable", `HTTP ${response.status}`);
    }

    const contentTypeHeader = response.headers.get("content-type");
    if (contentTypeHeader !== null) {
      const mainType = contentTypeHeader.split(";")[0]!.trim().toLowerCase();
      if (!ALLOWED_CONTENT_TYPES.has(mainType)) {
        if (response.body) {
          await response.body.cancel().catch(() => {});
        }
        return responseFailure(
          "unsupported_format",
          `unsupported content type '${contentTypeHeader}' (expected text/html)`,
        );
      }
    }

    const chunks: Uint8Array[] = [];
    let total = 0;
    const reader = response.body?.getReader();
    if (reader) {
      for (;;) {
        let result: ReadableStreamReadResult<Uint8Array>;
        try {
          result = await reader.read();
        } catch (error) {
          return earlyFailure("unreadable", errorMessage(error));
        }
        if (result.done) {
          break;
        }
        chunks.push(result.value);
        total += result.value.length;
        if (total > this.maxFileBytes) {
          await reader.cancel().catch(() => {});
          return responseFailure(
            "content_too_large",
            `exceeds the ${this.maxFileBytes}-byte fetch cap`,
          );
        }
        if (Date.now() > deadline) {
          await reader.cancel().catch(() => {});
          return responseFailure(
            "unreadable",
            `the fetch exceeded the ${this.maxTotalSeconds}-second total time budget`,
          );
        }
      }
    }

    const raw = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
    const rawContentSha256 = await sha256HexBytes(raw);
    const source = canonicalizeUrl(stripFragment(finalUrl));
    const displayName = urlDisplayName(finalUrl);

    const sourceDiagnostic = checkSourceId(source);
    if (sourceDiagnostic !== null) {
      return this.failure({
        source,
        displayName,
        code: sourceDiagnostic.code,
        message: sourceDiagnostic.message,
        rawContentSha256,
      });
    }

    return this.parseDocument({
      source,
      displayName,
      raw,
      rawContentSha256,
      contentTypeHeader,
      canonicalBase: finalUrl,
    });
  }

  /**
   * Follows redirects by hand (`redirect: "manual"`) rather than via
   * `fetch`'s own `redirect: "follow"`, specifically so
   * `assertPublicDestination` can validate EACH hop's destination before
   * that hop's request is made — the same per-hop guarantee the Python
   * module's httpx `request` event hook gives.
   */
  private async fetchFollowingRedirects(
    startUrl: string,
    deadline: number,
  ): Promise<{ response: Response; finalUrl: string }> {
    let currentUrl = startUrl;
    for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
      // `maxTotalSeconds` is the TOTAL wall-clock budget for the fetch
      // (its JSDoc's contract) — without this per-hop check, a redirect
      // chain alone could spend up to (MAX_REDIRECTS + 1) × `timeout`
      // seconds before the body-read loop ever sees the deadline.
      const remainingMs = deadline - Date.now();
      if (remainingMs <= 0) {
        throw new Error(
          `the fetch exceeded the ${this.maxTotalSeconds}-second total time budget`,
        );
      }
      if (!this.allowPrivateNetworks) {
        await assertPublicDestination(currentUrl);
      }
      const { url: fetchUrl, authorizationHeader } = prepareFetchUrl(currentUrl);
      const headers: Record<string, string> = { "User-Agent": this.userAgent };
      if (authorizationHeader !== null) {
        headers["Authorization"] = authorizationHeader;
      }
      const response = await fetch(fetchUrl, {
        method: "GET",
        redirect: "manual",
        headers,
        signal: AbortSignal.timeout(Math.min(this.timeout * 1000, remainingMs)),
      });
      const location = response.headers.get("location");
      const isRedirect = REDIRECT_STATUSES.has(response.status) && location !== null;
      if (!isRedirect) {
        return { response, finalUrl: currentUrl };
      }
      if (response.body) {
        await response.body.cancel().catch(() => {});
      }
      currentUrl = resolveUrl(currentUrl, location);
    }
    throw new Error(`exceeded ${MAX_REDIRECTS} redirects`);
  }

  private async parseDocument(params: {
    source: string;
    displayName: string;
    raw: Buffer;
    rawContentSha256: string;
    contentTypeHeader: string | null;
    canonicalBase: string | null;
  }): Promise<ConnectorDocument> {
    const { source, displayName, raw, rawContentSha256, contentTypeHeader, canonicalBase } = params;

    const failure = (code: DiagnosticCode, message: string) =>
      this.failure({ source, displayName, code, message, rawContentSha256 });

    const charset = resolveCharset(raw, contentTypeHeader);
    let textRaw: string;
    try {
      textRaw = decodeBuffer(raw, charset);
    } catch (error) {
      return failure("corrupt", `failed to decode as '${charset}': ${errorMessage(error)}`);
    }
    // paragraph.rs's splitter does no normalization of its own — a
    // leading BOM would otherwise land inside paragraph 0 verbatim
    // (`TextDecoder` already strips one for UTF-8 by default; this is a
    // harmless belt-and-suspenders strip for every other charset).
    if (textRaw.startsWith("\ufeff")) {
      textRaw = textRaw.slice(1);
    }
    if (textRaw.includes("\u0000")) {
      // NUL never occurs in legitimate HTML (WHATWG treats it as a parse
      // error), and it survives a UTF-8 decode without an exception \u2014 its
      // presence means the charset guess was wrong (binary data, or an
      // undeclared multi-byte encoding the heuristics above did not
      // catch). Parsing on would see no tags, so `<script>` bodies would
      // leak into the body text; refuse with a diagnostic instead.
      return failure(
        "corrupt",
        "decoded text contains NUL characters \u2014 the byte stream is binary or its " +
          "character encoding was misdeclared",
      );
    }

    let bodyText: string;
    let locators: LocatorEntry[];
    let sections: SectionEntry[];
    let hadDroppedFrame: boolean;
    let title: string | null;
    let canonicalUrl: string | null;
    try {
      const builder = new TreeBuilder();
      builder.feed(textRaw);
      const tree = builder.root;
      const extracted = extractBody(tree, {
        extractHeadings: this.extractHeadings,
        headingSeparator: this.headingSeparator,
      });
      bodyText = extracted.text;
      locators = extracted.locators;
      sections = extracted.sections;
      hadDroppedFrame = extracted.hadDroppedFrame;
      title = extractTitle(tree);
      canonicalUrl = this.resolveCanonical(tree, canonicalBase);
    } catch (error) {
      if (error instanceof RangeError) {
        // A pathologically deep tree blows the call stack in every
        // recursive walk below the (iterative) tree builder — `read`
        // never throws for an ordinary parse failure; this is that
        // failure, structured.
        return failure("corrupt", "the markup nests too deeply to extract");
      }
      return failure("corrupt", errorMessage(error));
    }

    if (!bodyText) {
      return failure("ocr_required", "no extractable body text after boilerplate removal");
    }
    if (byteLen(bodyText) > MAX_PASSAGE_BYTES) {
      return failure(
        "content_too_large",
        `${byteLen(bodyText)} bytes of extracted text exceeds the ${MAX_PASSAGE_BYTES}-byte passage cap`,
      );
    }

    const diagnostics: Diagnostic[] = [];
    if (hadDroppedFrame) {
      diagnostics.push(
        new Diagnostic({
          code: "partial_extraction",
          message: "an <iframe>/<frame>'s content was not fetched and is excluded from text",
          source,
        }),
      );
    }

    return new ConnectorDocument({
      source,
      text: bodyText,
      locators,
      sections,
      metadata: new ConnectorMetadata({
        originUri: source,
        displayName,
        title,
        canonicalUrl,
        contentType: "text/html",
      }),
      fingerprintInputs: await this.fingerprint(rawContentSha256),
      diagnostics,
    });
  }

  private resolveCanonical(tree: HtmlNode, base: string | null): string | null {
    const href = extractCanonicalHref(tree);
    if (!href) {
      return null;
    }
    const resolved = base === null ? href : resolveUrl(base, href);
    if (!SUPPORTED_URL_SCHEMES.has(urlScheme(resolved))) {
      // A local-file read with a relative (or non-http(s)) canonical href
      // has no stable base worth resolving against — recorded as absent
      // rather than a meaningless value.
      return null;
    }
    return canonicalizeUrl(resolved);
  }
}
