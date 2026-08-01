"""The HTML connector (ADR 0007 §7/§8, issue #349): `.html`/`.htm`/`.xhtml`
files and `http(s)://` URLs into
:class:`~taguru_langchain.ingest_connectors.document.ConnectorDocument`.

Boilerplate (script/style/nav/aside/hidden regions, and a page's own
header/footer when no `<main>`/`<article>` scopes the content) is stripped
before `text` is built; the heading hierarchy survives as a breadcrumb
`section` per paragraph (`"Guide > Installation"` — `sections` is flat, so a
multi-level heading path is flattened into one label rather than dropped);
and each heading's own `id` (or its nearest `id`-bearing ancestor's) becomes
a `{"kind": "fragment", "value": ...}` locator on every paragraph from that
heading up to the next one — combined with `metadata.canonical_url`, a
citation can point at a real in-page deep link.

Parsing is `html.parser.HTMLParser` (stdlib) only — no `html` extra, no
BeautifulSoup/lxml/trafilatura — the same "minimal reference implementation,
not a full parser" posture `text.py` already takes for Markdown. A page
whose markup this parser cannot make sense of, or whose body carries no
extractable text once boilerplate is stripped (an image-only page, an
unrendered JS-shell SPA), degrades to a diagnostic (`corrupt`/`ocr_required`)
with empty `text` — never a raised exception, never a silently thin passage.

URL fetches use `httpx` (already a transitive dependency via `taguru`,
declared directly here since this module imports it) with
`follow_redirects=True`; the source id is the *final*, fragment-stripped,
canonicalized URL (ADR 0007 §6.1) — a page's own `<link rel="canonical">`,
when present, is recorded in `metadata.canonical_url` but never substituted
for the fetch URL as the source id, since a page can claim any canonical
and two distinct pages claiming the same one would otherwise collide.

Fetching an arbitrary `http(s)://` URL is only ever safe when the caller
controls (or trusts) the URL — `HtmlConnector` is not a defense against a
malicious *caller* feeding it a URL chosen by an untrusted third party
(user-submitted, scraped from an external feed, produced by an LLM). By
default, though, it still refuses a destination that resolves to a
private, loopback, link-local, multicast, or otherwise non-global IP
address — including one reached only after following a redirect — so an
*otherwise-trusted* URL cannot be turned into a probe of `localhost`, the
caller's own internal network, or a cloud metadata endpoint
(`169.254.169.254`) by a redirect the origin server controls. This check
resolves DNS itself and cannot fully close a same-instant DNS-rebinding
race against a hostile resolver; a deployment that fetches URLs supplied
by a genuinely untrusted party should not rely on it alone. Pass
`allow_private_networks=True` to disable the check entirely — this
connector's own tests need it to reach their loopback test server.

Out of scope, deliberately: ADR 0007 §7.3's `{"kind": "table", ...}`
locator. At most one locator is stored per paragraph (§7.2), and this
connector already spends that one slot on `fragment`; a `<table>` still
becomes its own paragraph (never folded into a neighbour, per §7.3) and
inherits whichever `fragment` locator its surrounding heading set, but gets
no locator of its own kind. A future connector revision can add
`{"kind": "table"}` (or a per-cell locator) once a caller needs
table-granular citations distinct from the enclosing section.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from taguru import Locator

from .._extract import MAX_PASSAGE_BYTES
from .document import (
    MAX_SECTION_BYTES,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    DiagnosticCode,
    FingerprintInputs,
    LocatorEntry,
    SectionEntry,
    options_digest,
)
from .sources import canonicalize_url, check_source_id, file_source_id

PARSER_NAME: Final = "taguru-html-connector"
PARSER_VERSION: Final = "1.0.0"

_SUPPORTED_SUFFIXES: Final = frozenset({".html", ".htm", ".xhtml"})
_SUPPORTED_URL_SCHEMES: Final = frozenset({"http", "https"})
_ALLOWED_CONTENT_TYPES: Final = frozenset({"text/html", "application/xhtml+xml"})

# A Windows drive-letter path (`C:\docs\a.html`, `C:/docs/a.html`) parses
# under `urlsplit` as scheme `"c"` — indistinguishable, by scheme alone,
# from a genuine (if unusual) single-letter URL scheme. Checked before any
# scheme dispatch so a Windows absolute path is never misrouted into the
# URL-fetch path or refused as `unsupported_format`.
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:[\\/]")

# The raw-file/fetch cap, checked before (local) or during (URL, streamed)
# reading — independent of MAX_PASSAGE_BYTES (checked again below, against
# the EXTRACTED text): markup overhead means a large-but-sparse page can
# stay well under the text cap, and a heavily-templated small page's raw
# bytes can still exceed a naive per-file cap. Same two-cap shape PdfConnector
# uses for the same reason.
_DEFAULT_MAX_FILE_BYTES: Final = 16 * 1024 * 1024
# httpx's own `timeout` bounds each individual phase (connect/write/read) —
# a server trickling a chunk every `timeout - epsilon` seconds forever never
# trips it. This is the separate ceiling on the fetch's total wall-clock
# time, enforced by hand in `_read_url` via `time.monotonic()`.
_DEFAULT_TIMEOUT: Final = 30.0
_DEFAULT_MAX_TOTAL_SECONDS: Final = 60.0

# Dropped entirely, with every descendant: never rendered content, never
# navigation chrome. `header`/`footer` are handled separately (below) since
# whether they count as boilerplate depends on whether a `<main>`/`<article>`
# already scopes the content.
_DROP_TAGS: Final = frozenset(
    {
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
    }
)
# Dropped only when no `<main>`/`<article>` was found to scope the content —
# inside one, a `<header>`/`<footer>` is the article's own (often holding its
# `<h1>` or a byline), not site chrome.
_CONDITIONAL_DROP_TAGS: Final = frozenset({"header", "footer"})
# ARIA landmark roles that are always chrome, regardless of tag name.
_ROLE_DENYLIST: Final = frozenset(
    {"navigation", "banner", "contentinfo", "complementary", "search"}
)
# Elements with no closing tag in ordinary (non-XHTML) HTML — never pushed
# onto the tree-builder's open-element stack.
_VOID_TAGS: Final = frozenset(
    {
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
    }
)
_HEADING_TAGS: Final = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# Block-level: each one flushes whatever inline text preceded it into its
# own paragraph, and is itself followed by a flush — so nesting (`<div><p>
# ...`) never bleeds one block's text into a sibling's. `tr` is included as
# a fallback for a stray `<tr>` outside any `<table>` (malformed markup);
# ordinarily table rows are consumed whole by the table handler below and
# never reach this generic dispatch.
_BLOCK_TAGS: Final = frozenset(
    {"p", "div", "section", "article", "li", "blockquote", "figcaption", "dt", "dd", "tr"}
    | _HEADING_TAGS
)

_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()

_WS_RE: Final = re.compile(r"\s+")
# Guarantees no paragraph this connector emits ever contains an internal
# blank-line run: `text` is later rebuilt as `"\n\n".join(paragraphs)` and
# re-split by the server's own blank-line-delimited rule
# (`taguru_langchain._extract.split_paragraphs`, mirroring src/paragraph.rs)
# — a stray blank line INSIDE what this connector considers one paragraph
# (e.g. two consecutive `<br><br>`, or a `<pre>` block with a blank line)
# would otherwise silently register as an extra paragraph boundary there,
# offsetting every locator/section paragraph index after it. Collapsing to
# a single `\n` keeps the line break (an interior line break stays IN a
# paragraph, exactly as `<br>` intends) without ever producing that boundary.
_BLANK_RUN_RE: Final = re.compile(r"\n\s*\n+")
_CHARSET_RE: Final = re.compile(rb'charset\s*=\s*["\']?\s*([\w-]+)', re.IGNORECASE)
_HEADER_CHARSET_RE: Final = re.compile(r'charset\s*=\s*["\']?\s*([\w-]+)', re.IGNORECASE)


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _strip_fragment(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _url_display_name(url: str) -> str:
    parts = urlsplit(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    if segments:
        return segments[-1]
    return parts.netloc or url


def _is_windows_drive_path(reference: str) -> bool:
    return _WINDOWS_DRIVE_RE.match(reference) is not None


class _BlockedDestination(Exception):
    """Raised from `_validate_public_destination` (an httpx `request` event
    hook) to abort a fetch whose destination resolved to a non-global IP
    address — caught in `_read_url` alongside `httpx.HTTPError` and mapped
    to the same `unreadable` diagnostic every other fetch failure gets."""


def _is_blocked_ip(raw_address: str) -> bool:
    address = raw_address.split("%", 1)[0]  # drop a getaddrinfo IPv6 zone id
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False  # unparseable: not this check's job to fail the fetch
    # `is_global` already excludes private/loopback/link-local/reserved/
    # unspecified (covering the cloud metadata endpoint 169.254.169.254,
    # which is link-local); multicast is the one range stdlib still counts
    # as "global" that is never a sane HTTP destination, so it needs its
    # own explicit check.
    return not ip.is_global or ip.is_multicast


def _validate_public_destination(request: httpx.Request) -> None:
    """An httpx `request` event hook — fires for the initial request AND
    for every request a followed redirect issues, so a redirect to a
    private/internal address is caught exactly like a direct one (verified:
    httpx invokes `request` hooks per hop, not once per `client.stream`
    call). A hostname that fails to resolve is left alone here — the
    connection attempt that follows will fail on its own, through the
    ordinary `httpx.HTTPError` path, so this hook only has an opinion once
    resolution actually names a destination."""
    host = request.url.host
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    try:
        resolved = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return
    for _family, _kind, _proto, _canon, sockaddr in resolved:
        address = str(sockaddr[0])
        if _is_blocked_ip(address):
            raise _BlockedDestination(
                f"refusing to fetch {host!r}: resolves to a private/internal address"
            )


def _resolve_charset(raw: bytes, content_type_header: str | None) -> str:
    """`Content-Type` header `charset=` first, else an ASCII-safe scan of a
    `<meta charset=...>`/`<meta http-equiv="Content-Type" content="...
    charset=...">` tag in the first 2 KiB (where HTML5 requires one to
    live), else UTF-8. A BOM is handled separately, after decoding (see
    `read`'s `str.removeprefix("\\ufeff")`, the same convention `text.py`
    already uses) rather than sniffed here — simpler, and correct for the
    overwhelmingly common UTF-8(-with-BOM) case this reference connector
    targets."""
    if content_type_header:
        header_match = _HEADER_CHARSET_RE.search(content_type_header)
        if header_match:
            return header_match.group(1)
    meta_match = _CHARSET_RE.search(raw[:2048])
    if meta_match:
        return meta_match.group(1).decode("ascii", errors="ignore")
    return "utf-8"


def _sanitize_paragraph_text(text: str) -> str:
    return _BLANK_RUN_RE.sub("\n", text).strip()


class _Node:
    """One element in the tree `_TreeBuilder` assembles — attribute names are
    already lowercased by `HTMLParser`; values are compared case-
    insensitively where HTML itself is case-insensitive (`rel`, `role`,
    `aria-hidden`)."""

    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs: dict[str, str]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[_Node | str] = []


class _TreeBuilder(HTMLParser):
    """Builds a minimal DOM from HTML text (stdlib `html.parser.HTMLParser`
    underneath), tolerant of malformed markup: an unmatched close tag is
    ignored rather than raised (the same "degrade quietly, never
    incorrectly" posture `text.py`'s heading regex already takes) — real,
    hand-authored HTML on the web is routinely not well-formed, and a
    connector that raises on every such page would defeat its own purpose.

    `html.parser` itself is lenient but not contractually exception-free on
    adversarial input; any `feed` that does raise is caught by `read`'s own
    `corrupt`-diagnostic handling, the same boundary `PdfConnector` draws
    around `pypdf`.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#document", {})
        self._stack: list[_Node] = [self.root]

    def feed(self, text: str) -> None:
        super().feed(text)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, {name: value if value is not None else "" for name, value in attrs})
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag, {name: value if value is not None else "" for name, value in attrs})
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return
        # No matching open tag on the stack: malformed markup, ignored.

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def _find_first(node: _Node, tag: str) -> _Node | None:
    for child in node.children:
        if isinstance(child, str):
            continue
        if child.tag == tag:
            return child
        found = _find_first(child, tag)
        if found is not None:
            return found
    return None


def _find_all(node: _Node, tag: str, stop_tags: frozenset[str] = frozenset()) -> list[_Node]:
    results: list[_Node] = []
    for child in node.children:
        if isinstance(child, str) or child.tag in stop_tags:
            continue
        if child.tag == tag:
            results.append(child)
        results.extend(_find_all(child, tag, stop_tags))
    return results


def _element_text(node: _Node, *, stop_tags: frozenset[str] = frozenset()) -> str:
    """Whitespace-collapsed concatenation of every descendant text node,
    skipping `_DROP_TAGS` (so e.g. a `<title>` never absorbs stray
    `<script>` content some templating leaves inside it) and, when passed,
    never descending into `stop_tags` — used for a table cell's own text
    (`stop_tags={"table"}`) so a nested table's cells are never glued into
    the outer cell's text; a nested table is unsupported by this reference
    connector and its content is dropped rather than merged in confusingly
    (the same "never double up nested table content" reasoning
    `_handle_table`'s own `_find_all(..., stop_tags={"table"})` call
    already applies to row-finding)."""
    parts: list[str] = []

    def visit(current: _Node) -> None:
        for child in current.children:
            if isinstance(child, str):
                parts.append(child)
            elif child.tag not in _DROP_TAGS and child.tag not in stop_tags:
                visit(child)

    visit(node)
    return _WS_RE.sub(" ", "".join(parts)).strip()


def _extract_title(tree: _Node) -> str | None:
    title_node = _find_first(tree, "title")
    if title_node is not None:
        text = _element_text(title_node)
        if text:
            return text
    h1 = _find_first(tree, "h1")
    if h1 is not None:
        text = _element_text(h1)
        if text:
            return text
    return None


def _extract_canonical_href(tree: _Node) -> str | None:
    for link in _find_all(tree, "link"):
        rel_values = {token.lower() for token in link.attrs.get("rel", "").split()}
        if "canonical" in rel_values:
            href = link.attrs.get("href")
            if href:
                return href
    return None


class _Extractor:
    """One pass over the content root: builds paragraph-joined `text` plus
    paragraph-indexed `locators`/`sections`, mirroring exactly the shape
    `taguru_langchain._extract.split_paragraphs` will re-derive from the
    final `"\\n\\n".join(paragraphs)` text — never the reverse (this
    connector builds paragraphs FROM the DOM's own structure, then never
    calls the splitter itself; `tests/unit/test_html_connector.py`'s resplit
    invariant is what keeps the two in lockstep, the same invariant
    `test_pdf_connector.py` already asserts for page locators).
    """

    def __init__(
        self, *, extract_headings: bool, heading_separator: str, drop_header_footer: bool
    ) -> None:
        self._extract_headings = extract_headings
        self._heading_separator = heading_separator
        self._drop_header_footer = drop_header_footer
        self.paragraphs: list[str] = []
        self.locators: list[LocatorEntry] = []
        self.sections: list[SectionEntry] = []
        self.had_dropped_frame = False
        self._heading_stack: list[tuple[int, str]] = []
        self._current_anchor: str | None = None
        self._buffer: list[str] = []

    def run(self, root: _Node) -> str:
        self._visit(root, root.attrs.get("id"))
        self._flush_buffer()
        return "\n\n".join(self.paragraphs)

    def _should_drop(self, node: _Node) -> bool:
        if node.tag in _DROP_TAGS:
            return True
        if node.tag in _CONDITIONAL_DROP_TAGS and self._drop_header_footer:
            return True
        if "hidden" in node.attrs:
            return True
        if node.attrs.get("aria-hidden", "").strip().lower() == "true":
            return True
        if node.attrs.get("role", "").strip().lower() in _ROLE_DENYLIST:
            return True
        return False

    def _commit_paragraph(self, text: str) -> int | None:
        """Appends `text` as a new paragraph (attaching the current
        section's `fragment` locator, if any) and returns its index — or
        `None` if `text` sanitized to nothing, in which case no paragraph
        was appended."""
        text = _sanitize_paragraph_text(text)
        if not text:
            return None
        index = len(self.paragraphs)
        self.paragraphs.append(text)
        if self._current_anchor is not None:
            self.locators.append(
                LocatorEntry(
                    paragraph=index, locator=Locator(kind="fragment", value=self._current_anchor)
                )
            )
        return index

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer = []
        self._commit_paragraph(text)

    def _visit(self, node: _Node, ambient_id: str | None) -> None:
        if self._should_drop(node):
            if node.tag in ("iframe", "frame"):
                self.had_dropped_frame = True
            return
        own_id = node.attrs.get("id")
        effective_id = own_id if own_id else ambient_id
        tag = node.tag
        if tag == "br":
            self._buffer.append("\n")
            return
        if tag in _HEADING_TAGS:
            self._flush_buffer()
            self._handle_heading(node, int(tag[1]), effective_id)
            return
        if tag == "table":
            self._flush_buffer()
            self._handle_table(node)
            return
        if tag == "pre":
            self._flush_buffer()
            self._commit_paragraph(self._pre_text(node))
            return
        if tag in _BLOCK_TAGS:
            self._flush_buffer()
            for child in node.children:
                self._dispatch_child(child, effective_id)
            self._flush_buffer()
            return
        # Inline (or unrecognized) tag: no paragraph boundary of its own.
        for child in node.children:
            self._dispatch_child(child, effective_id)

    def _dispatch_child(self, child: _Node | str, ambient_id: str | None) -> None:
        if isinstance(child, str):
            self._buffer.append(_WS_RE.sub(" ", child))
        else:
            self._visit(child, ambient_id)

    def _pre_text(self, node: _Node) -> str:
        """Raw text, whitespace UNCOLLAPSED (unlike every other paragraph
        kind) — a `<pre>` block's own line breaks and indentation are
        content, not formatting. Still passed through
        `_sanitize_paragraph_text` like every paragraph, so an internal
        blank line collapses to a single `\\n` rather than covertly
        splitting into two paragraphs on resplit; this is `<pre>`'s one
        documented fidelity trade-off against that invariant."""
        parts: list[str] = []

        def visit(current: _Node) -> None:
            for child in current.children:
                if isinstance(child, str):
                    parts.append(child)
                elif not self._should_drop(child):
                    if child.tag == "br":
                        parts.append("\n")
                    else:
                        visit(child)

        visit(node)
        return "".join(parts)

    def _handle_heading(self, node: _Node, level: int, anchor: str | None) -> None:
        label = _element_text(node)
        if not label:
            return
        if self._extract_headings:
            self._current_anchor = anchor
        index = self._commit_paragraph(label)
        if index is None or not self._extract_headings:
            return
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((level, label))
        breadcrumb = self._fit_breadcrumb([crumb for _, crumb in self._heading_stack])
        if breadcrumb is not None:
            self.sections.append(SectionEntry(paragraph=index, section=breadcrumb))

    def _fit_breadcrumb(self, crumbs: list[str]) -> str | None:
        while crumbs:
            candidate = self._heading_separator.join(crumbs)
            if _byte_len(candidate) <= MAX_SECTION_BYTES:
                return candidate
            crumbs = crumbs[1:]  # drop the outermost (least specific) ancestor first
        return None

    def _handle_table(self, node: _Node) -> None:
        rows: list[str] = []
        for row in _find_all(node, "tr", stop_tags=frozenset({"table"})):
            cells = [
                _element_text(cell, stop_tags=frozenset({"table"}))
                for cell in row.children
                if isinstance(cell, _Node) and cell.tag in ("td", "th")
            ]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            self._commit_paragraph("\n".join(rows))


def _extract_body(
    tree: _Node, *, extract_headings: bool, heading_separator: str
) -> tuple[str, list[LocatorEntry], list[SectionEntry], bool]:
    root = _find_first(tree, "main") or _find_first(tree, "article")
    drop_header_footer = root is None
    if root is None:
        root = _find_first(tree, "body") or tree
    extractor = _Extractor(
        extract_headings=extract_headings,
        heading_separator=heading_separator,
        drop_header_footer=drop_header_footer,
    )
    text = extractor.run(root)
    return text, extractor.locators, extractor.sections, extractor.had_dropped_frame


class HtmlConnector:
    """Reference connector for `.html`/`.htm`/`.xhtml` files and
    `http(s)://` URLs (issue #349).

    `extract_headings` (default `True`) controls whether headings become
    breadcrumb `sections` and `fragment` locators at all — `False` yields a
    plain body-text passage with neither (headings still form their own
    paragraph either way). `heading_separator` (default `" > "`) joins a
    heading's ancestor chain into one `section` label. `max_file_bytes`
    (default 16 MiB) caps the raw HTML checked before (local file) or during
    (URL, streamed) reading. `timeout` (default 30s) is httpx's own
    per-phase (connect/write/read) timeout; `max_total_seconds` (default
    60s) is the separate ceiling on the fetch's total wall-clock time, since
    a per-phase timeout alone never trips against a server trickling data
    slower than that but without ever going fully idle. `user_agent` names
    this connector to the server. `allow_private_networks` (default
    `False`) disables the destination check documented in this module's own
    docstring — set `True` only for a caller (or a test) that intentionally
    fetches from a private/loopback address. None of `timeout`/
    `max_total_seconds`/`user_agent`/`allow_private_networks` change a
    document's own content, so all four are excluded from
    `fingerprint_inputs.parse_options_digest`.
    """

    def __init__(
        self,
        *,
        extract_headings: bool = True,
        heading_separator: str = " > ",
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        timeout: float = _DEFAULT_TIMEOUT,
        max_total_seconds: float = _DEFAULT_MAX_TOTAL_SECONDS,
        user_agent: str = f"taguru-html-connector/{PARSER_VERSION}",
        allow_private_networks: bool = False,
    ) -> None:
        self._extract_headings = extract_headings
        self._heading_separator = heading_separator
        self._max_file_bytes = max_file_bytes
        self._timeout = timeout
        self._max_total_seconds = max_total_seconds
        self._user_agent = user_agent
        self._allow_private_networks = allow_private_networks

    @property
    def parser(self) -> str:
        return PARSER_NAME

    @property
    def parser_version(self) -> str:
        return PARSER_VERSION

    def supports(self, reference: str) -> bool:
        if _is_windows_drive_path(reference):
            return Path(reference).suffix.lower() in _SUPPORTED_SUFFIXES
        scheme = urlsplit(reference).scheme.lower()
        if scheme in _SUPPORTED_URL_SCHEMES:
            return True
        if scheme:
            return False
        return Path(reference).suffix.lower() in _SUPPORTED_SUFFIXES

    def _options_digest(self) -> str:
        return options_digest(
            {
                "extract_headings": self._extract_headings,
                "heading_separator": self._heading_separator,
                "max_file_bytes": self._max_file_bytes,
            }
        )

    def _fingerprint(self, raw_content_sha256: str) -> FingerprintInputs:
        return FingerprintInputs(
            raw_content_sha256=raw_content_sha256,
            parser=self.parser,
            parser_version=self.parser_version,
            parse_options_digest=self._options_digest(),
        )

    def _failure(
        self,
        *,
        source: str,
        display_name: str,
        code: DiagnosticCode,
        message: str,
        raw_content_sha256: str,
    ) -> ConnectorDocument:
        return ConnectorDocument(
            source=source,
            text="",
            metadata=ConnectorMetadata(
                origin_uri=source, display_name=display_name, content_type=None
            ),
            fingerprint_inputs=self._fingerprint(raw_content_sha256),
            diagnostics=(Diagnostic(code=code, message=message, source=source),),
        )

    def read(self, reference: str) -> ConnectorDocument:
        if _is_windows_drive_path(reference):
            return self._read_file(reference)
        scheme = urlsplit(reference).scheme.lower()
        if scheme in _SUPPORTED_URL_SCHEMES:
            return self._read_url(reference)
        if scheme:
            return self._failure(
                source=reference,
                display_name=reference,
                code="unsupported_format",
                message=(
                    f"unsupported scheme {scheme!r} (only http/https URLs or a local "
                    ".html/.htm/.xhtml path)"
                ),
                raw_content_sha256=_EMPTY_SHA256,
            )
        return self._read_file(reference)

    def _read_file(self, reference: str) -> ConnectorDocument:
        # `reference` itself is the source id, never a `Path`-normalized
        # rewrite of it — the same identity rule `text.py`/`pdf.py` already
        # follow (ADR 0007 §6.1).
        path = Path(reference)
        source = file_source_id(reference)
        display_name = path.name

        def failure(
            code: DiagnosticCode, message: str, raw_content_sha256: str = _EMPTY_SHA256
        ) -> ConnectorDocument:
            return self._failure(
                source=source,
                display_name=display_name,
                code=code,
                message=message,
                raw_content_sha256=raw_content_sha256,
            )

        source_diagnostic = check_source_id(source)
        if source_diagnostic is not None:
            return failure(source_diagnostic.code, source_diagnostic.message)

        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            return failure(
                "unsupported_format",
                f"unsupported extension {path.suffix!r} (only .html/.htm/.xhtml)",
            )

        try:
            size = path.stat().st_size
        except OSError as error:
            return failure("unreadable", str(error))
        if size > self._max_file_bytes:
            return failure(
                "content_too_large",
                f"{size} bytes exceeds the {self._max_file_bytes}-byte file cap",
            )

        try:
            raw = path.read_bytes()
        except OSError as error:
            return failure("unreadable", str(error))
        raw_content_sha256 = hashlib.sha256(raw).hexdigest()

        return self._parse(
            source=source,
            display_name=display_name,
            raw=raw,
            raw_content_sha256=raw_content_sha256,
            content_type_header=None,
            canonical_base=None,
        )

    def _read_url(self, reference: str) -> ConnectorDocument:
        pre_source = canonicalize_url(_strip_fragment(reference))

        def early_failure(code: DiagnosticCode, message: str) -> ConnectorDocument:
            return self._failure(
                source=pre_source,
                display_name=pre_source,
                code=code,
                message=message,
                raw_content_sha256=_EMPTY_SHA256,
            )

        pre_check = check_source_id(pre_source)
        if pre_check is not None:
            return early_failure(pre_check.code, pre_check.message)

        if self._allow_private_networks:
            client = httpx.Client(follow_redirects=True, timeout=self._timeout)
        else:
            client = httpx.Client(
                follow_redirects=True,
                timeout=self._timeout,
                event_hooks={"request": [_validate_public_destination]},
            )

        deadline = time.monotonic() + self._max_total_seconds
        try:
            with client:
                headers = {"User-Agent": self._user_agent}
                with client.stream("GET", reference, headers=headers) as response:
                    if response.status_code >= 400:
                        return early_failure("unreadable", f"HTTP {response.status_code}")
                    content_type_header = response.headers.get("content-type")
                    if content_type_header is not None:
                        main_type = content_type_header.split(";", 1)[0].strip().lower()
                        if main_type not in _ALLOWED_CONTENT_TYPES:
                            return early_failure(
                                "unsupported_format",
                                f"unsupported content type {content_type_header!r} "
                                "(expected text/html)",
                            )
                    final_url = str(response.url)
                    chunks = bytearray()
                    for chunk in response.iter_bytes():
                        chunks += chunk
                        if len(chunks) > self._max_file_bytes:
                            return self._failure(
                                source=canonicalize_url(_strip_fragment(final_url)),
                                display_name=_url_display_name(final_url),
                                code="content_too_large",
                                message=f"exceeds the {self._max_file_bytes}-byte fetch cap",
                                raw_content_sha256=_EMPTY_SHA256,
                            )
                        if time.monotonic() > deadline:
                            return self._failure(
                                source=canonicalize_url(_strip_fragment(final_url)),
                                display_name=_url_display_name(final_url),
                                code="unreadable",
                                message=(
                                    f"the fetch exceeded the {self._max_total_seconds}-second "
                                    "total time budget"
                                ),
                                raw_content_sha256=_EMPTY_SHA256,
                            )
        except (httpx.HTTPError, httpx.InvalidURL, _BlockedDestination) as error:
            return early_failure("unreadable", str(error))

        raw = bytes(chunks)
        raw_content_sha256 = hashlib.sha256(raw).hexdigest()
        source = canonicalize_url(_strip_fragment(final_url))
        display_name = _url_display_name(final_url)

        source_diagnostic = check_source_id(source)
        if source_diagnostic is not None:
            return self._failure(
                source=source,
                display_name=display_name,
                code=source_diagnostic.code,
                message=source_diagnostic.message,
                raw_content_sha256=raw_content_sha256,
            )

        return self._parse(
            source=source,
            display_name=display_name,
            raw=raw,
            raw_content_sha256=raw_content_sha256,
            content_type_header=content_type_header,
            canonical_base=final_url,
        )

    def _parse(
        self,
        *,
        source: str,
        display_name: str,
        raw: bytes,
        raw_content_sha256: str,
        content_type_header: str | None,
        canonical_base: str | None,
    ) -> ConnectorDocument:
        def failure(code: DiagnosticCode, message: str) -> ConnectorDocument:
            return self._failure(
                source=source,
                display_name=display_name,
                code=code,
                message=message,
                raw_content_sha256=raw_content_sha256,
            )

        charset = _resolve_charset(raw, content_type_header)
        try:
            text_raw = raw.decode(charset)
        except (LookupError, UnicodeDecodeError) as error:
            return failure("corrupt", f"failed to decode as {charset!r}: {error}")
        # paragraph.rs's splitter does no normalization of its own — a
        # leading BOM would otherwise land inside paragraph 0 verbatim,
        # exactly the case `text.py` already guards.
        text_raw = text_raw.removeprefix("﻿")

        try:
            builder = _TreeBuilder()
            builder.feed(text_raw)
            tree = builder.root
            body_text, locators, sections, had_dropped_frame = _extract_body(
                tree,
                extract_headings=self._extract_headings,
                heading_separator=self._heading_separator,
            )
            title = _extract_title(tree)
            canonical_url = self._resolve_canonical(tree, canonical_base)
        except RecursionError:
            # A pathologically deep tree (thousands of nested <div>s) builds
            # fine — `_TreeBuilder` is iterative — but every walk below it
            # (`_extract_body`, `_extract_title`, `_resolve_canonical`) is
            # recursive and can blow the interpreter's stack. `read` never
            # raises for an ordinary parse failure (`protocol.py`'s
            # `Connector.read` contract); this is that failure, structured.
            return failure("corrupt", "the markup nests too deeply to extract")
        except Exception as error:  # noqa: BLE001 - html.parser is lenient
            # but not contractually exception-free on adversarial input;
            # any failure here is this document's own markup failing to
            # parse, the same bucket pypdf's PyPdfError lands in for
            # PdfConnector.
            return failure("corrupt", str(error))

        if not body_text:
            return failure("ocr_required", "no extractable body text after boilerplate removal")
        if _byte_len(body_text) > MAX_PASSAGE_BYTES:
            return failure(
                "content_too_large",
                f"{_byte_len(body_text)} bytes of extracted text exceeds the "
                f"{MAX_PASSAGE_BYTES}-byte passage cap",
            )

        diagnostics: list[Diagnostic] = []
        if had_dropped_frame:
            diagnostics.append(
                Diagnostic(
                    code="partial_extraction",
                    message=(
                        "an <iframe>/<frame>'s content was not fetched and is excluded from text"
                    ),
                    source=source,
                )
            )

        return ConnectorDocument(
            source=source,
            text=body_text,
            locators=tuple(locators),
            sections=tuple(sections),
            metadata=ConnectorMetadata(
                origin_uri=source,
                display_name=display_name,
                title=title,
                canonical_url=canonical_url,
                content_type="text/html",
            ),
            fingerprint_inputs=self._fingerprint(raw_content_sha256),
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def _resolve_canonical(tree: _Node, base: str | None) -> str | None:
        href = _extract_canonical_href(tree)
        if not href:
            return None
        resolved = href if base is None else urljoin(base, href)
        if urlsplit(resolved).scheme.lower() not in _SUPPORTED_URL_SCHEMES:
            # A local-file read with a relative (or non-http(s)) canonical
            # href has no stable base worth resolving against — recorded as
            # absent rather than a meaningless value.
            return None
        return canonicalize_url(resolved)
