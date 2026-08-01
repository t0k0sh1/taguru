"""HtmlConnector — the local-file half of the `.html`/`.htm`/`.xhtml`
connector (ADR 0007 §7/§8, issue #349). URL-fetch behavior is
`test_html_connector_fetch.py`'s own file."""

from __future__ import annotations

import hashlib
from pathlib import Path

from taguru import Locator

from taguru_langchain._extract import MAX_PASSAGE_BYTES, split_paragraphs
from taguru_langchain.ingest_connectors import HtmlConnector
from taguru_langchain.ingest_connectors.document import MAX_SECTION_BYTES


def _write(tmp_path: Path, name: str, html: str | bytes) -> Path:
    path = tmp_path / name
    if isinstance(html, str):
        path.write_text(html, encoding="utf-8")
    else:
        path.write_bytes(html)
    return path


def test_boilerplate_is_removed_and_main_is_preferred(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><head><title>T</title></head><body>
        <nav>Site nav</nav>
        <header>Site banner</header>
        <main><p>Real content.</p></main>
        <aside>Related links</aside>
        <footer>Site footer</footer>
        <script>doStuff();</script>
        <style>.x{color:red}</style>
        </body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Real content."


def test_headings_become_breadcrumb_sections_and_fragment_locators(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main>
        <h1 id="top">Guide</h1>
        <p>Intro.</p>
        <h2 id="install">Installation</h2>
        <p>Step one.</p>
        <p>Step two.</p>
        </main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert [(s.paragraph, s.section) for s in document.sections] == [
        (0, "Guide"),
        (2, "Guide > Installation"),
    ]
    assert [(entry.paragraph, entry.locator) for entry in document.locators] == [
        (0, Locator(kind="fragment", value="top")),
        (1, Locator(kind="fragment", value="top")),
        (2, Locator(kind="fragment", value="install")),
        (3, Locator(kind="fragment", value="install")),
        (4, Locator(kind="fragment", value="install")),
    ]


def test_heading_without_id_falls_back_to_the_nearest_enclosing_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main>
        <section id="sec-1"><h2>No id of its own</h2><p>Body text.</p></section>
        </main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert {entry.locator.value for entry in document.locators} == {"sec-1"}


def test_extract_headings_false_disables_sections_and_locators(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main><h1 id="top">Guide</h1><p>Body.</p></main></body></html>""",
    )
    document = HtmlConnector(extract_headings=False).read(str(path))

    assert document.sections == ()
    assert document.locators == ()
    assert document.text == "Guide\n\nBody."


def test_title_falls_back_to_first_h1_when_no_title_tag(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "doc.html", """<html><body><main><h1>Fallback Title</h1></main></body></html>"""
    )
    document = HtmlConnector().read(str(path))

    assert document.metadata.title == "Fallback Title"


def test_canonical_link_is_honored_only_when_absolute(tmp_path: Path) -> None:
    absolute = _write(
        tmp_path,
        "abs.html",
        """<html><head><link rel="canonical" href="https://example.com/guide">
        </head><body><main><p>Text.</p></main></body></html>""",
    )
    relative = _write(
        tmp_path,
        "rel.html",
        """<html><head><link rel="canonical" href="/guide">
        </head><body><main><p>Text.</p></main></body></html>""",
    )

    assert HtmlConnector().read(str(absolute)).metadata.canonical_url == "https://example.com/guide"
    assert HtmlConnector().read(str(relative)).metadata.canonical_url is None


def test_br_is_an_interior_line_break_not_a_paragraph_boundary(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.html", """<html><body><p>Line one<br>Line two</p></body></html>""")
    document = HtmlConnector().read(str(path))

    assert document.text == "Line one\nLine two"
    assert split_paragraphs(document.text) == ["Line one\nLine two"]


def test_double_br_never_creates_a_false_paragraph_boundary(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.html", """<html><body><p>Before<br><br>After</p></body></html>""")
    document = HtmlConnector().read(str(path))

    # A literal blank line here would silently register as a second
    # paragraph on resplit, offsetting every locator/section index after it
    # — the exact hazard `_sanitize_paragraph_text` exists to rule out.
    assert split_paragraphs(document.text) == [document.text]


def test_pre_preserves_whitespace_but_collapses_an_internal_blank_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        "<html><body><pre>line one\n\n\nline two</pre></body></html>",
    )
    document = HtmlConnector().read(str(path))

    assert document.text == "line one\nline two"


def test_table_becomes_one_paragraph_with_rows_and_cells_joined(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main><table>
        <tr><th>Name</th><th>Value</th></tr>
        <tr><td>a</td><td>1</td></tr>
        </table></main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert document.text == "Name | Value\na | 1"


def test_nested_table_content_is_dropped_not_glued_into_the_outer_cell(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main><table>
        <tr><td>outer<table><tr><td>inner</td></tr></table></td></tr>
        </table></main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    # Nested tables are unsupported: the inner table's cell never leaks into
    # the outer cell's own text, and is never independently emitted either.
    assert document.text == "outer"


def test_resplit_invariant_holds_for_a_mixed_document(tmp_path: Path) -> None:
    """The invariant this connector's correctness rests on (mirrors
    test_pdf_connector.py's page-locator equivalent): re-running the
    server's own paragraph splitter over `text` must reproduce exactly the
    paragraphs `locators`/`sections` index — otherwise a citation silently
    points at the wrong paragraph once the batch reaches `taguru import`."""
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main>
        <h1 id="top">Title</h1>
        <p>First<br>paragraph.</p>
        <pre>pre\n\nblock</pre>
        <table><tr><td>c1</td><td>c2</td></tr></table>
        <h2 id="next">Next</h2>
        <p>Last paragraph.</p>
        </main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    resplit = split_paragraphs(document.text)
    assert "\n\n".join(resplit) == document.text
    for entry in document.locators:
        assert 0 <= entry.paragraph < len(resplit)
    for section in document.sections:
        assert 0 <= section.paragraph < len(resplit)


def test_hidden_aria_hidden_and_denylisted_role_elements_are_dropped(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main>
        <p>Visible.</p>
        <div hidden>Hidden attr.</div>
        <div aria-hidden="true">Aria hidden.</div>
        <div role="navigation">Role nav.</div>
        <div role="search">Role search.</div>
        </main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert document.text == "Visible."


def test_header_and_footer_are_kept_inside_article_but_dropped_at_body_level(
    tmp_path: Path,
) -> None:
    with_article = _write(
        tmp_path,
        "article.html",
        """<html><body><article><header><h1>Byline header</h1></header>
        <p>Body.</p></article></body></html>""",
    )
    without_scope = _write(
        tmp_path,
        "plain.html",
        """<html><body><header>Site banner</header><p>Body.</p></body></html>""",
    )

    assert "Byline header" in HtmlConnector().read(str(with_article)).text
    assert HtmlConnector().read(str(without_scope)).text == "Body."


def test_malformed_markup_degrades_quietly_instead_of_raising(tmp_path: Path) -> None:
    # The stray unmatched </div> is ignored rather than raised.
    path = _write(tmp_path, "doc.html", "<html><body><p>Hello</div><p>World</p></body></html>")
    document = HtmlConnector().read(str(path))

    assert document.diagnostics == ()
    assert "Hello" in document.text
    assert "World" in document.text


def test_iframe_content_is_excluded_and_reported_as_partial_extraction(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><main><p>Real text.</p>
        <iframe src="https://example.com/embed"></iframe>
        </main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert document.text == "Real text."
    assert [d.code for d in document.diagnostics] == ["partial_extraction"]


def test_empty_body_after_boilerplate_removal_is_ocr_required(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.html",
        """<html><body><nav>Nav only</nav><script>x()</script></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["ocr_required"]


def test_bom_is_stripped_from_the_first_paragraph(tmp_path: Path) -> None:
    path = tmp_path / "doc.html"
    path.write_bytes("<html><body><p>Text.</p></body></html>".encode("utf-8-sig"))
    document = HtmlConnector().read(str(path))

    assert document.text == "Text."


def test_meta_charset_is_honored_for_non_utf8_content(tmp_path: Path) -> None:
    path = tmp_path / "doc.html"
    html = '<html><head><meta charset="iso-8859-1"></head><body><p>café</p></body></html>'
    path.write_bytes(html.encode("iso-8859-1"))
    document = HtmlConnector().read(str(path))

    assert document.text == "café"


def test_undecodable_bytes_under_the_declared_charset_are_reported_corrupt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doc.html"
    header = b'<html><head><meta charset="utf-8"></head><body><p>'
    path.write_bytes(header + b"\xff\xfe" + b"</p></body></html>")
    document = HtmlConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["corrupt"]


def test_section_breadcrumb_over_cap_drops_the_outermost_crumbs_first(tmp_path: Path) -> None:
    # At the cap alone (fits as the h1's own single-crumb section); combined
    # with "Child" via the separator it would exceed MAX_SECTION_BYTES,
    # forcing the breadcrumb builder to drop this outermost crumb first.
    long_ancestor = "A" * MAX_SECTION_BYTES
    path = _write(
        tmp_path,
        "doc.html",
        f"""<html><body><main>
        <h1 id="a">{long_ancestor}</h1>
        <h2 id="b">Child</h2>
        <p>Body.</p>
        </main></body></html>""",
    )
    document = HtmlConnector().read(str(path))

    child_section = next(s for s in document.sections if s.section.endswith("Child"))
    assert child_section.section == "Child"  # the long ancestor crumb was dropped, not truncated
    assert len(child_section.section.encode("utf-8")) <= MAX_SECTION_BYTES


def test_unsupported_extension_is_reported_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "doc.txt", "whatever")
    document = HtmlConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]


def test_oversized_file_is_reported_without_parsing(tmp_path: Path) -> None:
    path = tmp_path / "big.html"
    with open(path, "wb") as handle:
        handle.seek(1024)
        handle.write(b"\0")
    document = HtmlConnector(max_file_bytes=1023).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_oversized_extracted_text_is_reported_content_too_large(tmp_path: Path) -> None:
    huge = "x" * (MAX_PASSAGE_BYTES + 1)
    path = _write(tmp_path, "huge.html", f"<html><body><p>{huge}</p></body></html>")
    document = HtmlConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_oversized_source_id_is_reported_without_reading_the_file() -> None:
    long_reference = ("x" * 1025) + ".html"
    document = HtmlConnector().read(long_reference)

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["source_id_too_long"]


def test_missing_file_is_reported_unreadable(tmp_path: Path) -> None:
    document = HtmlConnector().read(str(tmp_path / "missing.html"))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_supports_matches_html_suffixes_and_http_urls_only(tmp_path: Path) -> None:
    connector = HtmlConnector()
    assert connector.supports("a.html") is True
    assert connector.supports("a.HTM") is True
    assert connector.supports("a.xhtml") is True
    assert connector.supports("a.txt") is False
    assert connector.supports("https://example.com/guide") is True
    assert connector.supports("http://example.com/guide") is True
    assert connector.supports("ftp://example.com/guide.html") is False


def test_unsupported_scheme_is_reported_without_touching_the_filesystem() -> None:
    document = HtmlConnector().read("ftp://example.com/doc.html")

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]


def test_parser_identity_is_stamped_into_the_fingerprint(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.html", "<html><body><p>Body.</p></body></html>")
    connector = HtmlConnector()
    document = connector.read(str(path))

    assert document.fingerprint_inputs.parser == connector.parser
    assert document.fingerprint_inputs.parser_version == connector.parser_version


def test_fingerprint_hashes_the_raw_bytes_and_the_effective_options(tmp_path: Path) -> None:
    data = "<html><body><p>Body.</p></body></html>"
    path = _write(tmp_path, "doc.html", data)

    default_document = HtmlConnector().read(str(path))
    assert (
        default_document.fingerprint_inputs.raw_content_sha256
        == hashlib.sha256(data.encode("utf-8")).hexdigest()
    )

    other_document = HtmlConnector(heading_separator=" / ").read(str(path))
    assert (
        other_document.fingerprint_inputs.parse_options_digest
        != default_document.fingerprint_inputs.parse_options_digest
    )
    # Changing an option never touches the raw-bytes hash — the two
    # fingerprint fields answer independent questions (ADR 0007 §5/§6.2).
    assert (
        other_document.fingerprint_inputs.raw_content_sha256
        == default_document.fingerprint_inputs.raw_content_sha256
    )


def test_timeout_and_user_agent_are_excluded_from_the_options_digest(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.html", "<html><body><p>Body.</p></body></html>")

    a = HtmlConnector(timeout=5.0, user_agent="a").read(str(path))
    b = HtmlConnector(timeout=99.0, user_agent="b").read(str(path))

    assert a.fingerprint_inputs.parse_options_digest == b.fingerprint_inputs.parse_options_digest
