"""HtmlConnector — the URL-fetch half of the `.html` connector (ADR 0007
§6.1/§7/§8, issue #349), exercised against a real (loopback) HTTP server via
`tests/_httpd.py`. Local-file behavior is `test_html_connector.py`'s own
file.

Every fetch here targets `127.0.0.1` — a private/loopback address the
connector refuses by default (its own SSRF mitigation, documented in
`html.py`'s module docstring) — so every `HtmlConnector` below except
`test_a_private_destination_is_refused_by_default` itself passes
`allow_private_networks=True` to reach the test server at all.
"""

from __future__ import annotations

from taguru_langchain.ingest_connectors import HtmlConnector

from .._httpd import Route, serve

_PAGE = b"""<html><head><title>Page</title></head>
<body><main><h1 id="h">Heading</h1><p>Body text.</p></main></body></html>"""


def test_a_200_html_response_is_read_normally() -> None:
    with serve({"/page": Route(body=_PAGE)}) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/page")

    assert document.diagnostics == ()
    assert document.text == "Heading\n\nBody text."
    assert document.source == f"{server.base_url}/page"
    assert document.metadata.title == "Page"


def test_a_private_destination_is_refused_by_default() -> None:
    with serve({"/page": Route(body=_PAGE)}) as server:
        document = HtmlConnector().read(f"{server.base_url}/page")

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_404_is_reported_unreadable() -> None:
    with serve({"/page": Route(status=404, body=b"not found")}) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/page")

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]
    assert "404" in document.diagnostics[0].message


def test_server_error_is_reported_unreadable() -> None:
    with serve({"/page": Route(status=500, body=b"boom")}) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/page")

    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_non_html_content_type_is_reported_unsupported_format() -> None:
    with serve({"/page": Route(content_type="application/json", body=b"{}")}) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/page")

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]


def test_missing_content_type_header_is_still_parsed() -> None:
    with serve({"/page": Route(content_type=None, body=_PAGE)}) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/page")

    assert document.diagnostics == ()
    assert document.text == "Heading\n\nBody text."


def test_xhtml_content_type_is_accepted() -> None:
    with serve({"/page": Route(content_type="application/xhtml+xml", body=_PAGE)}) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/page")

    assert document.diagnostics == ()


def test_connection_refused_is_reported_unreadable() -> None:
    # An ephemeral loopback port nothing is listening on.
    document = HtmlConnector(timeout=2.0, allow_private_networks=True).read(
        "http://127.0.0.1:1/page"
    )

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_a_slow_response_past_the_timeout_is_reported_unreadable() -> None:
    with serve({"/page": Route(body=_PAGE, delay=0.5)}) as server:
        document = HtmlConnector(timeout=0.05, allow_private_networks=True).read(
            f"{server.base_url}/page"
        )

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_a_response_trickling_past_the_total_time_budget_is_reported_unreadable() -> None:
    # Each inter-chunk gap (0.02s) lands well inside `timeout` (5s) — no
    # phase ever goes idle for long enough to trip it — but the whole
    # trickled transfer still overruns `max_total_seconds`, the failure
    # mode a per-phase timeout alone can never catch.
    body = _PAGE + b"<!-- padding -->" * 20
    with serve({"/page": Route(body=body, chunk_delay=0.02)}) as server:
        document = HtmlConnector(
            timeout=5.0, max_total_seconds=0.1, allow_private_networks=True
        ).read(f"{server.base_url}/page")

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_the_source_id_is_the_final_url_after_a_redirect() -> None:
    with serve(
        {
            "/start": Route(location="/final"),
            "/final": Route(body=_PAGE),
        }
    ) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/start")

    assert document.source == f"{server.base_url}/final"
    assert document.diagnostics == ()


def test_a_redirected_failure_names_the_final_url_not_the_start() -> None:
    """The module contract — "the source id is the final URL" — holds on
    failure paths too: two start URLs redirecting to one failing endpoint
    must fold to one source, and the diagnostic must name the URL that
    actually answered."""
    with serve(
        {
            "/start": Route(location="/final-404"),
            "/final-404": Route(status=404, body=b"gone"),
        }
    ) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/start")

    assert [d.code for d in document.diagnostics] == ["unreadable"]
    assert document.source == f"{server.base_url}/final-404"
    assert document.metadata.display_name == "final-404"


def test_a_redirected_wrong_content_type_names_the_final_url() -> None:
    with serve(
        {
            "/start": Route(location="/final-json"),
            "/final-json": Route(content_type="application/json", body=b"{}"),
        }
    ) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/start")

    assert [d.code for d in document.diagnostics] == ["unsupported_format"]
    assert document.source == f"{server.base_url}/final-json"


def test_userinfo_and_signed_query_params_are_stripped_from_the_source_id() -> None:
    with serve({"/page": Route(body=_PAGE)}) as server:
        host = server.base_url.removeprefix("http://")
        reference = f"http://user:pass@{host}/page?token=SECRET&keep=1"
        document = HtmlConnector(allow_private_networks=True).read(reference)

    # Also confirms the request actually reached the /page route (i.e. the
    # query string didn't defeat the test server's own route lookup) rather
    # than the pre-fetch canonicalized id merely happening to match on a
    # failed request.
    assert document.diagnostics == ()
    assert document.source == f"{server.base_url}/page?keep=1"
    assert "pass" not in document.source
    assert "SECRET" not in document.source


def test_the_pages_own_fragment_is_stripped_from_the_source_id() -> None:
    with serve({"/page": Route(body=_PAGE)}) as server:
        document = HtmlConnector(allow_private_networks=True).read(
            f"{server.base_url}/page#section-two"
        )

    assert document.source == f"{server.base_url}/page"


def test_a_response_over_the_byte_cap_is_refused_mid_stream() -> None:
    body = _PAGE + b"<!-- padding -->" * 10_000
    with serve({"/page": Route(body=body)}) as server:
        document = HtmlConnector(max_file_bytes=len(_PAGE), allow_private_networks=True).read(
            f"{server.base_url}/page"
        )

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_canonical_link_is_resolved_relative_to_the_final_fetched_url() -> None:
    body = b"""<html><head><link rel="canonical" href="/canon"></head>
    <body><main><p>Text.</p></main></body></html>"""
    with serve(
        {
            "/start": Route(location="/final"),
            "/final": Route(body=body),
        }
    ) as server:
        document = HtmlConnector(allow_private_networks=True).read(f"{server.base_url}/start")

    assert document.metadata.canonical_url == f"{server.base_url}/canon"


def test_supports_true_but_unsupported_scheme_reads_never_touch_the_network() -> None:
    document = HtmlConnector().read("ftp://127.0.0.1:1/doc.html")

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]


def test_oversized_url_source_id_is_refused_before_any_fetch() -> None:
    reference = "http://127.0.0.1:1/" + ("x" * 1200)
    document = HtmlConnector().read(reference)

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["source_id_too_long"]
