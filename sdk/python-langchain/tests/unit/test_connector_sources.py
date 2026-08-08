"""Source id derivation and URL canonicalization (ADR 0007 §6.1, issue #347)."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from taguru_langchain.ingest_connectors import (
    SourceIdRegistry,
    canonicalize_url,
    check_source_id,
    file_source_id,
    sub_source_id,
)


def test_file_source_id_is_the_path_verbatim() -> None:
    # Unchanged from taguru extract's own path.to_string_lossy() (src/extract.rs:481).
    assert file_source_id("docs/manual.pdf") == "docs/manual.pdf"


def test_sub_source_id_appends_a_fragment() -> None:
    assert sub_source_id("manual.pdf", "p12") == "manual.pdf#p12"
    assert sub_source_id("manual.pdf", "installation") == "manual.pdf#installation"


def test_check_source_id_is_none_within_the_cap() -> None:
    assert check_source_id("docs/manual.pdf") is None
    assert check_source_id("x" * 1024) is None


def test_check_source_id_flags_an_oversized_source_without_truncating() -> None:
    source = "x" * 1025
    diagnostic = check_source_id(source)
    assert diagnostic is not None
    assert diagnostic.code == "source_id_too_long"
    assert diagnostic.source == source  # never truncated — collision risk


def test_canonicalize_url_strips_userinfo() -> None:
    assert canonicalize_url("https://user:pass@example.com/report.html") == (
        "https://example.com/report.html"
    )


@pytest.mark.parametrize(
    "denied_key",
    [
        "signature",
        "sig",
        "token",
        "access_token",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "apikey",
        "api_key",
        "X-AMZ-SIGNATURE",  # case-insensitive match on the key
        "x-amz-date",
        "x-amz-expires",
        "x-amz-algorithm",
        "x-amz-signedheaders",
        "x-goog-signature",
        "x-goog-credential",
        "x-goog-date",
        "x-goog-expires",
        "x-goog-algorithm",
        "x-goog-signedheaders",
    ],
)
def test_canonicalize_url_strips_denylisted_query_parameters(denied_key: str) -> None:
    url = f"https://example.com/report.html?{denied_key}=secret&kept=1"
    canonical = canonicalize_url(url)
    query = parse_qs(urlsplit(canonical).query)
    assert "secret" not in canonical
    assert denied_key.lower() not in {key.lower() for key in query}
    assert query["kept"] == ["1"]


def test_canonicalize_url_gives_the_same_id_to_the_same_object_presigned_twice() -> None:
    # AWS SigV4 re-issues x-amz-date/x-amz-expires on every presign of the
    # same S3 object; canonicalization must collapse both issuances to one
    # source id or the object gets re-ingested as if it were new each time.
    first = canonicalize_url(
        "https://bucket.s3.amazonaws.com/report.html"
        "?x-amz-credential=AKIA%2F20260101%2Fus-east-1%2Fs3%2Faws4_request"
        "&x-amz-date=20260101T000000Z"
        "&x-amz-expires=3600"
        "&x-amz-signature=aaaa"
    )
    second = canonicalize_url(
        "https://bucket.s3.amazonaws.com/report.html"
        "?x-amz-credential=AKIA%2F20260101%2Fus-east-1%2Fs3%2Faws4_request"
        "&x-amz-date=20260102T000000Z"
        "&x-amz-expires=7200"
        "&x-amz-signature=bbbb"
    )
    assert first == second


def test_canonicalize_url_keeps_a_legitimate_date_parameter_on_a_non_aws_url() -> None:
    # "date" is a plausible, innocent app query parameter; only the
    # AWS/GCS-prefixed forms (x-amz-date, x-goog-date) are signature-shaped
    # and denylisted.
    url = "https://example.com/report.html?date=2026-01-01"
    assert canonicalize_url(url) == url


def test_canonicalize_url_preserves_order_of_kept_query_parameters() -> None:
    url = "https://example.com/report.html?b=2&a=1&token=secret&c=3"
    canonical = canonicalize_url(url)
    assert canonical == "https://example.com/report.html?b=2&a=1&c=3"


def test_canonicalize_url_is_stable_and_safe_for_display() -> None:
    """ADR 0007 §6.1: one canonical value safe for identity, storage, AND
    display — no separate redacted form."""
    url = "https://user:pass@example.com/report.html?token=abc&page=2"
    first = canonicalize_url(url)
    second = canonicalize_url(url)
    assert first == second
    assert "user" not in first and "pass" not in first and "abc" not in first


def test_canonicalize_url_leaves_an_already_clean_url_unchanged() -> None:
    url = "https://example.com/report.html?page=2"
    assert canonicalize_url(url) == url


def test_source_id_registry_claims_each_id_once() -> None:
    registry = SourceIdRegistry()
    assert registry.claim("docs/a.md") is True
    assert registry.claim("docs/b.md") is True
    assert registry.claim("docs/a.md") is False
