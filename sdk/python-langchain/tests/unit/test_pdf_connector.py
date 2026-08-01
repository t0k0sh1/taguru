"""PdfConnector — the .pdf connector (ADR 0007 §7/§8/§10, issue #348)."""

from __future__ import annotations

from pathlib import Path

import pytest
from taguru import Locator

pytest.importorskip("pypdf")

from taguru_langchain._extract import MAX_PASSAGE_BYTES, split_paragraphs  # noqa: E402
from taguru_langchain.ingest_connectors import PdfConnector  # noqa: E402

from .._pdfs import corrupt_pdf, encrypted_pdf, scanned_pdf, text_pdf  # noqa: E402


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_page_boundaries_become_one_page_locator_per_paragraph(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(["First page, one paragraph.", "Second page.\n\nA second paragraph."]),
    )
    document = PdfConnector().read(str(path))

    assert document.diagnostics == ()
    assert [entry.paragraph for entry in document.locators] == [0, 1, 2]
    assert [entry.locator for entry in document.locators] == [
        Locator(kind="page", value="1"),
        Locator(kind="page", value="2"),
        Locator(kind="page", value="2"),
    ]


def test_built_text_resplits_into_exactly_the_paragraphs_the_locators_index(
    tmp_path: Path,
) -> None:
    """The invariant the connector's correctness rests on: re-running the
    server's own paragraph splitter over the ``text`` this connector
    produces must yield exactly as many paragraphs as there are locator
    entries, each at its own index — otherwise a locator silently points
    at the wrong paragraph once the batch reaches ``taguru import``."""
    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(
            [
                "Page one heading.\n\nPage one body.",
                "Page two, single paragraph, two lines.\nsecond line.",
                "Page three.",
            ]
        ),
    )
    document = PdfConnector().read(str(path))

    resplit = split_paragraphs(document.text)
    assert len(resplit) == len(document.locators)
    for index, entry in enumerate(document.locators):
        assert entry.paragraph == index
    # The joined text must round-trip: no paragraph gained or lost a break.
    assert "\n\n".join(resplit) == document.text


def test_outline_entries_become_sections_at_the_matching_paragraph(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(
            [
                "Introduction text for the whole document.",
                "Chapter One body.\n\nMore chapter one content.",
            ],
            outline=[("Introduction", 0), ("Chapter One", 1)],
        ),
    )
    document = PdfConnector().read(str(path))

    assert [(entry.paragraph, entry.section) for entry in document.sections] == [
        (0, "Introduction"),
        (1, "Chapter One"),
    ]
    assert document.metadata.title == "Introduction"


def test_extract_outline_false_disables_sections_and_title(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(["Introduction text for the whole document."], outline=[("Introduction", 0)]),
    )
    document = PdfConnector(extract_outline=False).read(str(path))

    assert document.sections == ()
    assert document.metadata.title is None


def test_outline_entry_on_an_unusable_page_falls_forward_to_the_next_page(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(
            ["", "Real content starts here."],
            outline=[("Front Matter", 0)],
        ),
    )
    document = PdfConnector(min_chars_per_page=1).read(str(path))

    # Page 1 (index 0) is empty and diagnosed ocr_required; the bookmark
    # that names it lands on the first paragraph that actually exists.
    assert [d.code for d in document.diagnostics] == ["ocr_required"]
    assert [(entry.paragraph, entry.section) for entry in document.sections] == [
        (0, "Front Matter")
    ]


def test_scanned_pdf_is_ocr_required_with_empty_text(tmp_path: Path) -> None:
    path = _write(tmp_path, "scan.pdf", scanned_pdf(2))
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert document.locators == ()
    assert document.sections == ()
    assert [d.code for d in document.diagnostics] == ["ocr_required"]
    assert "1" in document.diagnostics[0].message
    assert "2" in document.diagnostics[0].message


def test_partially_scanned_pdf_keeps_the_readable_pages_and_names_the_rest(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "mixed.pdf",
        text_pdf(
            [
                "Readable page one with plenty of extractable text.",
                "",
                "Readable page three with plenty of extractable text.",
            ]
        ),
    )
    document = PdfConnector().read(str(path))

    assert document.text != ""
    assert [entry.locator.value for entry in document.locators] == ["1", "3"]
    assert [d.code for d in document.diagnostics] == ["ocr_required"]
    assert document.diagnostics[0].message.endswith("page(s) 2")


def test_page_extraction_failure_is_reported_as_partial_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf._page import PageObject

    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(
            [
                "Page one has plenty of readable text.",
                "Page two has plenty of readable text.",
                "Page three has plenty of readable text.",
            ]
        ),
    )

    original = PageObject.extract_text
    calls = {"n": 0}

    def flaky(self: PageObject, *args: object, **kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated per-page decode failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    document = PdfConnector().read(str(path))

    assert [entry.locator.value for entry in document.locators] == ["1", "3"]
    assert [d.code for d in document.diagnostics] == ["partial_extraction"]
    assert document.diagnostics[0].message.endswith("page(s) 2")


def test_all_pages_failing_to_extract_is_reported_as_corrupt_not_ocr_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf._page import PageObject

    path = _write(tmp_path, "doc.pdf", text_pdf(["Page one.", "Page two."]))

    def always_fails(self: PageObject, *args: object, **kwargs: object) -> str:
        raise RuntimeError("simulated per-page decode failure")

    monkeypatch.setattr(PageObject, "extract_text", always_fails)
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["corrupt"]


def test_encrypted_pdf_with_a_user_password_is_reported_as_encrypted(tmp_path: Path) -> None:
    path = _write(tmp_path, "enc.pdf", encrypted_pdf(user_password="secret"))
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["encrypted"]


def test_owner_password_only_pdf_still_extracts(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "restricted.pdf",
        encrypted_pdf(
            ["Readable despite owner restrictions."],
            user_password="",
            owner_password="owner-secret",
        ),
    )
    document = PdfConnector().read(str(path))

    assert document.diagnostics == ()
    assert "Readable despite owner restrictions." in document.text


def test_corrupt_pdf_is_reported_as_corrupt(tmp_path: Path) -> None:
    path = _write(tmp_path, "broken.pdf", corrupt_pdf())
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["corrupt"]


def test_empty_pdf_with_zero_pages_is_reported_as_corrupt(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.pdf", text_pdf([]))
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["corrupt"]


def test_unsupported_extension_is_reported_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "doc.txt", b"whatever")
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]


def test_oversized_file_is_reported_without_parsing(tmp_path: Path) -> None:
    path = tmp_path / "big.pdf"
    with open(path, "wb") as handle:
        handle.seek(1024)
        handle.write(b"\0")
    document = PdfConnector(max_file_bytes=1023).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_oversized_extracted_text_is_reported_content_too_large(tmp_path: Path) -> None:
    huge_page = "x" * (MAX_PASSAGE_BYTES + 1)
    path = _write(tmp_path, "huge.pdf", text_pdf([huge_page]))
    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_oversized_source_id_is_reported_without_reading_the_file() -> None:
    long_reference = ("x" * 1025) + ".pdf"
    document = PdfConnector().read(long_reference)

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["source_id_too_long"]


def test_missing_file_is_reported_unreadable(tmp_path: Path) -> None:
    document = PdfConnector().read(str(tmp_path / "missing.pdf"))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_supports_matches_only_pdf(tmp_path: Path) -> None:
    connector = PdfConnector()
    assert connector.supports("a.pdf") is True
    assert connector.supports("a.PDF") is True
    assert connector.supports("a.txt") is False


def test_parser_identity_is_stamped_into_the_fingerprint(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.pdf", text_pdf(["Body."]))
    connector = PdfConnector()
    document = connector.read(str(path))

    assert document.fingerprint_inputs.parser == connector.parser
    assert document.fingerprint_inputs.parser_version == connector.parser_version


def test_fingerprint_hashes_the_raw_bytes_and_the_effective_options(tmp_path: Path) -> None:
    import hashlib

    data = text_pdf(["Body."])
    path = _write(tmp_path, "doc.pdf", data)

    default_document = PdfConnector().read(str(path))
    assert (
        default_document.fingerprint_inputs.raw_content_sha256 == hashlib.sha256(data).hexdigest()
    )

    other_threshold_document = PdfConnector(min_chars_per_page=5).read(str(path))
    assert (
        other_threshold_document.fingerprint_inputs.parse_options_digest
        != default_document.fingerprint_inputs.parse_options_digest
    )
    # Changing an option never touches the raw-bytes hash — the two
    # fingerprint fields answer independent questions (ADR 0007 §5/§6.2).
    assert (
        other_threshold_document.fingerprint_inputs.raw_content_sha256
        == default_document.fingerprint_inputs.raw_content_sha256
    )


def test_missing_pypdf_raises_a_clear_error_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import taguru_langchain.ingest_connectors.pdf as pdf_module

    monkeypatch.setattr(pdf_module, "PdfReader", None)
    with pytest.raises(ImportError, match="pypdf"):
        PdfConnector()
