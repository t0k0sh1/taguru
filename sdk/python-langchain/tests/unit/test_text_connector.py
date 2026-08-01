"""TextFileConnector — the reference .md/.txt connector (ADR 0007, issue #347)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from taguru_langchain._extract import MAX_PASSAGE_BYTES, split_paragraphs
from taguru_langchain.ingest_connectors import TextFileConnector


def _write(tmp_path: Path, name: str, content: str, *, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(content.encode(encoding))
    return path


def test_markdown_headings_become_sections_at_split_paragraphs_indices(tmp_path: Path) -> None:
    text = "# Title\n\nIntro paragraph.\n\n## Section Two\n\nMore text.\n"
    path = _write(tmp_path, "doc.md", text)
    document = TextFileConnector().read(str(path))

    assert document.text == text
    paragraphs = split_paragraphs(document.text)
    assert paragraphs[0] == "# Title"
    assert paragraphs[2] == "## Section Two"
    assert {(entry.paragraph, entry.section) for entry in document.sections} == {
        (0, "Title"),
        (2, "Section Two"),
    }
    assert document.metadata.title == "Title"
    assert document.metadata.content_type == "text/markdown"
    assert document.diagnostics == ()


def test_txt_never_carries_sections_even_with_heading_like_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.txt", "# Not a heading\n\nJust text.\n")
    document = TextFileConnector().read(str(path))
    assert document.sections == ()
    assert document.metadata.title is None
    assert document.metadata.content_type == "text/plain"


def test_extract_headings_false_disables_section_extraction_for_markdown(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.md", "# Title\n\nBody.\n")
    document = TextFileConnector(extract_headings=False).read(str(path))
    assert document.sections == ()
    assert document.metadata.title is None


def test_a_heading_sharing_a_paragraph_with_body_text_is_not_recognized(tmp_path: Path) -> None:
    """No blank line separates the heading from the next line, so the two
    fold into one multi-line paragraph — the connector never guesses where
    a heading ends without a blank-line boundary to trust (a documented
    limitation, not a bug)."""
    path = _write(tmp_path, "doc.md", "# Title\nSame paragraph body.\n")
    document = TextFileConnector().read(str(path))
    assert document.sections == ()


def test_closing_hash_sequence_is_stripped_only_when_preceded_by_space(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.md", "## Heading ##\n\n# C#\n")
    document = TextFileConnector().read(str(path))
    labels = [entry.section for entry in document.sections]
    assert "Heading" in labels
    assert "C#" in labels  # trailing '#' with no preceding space is literal text


def test_leading_bom_is_stripped_and_does_not_shift_paragraph_indices(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.md", "﻿# Title\n\nBody.\n")
    document = TextFileConnector().read(str(path))
    assert document.text.startswith("# Title")
    assert document.sections[0].paragraph == 0
    assert document.sections[0].section == "Title"


def test_crlf_line_endings_still_yield_a_single_line_heading_paragraph(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.md", "# Title\r\n\r\nBody line.\r\n")
    document = TextFileConnector().read(str(path))
    assert len(document.sections) == 1
    assert document.sections[0].paragraph == 0
    assert document.sections[0].section == "Title"


def test_raw_content_sha256_is_deterministic_and_independent_of_extension(
    tmp_path: Path,
) -> None:
    content = "# Title\n\nBody.\n"
    md_path = _write(tmp_path, "doc.md", content)
    txt_path = _write(tmp_path, "doc.txt", content)
    md_document = TextFileConnector().read(str(md_path))
    txt_document = TextFileConnector().read(str(txt_path))
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert md_document.fingerprint_inputs.raw_content_sha256 == expected
    assert txt_document.fingerprint_inputs.raw_content_sha256 == expected


def test_missing_file_is_reported_unreadable(tmp_path: Path) -> None:
    document = TextFileConnector().read(str(tmp_path / "missing.md"))
    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_unsupported_extension_is_reported_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "doc.pdf", "whatever")
    document = TextFileConnector().read(str(path))
    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]


def test_oversized_file_is_reported_content_too_large(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    with open(path, "wb") as handle:
        handle.seek(MAX_PASSAGE_BYTES)
        handle.write(b"\0")
    document = TextFileConnector().read(str(path))
    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_non_utf8_bytes_are_reported_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe not utf-8")
    document = TextFileConnector().read(str(path))
    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["corrupt"]


def test_oversized_source_id_is_reported_without_reading_the_file() -> None:
    # The length check runs before any filesystem access, so a fabricated
    # path that is never actually created still exercises it cleanly.
    long_reference = ("x" * 1025) + ".txt"
    document = TextFileConnector().read(long_reference)
    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["source_id_too_long"]


def test_supports_matches_only_md_and_txt(tmp_path: Path) -> None:
    connector = TextFileConnector()
    assert connector.supports("a.md") is True
    assert connector.supports("a.TXT") is True
    assert connector.supports("a.pdf") is False


def test_parser_identity_is_stamped_into_the_fingerprint(tmp_path: Path) -> None:
    path = _write(tmp_path, "doc.txt", "body")
    connector = TextFileConnector()
    document = connector.read(str(path))
    assert document.fingerprint_inputs.parser == connector.parser
    assert document.fingerprint_inputs.parser_version == connector.parser_version
