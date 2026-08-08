"""PptxConnector — the .pptx connector (ADR 0007 §5/§7/§8/§10, issue #352)."""

from __future__ import annotations

from pathlib import Path

import pytest
from taguru import Locator

pytest.importorskip("pptx")

from taguru_langchain._extract import MAX_PASSAGE_BYTES, split_paragraphs  # noqa: E402
from taguru_langchain.ingest_connectors import PptxConnector  # noqa: E402
from taguru_langchain.ingest_connectors.document import (  # noqa: E402
    MAX_SECTION_BYTES,
    LocatorEntry,
)

from .. import _pptx  # noqa: E402


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_title_and_body_paragraphs_carry_the_same_slide_locator(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(title="Slide Title", bodies=["Body one.", "Body two."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Slide Title\n\nBody one.\n\nBody two."
    assert [(entry.paragraph, entry.section) for entry in document.sections] == [(0, "Slide Title")]
    # Unlike DocxConnector (whose one-locator-per-paragraph budget goes to
    # tables), every body paragraph on a slide — title included — carries
    # the SAME `slide` locator (ADR 0007 §7.2's budget spent on
    # distinguishing body from speaker notes instead).
    assert [entry.locator for entry in document.locators] == [
        Locator(kind="slide", value="1"),
        Locator(kind="slide", value="1"),
        Locator(kind="slide", value="1"),
    ]


def test_table_becomes_one_paragraph_with_a_slide_locator(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(bodies=["Intro"], table=[["A1", "B1"], ["A2", "B2"]])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Intro\n\nA1 | B1\nA2 | B2"
    assert [entry.locator for entry in document.locators] == [
        Locator(kind="slide", value="1"),
        Locator(kind="slide", value="1"),
    ]


def test_speaker_notes_get_their_own_locator_after_the_slide_body(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(bodies=["Body."], notes=["Notes one.", "Notes two."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Body.\n\nNotes one.\n\nNotes two."
    assert [(entry.paragraph, entry.locator) for entry in document.locators] == [
        (0, Locator(kind="slide", value="1")),
        (1, Locator(kind="speaker_notes", value="1")),
        (2, Locator(kind="speaker_notes", value="1")),
    ]


def test_multiple_slides_are_numbered_in_document_order(tmp_path: Path) -> None:
    presentation = _pptx.blank_presentation()
    slide1 = _pptx.add_title_slide(presentation, None)
    _pptx.add_body(slide1, ["Slide one body."])
    slide2 = _pptx.add_title_slide(presentation, None)
    _pptx.add_body(slide2, ["Slide two body."])
    _pptx.add_notes(slide2, ["Slide two notes."])
    path = _write(tmp_path, "deck.pptx", _pptx.save_bytes(presentation))
    document = PptxConnector().read(str(path))

    assert [entry.locator for entry in document.locators] == [
        Locator(kind="slide", value="1"),
        Locator(kind="slide", value="2"),
        Locator(kind="speaker_notes", value="2"),
    ]


def test_group_shape_is_walked_recursively(tmp_path: Path) -> None:
    presentation = _pptx.blank_presentation()
    slide = _pptx.add_title_slide(presentation, None)
    _pptx.add_group(slide, ["Grouped one.", "Grouped two."])
    path = _write(tmp_path, "deck.pptx", _pptx.save_bytes(presentation))
    document = PptxConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Grouped one.\n\nGrouped two."
    assert [entry.locator for entry in document.locators] == [
        Locator(kind="slide", value="1"),
        Locator(kind="slide", value="1"),
    ]


def test_empty_title_placeholder_creates_no_section(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(title="", bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.sections == ()
    assert document.text == "Body."


def test_oversized_title_keeps_the_paragraph_but_creates_no_section(tmp_path: Path) -> None:
    huge_title = "x" * (MAX_SECTION_BYTES + 1)
    raw = _pptx.pptx_bytes(title=huge_title, bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.sections == ()
    assert document.text == f"{huge_title}\n\nBody."


def test_built_text_resplits_into_exactly_the_paragraphs_the_locators_and_sections_index(
    tmp_path: Path,
) -> None:
    """The invariant this connector's correctness rests on, mirroring
    ``test_docx_connector.py``'s/``test_pdf_connector.py``'s own:
    re-running the server's own paragraph splitter over ``text`` must
    yield exactly as many paragraphs as there are, with locator/section
    indices still pointing at the right one."""
    presentation = _pptx.blank_presentation()
    slide1 = _pptx.add_title_slide(presentation, "First Slide")
    _pptx.add_body(slide1, ["Body one.\nsecond line.", "Body two."])
    _pptx.add_table(slide1, [["A1", "B1"], ["A2", "B2"]])
    _pptx.add_notes(slide1, ["Notes."])
    slide2 = _pptx.add_title_slide(presentation, "Second Slide")
    _pptx.add_body(slide2, ["More body."])
    path = _write(tmp_path, "deck.pptx", _pptx.save_bytes(presentation))
    document = PptxConnector().read(str(path))

    resplit = split_paragraphs(document.text)
    assert "\n\n".join(resplit) == document.text
    for entry in document.locators:
        assert 0 <= entry.paragraph < len(resplit)
    for entry in document.sections:
        assert 0 <= entry.paragraph < len(resplit)


def test_real_content_placeholder_round_trips(tmp_path: Path) -> None:
    """Proves this connector reads an ordinary ``Title and Content``
    layout's real placeholders, not only the plain textboxes every other
    fixture in this module uses."""
    presentation = _pptx.blank_presentation()
    _pptx.add_title_and_content_slide(presentation, title="Real Title", body="Real body text.")
    path = _write(tmp_path, "deck.pptx", _pptx.save_bytes(presentation))
    document = PptxConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Real Title\n\nReal body text."
    assert [(entry.paragraph, entry.section) for entry in document.sections] == [(0, "Real Title")]


@pytest.mark.parametrize(
    ("build", "expected_kind"),
    [
        (lambda raw: _pptx.with_smartart(raw), "SmartArt diagram"),
        (lambda raw: _pptx.with_ole_object(raw), "embedded object"),
    ],
)
def test_unreachable_shape_content_is_named_partial_extraction(
    tmp_path: Path, build: object, expected_kind: str
) -> None:
    raw = _pptx.pptx_bytes(bodies=["Body."])
    raw = build(raw)  # type: ignore[operator]
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert [d.code for d in document.diagnostics] == ["partial_extraction"]
    assert expected_kind in document.diagnostics[0].message


def test_chart_content_is_named_partial_extraction(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(bodies=["Body."], chart=True)
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert [d.code for d in document.diagnostics] == ["partial_extraction"]
    assert "chart" in document.diagnostics[0].message


def test_corrupt_pptx_variants_are_reported_as_corrupt(tmp_path: Path) -> None:
    for kind in ("not_zip", "missing_part", "malformed_xml"):
        path = _write(tmp_path, f"broken-{kind}.pptx", _pptx.corrupt_pptx(kind))
        document = PptxConnector().read(str(path))

        assert document.text == ""
        assert [d.code for d in document.diagnostics] == ["corrupt"], kind


def test_encrypted_pptx_is_reported_encrypted_without_being_opened(tmp_path: Path) -> None:
    path = _write(tmp_path, "enc.pptx", _pptx.encrypted_pptx())
    document = PptxConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["encrypted"]


def test_empty_presentation_is_reported_ocr_required(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes()
    path = _write(tmp_path, "empty.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["ocr_required"]


def test_unsupported_extension_is_reported_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "deck.ppt", b"whatever")
    document = PptxConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unsupported_format"]
    # The extension mismatch is affirmative evidence the file isn't a PPTX
    # — content_type stays unclaimed rather than asserting a MIME type this
    # connector has no basis for (the same posture DocxConnector's own
    # `read` already takes).
    assert document.metadata.content_type is None


def test_pptm_extension_is_also_unsupported(tmp_path: Path) -> None:
    path = _write(tmp_path, "deck.pptm", _pptx.pptx_bytes(bodies=["x"]))
    document = PptxConnector().read(str(path))

    assert [d.code for d in document.diagnostics] == ["unsupported_format"]
    assert document.metadata.content_type is None


def test_other_failure_codes_still_claim_the_pptx_mime_type(tmp_path: Path) -> None:
    """Unlike `unsupported_format`, every other failure below happens AFTER
    the `.pptx` extension already matched, so claiming the PPTX MIME type
    is a reasonable inference from a trusted extension, not asserted
    fact — kept unchanged by the `unsupported_format` case above."""
    document = PptxConnector().read(str(tmp_path / "missing.pptx"))

    assert [d.code for d in document.diagnostics] == ["unreadable"]
    assert document.metadata.content_type == (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


def test_oversized_file_is_reported_without_parsing(tmp_path: Path) -> None:
    path = tmp_path / "big.pptx"
    with open(path, "wb") as handle:
        handle.seek(1024)
        handle.write(b"\0")
    document = PptxConnector(max_file_bytes=1023).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_oversized_extracted_text_is_reported_content_too_large(tmp_path: Path) -> None:
    huge_paragraph = "x" * (MAX_PASSAGE_BYTES + 1)
    raw = _pptx.pptx_bytes(bodies=[huge_paragraph])
    path = _write(tmp_path, "huge.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]


def test_zip_bomb_shaped_pptx_is_refused_before_parsing(tmp_path: Path) -> None:
    """`max_file_bytes` bounds only the COMPRESSED size of the zip a .pptx
    is; a small, high-ratio zip must still be refused via the separate
    `max_decompressed_bytes` cap before python-pptx ever decompresses it."""
    path = _write(tmp_path, "bomb.pptx", _pptx.zip_bomb_pptx())
    document = PptxConnector(max_decompressed_bytes=1024).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]
    assert "decompressed" in document.diagnostics[0].message


def test_normal_pptx_still_parses_under_the_decompressed_size_cap(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(title="Slide Title", bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.diagnostics == ()
    assert document.text == "Slide Title\n\nBody."


def test_a_pptx_with_forged_uncompressed_sizes_is_still_refused(tmp_path: Path) -> None:
    """A metadata-only cap (summing `ZipInfo.file_size`) is bypassable: the
    header can declare 50 bytes while the deflate stream expands to
    megabytes. The cap must measure the real, decompressed size."""
    raw = _pptx.forged_size_zip_bomb_pptx(payload_size=8 * 1024 * 1024)
    path = _write(tmp_path, "forged.pptx", raw)
    document = PptxConnector(max_decompressed_bytes=1024 * 1024).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["content_too_large"]
    assert "decompressed" in document.diagnostics[0].message


def test_oversized_source_id_is_reported_without_reading_the_file() -> None:
    long_reference = ("x" * 1025) + ".pptx"
    document = PptxConnector().read(long_reference)

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["source_id_too_long"]


def test_missing_file_is_reported_unreadable(tmp_path: Path) -> None:
    document = PptxConnector().read(str(tmp_path / "missing.pptx"))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["unreadable"]


def test_supports_matches_only_pptx() -> None:
    connector = PptxConnector()
    assert connector.supports("a.pptx") is True
    assert connector.supports("a.PPTX") is True
    assert connector.supports("a.ppt") is False
    assert connector.supports("a.pptm") is False
    assert connector.supports("a.txt") is False


def test_parser_identity_is_stamped_into_the_fingerprint(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)
    connector = PptxConnector()
    document = connector.read(str(path))

    assert document.fingerprint_inputs.parser == connector.parser
    assert document.fingerprint_inputs.parser_version == connector.parser_version


def test_fingerprint_hashes_the_raw_bytes_and_the_effective_options(tmp_path: Path) -> None:
    import hashlib

    data = _pptx.pptx_bytes(bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", data)

    default_document = PptxConnector().read(str(path))
    assert (
        default_document.fingerprint_inputs.raw_content_sha256 == hashlib.sha256(data).hexdigest()
    )

    other_document = PptxConnector(extract_titles=False).read(str(path))
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


def test_extract_titles_false_keeps_paragraph_text_but_drops_sections(
    tmp_path: Path,
) -> None:
    raw = _pptx.pptx_bytes(title="Title.", bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)

    default_document = PptxConnector().read(str(path))
    document = PptxConnector(extract_titles=False).read(str(path))

    assert document.sections == ()
    assert document.text == default_document.text
    assert (
        document.fingerprint_inputs.parse_options_digest
        != default_document.fingerprint_inputs.parse_options_digest
    )


def test_extract_speaker_notes_false_drops_notes_entirely(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(bodies=["Body."], notes=["Notes."])
    path = _write(tmp_path, "deck.pptx", raw)

    document = PptxConnector(extract_speaker_notes=False).read(str(path))

    assert document.text == "Body."
    assert document.locators == (
        LocatorEntry(paragraph=0, locator=Locator(kind="slide", value="1")),
    )


def test_extract_tables_false_drops_the_table_entirely(tmp_path: Path) -> None:
    """Unlike ``DocxConnector`` (whose ``extract_tables=False`` keeps a
    table's text but drops only its locator, since the text still needs
    *some* paragraph to live in), a PPTX table has no locator of its own to
    drop independently — dropping the locator would leave an ordinary body
    paragraph indistinguishable from one, so this connector drops the
    table's paragraph entirely instead (its own module docstring: "never
    folded into a neighboring shape's text")."""
    raw = _pptx.pptx_bytes(bodies=["Intro"], table=[["A1", "B1"]])
    path = _write(tmp_path, "deck.pptx", raw)

    default_document = PptxConnector().read(str(path))
    document = PptxConnector(extract_tables=False).read(str(path))

    assert "A1 | B1" in default_document.text
    assert document.text == "Intro"
    assert (
        document.fingerprint_inputs.parse_options_digest
        != default_document.fingerprint_inputs.parse_options_digest
    )


def test_metadata_title_prefers_core_properties_over_first_slide_title(
    tmp_path: Path,
) -> None:
    presentation = _pptx.blank_presentation()
    _pptx.set_core_title(presentation, "Explicit Title")
    slide = _pptx.add_title_slide(presentation, "Slide Title")
    _pptx.add_body(slide, ["Body."])
    path = _write(tmp_path, "deck.pptx", _pptx.save_bytes(presentation))
    document = PptxConnector().read(str(path))

    assert document.metadata.title == "Explicit Title"


def test_metadata_title_falls_back_to_first_slide_title(tmp_path: Path) -> None:
    raw = _pptx.pptx_bytes(title="Slide Title", bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.metadata.title == "Slide Title"


def test_metadata_content_type_is_the_ooxml_presentation_mime_type(
    tmp_path: Path,
) -> None:
    raw = _pptx.pptx_bytes(bodies=["Body."])
    path = _write(tmp_path, "deck.pptx", raw)
    document = PptxConnector().read(str(path))

    assert document.metadata.content_type == (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


def test_missing_python_pptx_raises_a_clear_error_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import taguru_langchain.ingest_connectors.pptx as pptx_module

    monkeypatch.setattr(pptx_module, "_Presentation", None)
    with pytest.raises(ImportError, match="python-pptx"):
        PptxConnector()
