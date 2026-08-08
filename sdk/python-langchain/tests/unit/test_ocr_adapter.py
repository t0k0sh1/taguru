"""The external OCR adapter boundary (ADR 0007 §10, issue #352) — exercised
through :class:`~taguru_langchain.ingest_connectors.pdf.PdfConnector`, the
one connector that calls out to a configured adapter today. A fake
:class:`~taguru_langchain.ingest_connectors.ocr.OcrAdapter` stands in for a
real OCR engine throughout, per §10's own "no OCR engine ships" posture —
these tests prove the CALL-OUT contract, never a real recognition result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from taguru import Locator

pytest.importorskip("pypdf")

from taguru_langchain.ingest_connectors import PdfConnector  # noqa: E402
from taguru_langchain.ingest_connectors.document import Diagnostic  # noqa: E402
from taguru_langchain.ingest_connectors.ocr import (  # noqa: E402
    OcrRecoveredUnit,
    OcrRequest,
    OcrResult,
)

from .._pdfs import scanned_pdf, text_pdf  # noqa: E402


@dataclass
class FakeOcrAdapter:
    """A minimal, in-memory stand-in for a real OCR engine. ``recovered``
    maps a requested locator's own ``value`` (e.g. a page number, as a
    string) to the text to hand back for it — any requested locator with no
    entry here is simply not recovered, the same "adapter recovered nothing
    for this page" case a real engine can hit. ``fixed_result``, when set,
    bypasses ``recovered`` entirely and is returned as-is regardless of what
    was requested — for exercising `PdfConnector`'s own defense against a
    misbehaving adapter (returning a locator nobody asked for, or text too
    thin to count). ``raise_error``, when set, is raised instead of
    returning anything, standing in for the adapter's own failure."""

    name: str = "fake-ocr"
    version: str = "1.0.0"
    recovered: dict[str, str] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    fixed_result: OcrResult | None = None
    raise_error: Exception | None = None
    received_requests: list[OcrRequest] = field(default_factory=list)

    def recognize(self, request: OcrRequest) -> OcrResult:
        self.received_requests.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        if self.fixed_result is not None:
            return self.fixed_result
        units = tuple(
            OcrRecoveredUnit(locator=locator, text=self.recovered[locator.value])
            for locator in request.locators
            if locator.value in self.recovered
        )
        return OcrResult(units=units, diagnostics=self.diagnostics)


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_adapter_not_invoked_when_no_page_is_unusable(tmp_path: Path) -> None:
    """A configured adapter is called out to only when there is something
    to recover — never on every parse regardless of need."""
    adapter = FakeOcrAdapter(raise_error=AssertionError("should not have been called"))
    path = _write(tmp_path, "text.pdf", text_pdf(["Plenty of real, extractable page text."]))

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.diagnostics == ()
    assert adapter.received_requests == []


def test_absent_adapter_leaves_ocr_required_exactly_as_before(tmp_path: Path) -> None:
    """Regression: `ocr_adapter=None` (the default) must behave identically
    to `PdfConnector` before it could accept one at all — the completion
    criterion issue #352 names verbatim."""
    path = _write(tmp_path, "scan.pdf", scanned_pdf(1))

    document = PdfConnector().read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["ocr_required"]


def test_full_recovery_clears_ocr_required_and_uses_page_locators(tmp_path: Path) -> None:
    adapter = FakeOcrAdapter(
        recovered={
            "1": "Recovered page one text, plenty of characters.",
            "2": "Recovered page two text, plenty of characters.",
        }
    )
    path = _write(tmp_path, "scan.pdf", scanned_pdf(2))

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.diagnostics == ()
    assert document.text == (
        "Recovered page one text, plenty of characters.\n\n"
        "Recovered page two text, plenty of characters."
    )
    assert [(entry.paragraph, entry.locator) for entry in document.locators] == [
        (0, Locator(kind="page", value="1")),
        (1, Locator(kind="page", value="2")),
    ]
    # The adapter was offered exactly the unusable pages, the raw bytes,
    # and this document's own source id/content type — never asked to
    # recover a page this connector already had usable text for.
    assert len(adapter.received_requests) == 1
    request = adapter.received_requests[0]
    assert request.locators == (Locator(kind="page", value="1"), Locator(kind="page", value="2"))
    assert request.content == scanned_pdf(2)
    assert request.content_type == "application/pdf"
    assert request.source == document.source


def test_partial_recovery_leaves_the_remaining_pages_named(tmp_path: Path) -> None:
    adapter = FakeOcrAdapter(recovered={"2": "Recovered middle page, plenty of characters."})
    path = _write(tmp_path, "scan.pdf", scanned_pdf(3))

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.text == "Recovered middle page, plenty of characters."
    assert [entry.locator for entry in document.locators] == [Locator(kind="page", value="2")]
    assert [d.code for d in document.diagnostics] == ["ocr_required"]
    assert "page(s) 1, 3" in document.diagnostics[0].message


def test_extraction_raising_page_is_recovered_when_an_adapter_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page whose ``extract_text()`` raises is exactly as unusable to this
    connector as one that decoded to nothing — a configured adapter must be
    offered a chance to recover it too, not only pages that merely decoded
    empty."""
    from pypdf._page import PageObject

    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(
            [
                "Page one has plenty of readable text.",
                "Page two has plenty of readable text.",
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
    adapter = FakeOcrAdapter(recovered={"2": "Recovered failed page, plenty of characters."})

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.diagnostics == ()
    assert "Recovered failed page, plenty of characters." in document.text
    assert adapter.received_requests[0].locators == (Locator(kind="page", value="2"),)


def test_extraction_raising_page_the_adapter_cannot_recover_stays_partial_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter's own failure to recover a raising page must leave it
    reported exactly as ``partial_extraction`` as if no adapter had been
    configured at all — matching the no-adapter behavior this connector
    already had for a page whose extraction raised."""
    from pypdf._page import PageObject

    path = _write(
        tmp_path,
        "doc.pdf",
        text_pdf(
            [
                "Page one has plenty of readable text.",
                "Page two has plenty of readable text.",
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
    adapter = FakeOcrAdapter()  # recovers nothing for any requested page

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert [d.code for d in document.diagnostics] == ["partial_extraction"]
    assert document.diagnostics[0].message.endswith("page(s) 2")
    assert adapter.received_requests[0].locators == (Locator(kind="page", value="2"),)


def test_adapter_exception_leaves_the_page_ocr_required_and_names_the_failure(
    tmp_path: Path,
) -> None:
    adapter = FakeOcrAdapter(raise_error=RuntimeError("engine unavailable"))
    path = _write(tmp_path, "scan.pdf", scanned_pdf(1))

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["ocr_required"]
    message = document.diagnostics[0].message
    assert "page(s) 1" in message
    assert "OCR adapter failed: engine unavailable" in message


def test_adapter_returning_an_unrequested_or_too_thin_unit_is_discarded(
    tmp_path: Path,
) -> None:
    """A misbehaving (or merely imperfect) adapter must not corrupt the
    document: a locator nobody asked about, and text too thin to clear the
    same `min_chars_per_page` bar every other page's text must clear, are
    both silently dropped rather than spliced in."""
    fixed = OcrResult(
        units=(
            OcrRecoveredUnit(
                locator=Locator(kind="page", value="99"),
                text="Text for a page nobody asked about, plenty long enough.",
            ),
            OcrRecoveredUnit(locator=Locator(kind="page", value="1"), text="x"),
        )
    )
    adapter = FakeOcrAdapter(fixed_result=fixed)
    path = _write(tmp_path, "scan.pdf", scanned_pdf(1))

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.text == ""
    assert [d.code for d in document.diagnostics] == ["ocr_required"]
    assert "OCR adapter failed" not in document.diagnostics[0].message


def test_ocr_result_diagnostics_propagate_to_the_document(tmp_path: Path) -> None:
    path = _write(tmp_path, "scan.pdf", scanned_pdf(1))
    source = str(path)
    adapter = FakeOcrAdapter(
        recovered={"1": "Recovered page text, plenty of characters."},
        diagnostics=(
            Diagnostic(
                code="partial_extraction", message="adapter skipped a region", source=source
            ),
        ),
    )

    document = PdfConnector(ocr_adapter=adapter).read(str(path))

    assert document.text == "Recovered page text, plenty of characters."
    assert [d.code for d in document.diagnostics] == ["partial_extraction"]
    assert document.diagnostics[0].message == "adapter skipped a region"


def test_configuring_an_adapter_changes_the_digest_but_not_the_raw_hash(
    tmp_path: Path,
) -> None:
    data = text_pdf(["Some ordinary page text."])
    path = _write(tmp_path, "doc.pdf", data)

    default_document = PdfConnector().read(str(path))
    adapter_document = PdfConnector(ocr_adapter=FakeOcrAdapter()).read(str(path))

    assert (
        adapter_document.fingerprint_inputs.parse_options_digest
        != default_document.fingerprint_inputs.parse_options_digest
    )
    assert (
        adapter_document.fingerprint_inputs.raw_content_sha256
        == default_document.fingerprint_inputs.raw_content_sha256
        == hashlib.sha256(data).hexdigest()
    )


def test_swapping_adapter_identity_changes_the_digest(tmp_path: Path) -> None:
    """Not just "an adapter is configured," but the adapter's own declared
    identity — swapping engines (or versions) must invalidate a §6.3
    checkpoint's prior skip decision just as surely as configuring one for
    the first time does."""
    data = text_pdf(["Some ordinary page text."])
    path = _write(tmp_path, "doc.pdf", data)

    first = PdfConnector(ocr_adapter=FakeOcrAdapter(name="engine-a")).read(str(path))
    second = PdfConnector(ocr_adapter=FakeOcrAdapter(name="engine-b")).read(str(path))

    assert (
        first.fingerprint_inputs.parse_options_digest
        != second.fingerprint_inputs.parse_options_digest
    )
