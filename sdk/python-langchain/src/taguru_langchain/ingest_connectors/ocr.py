"""The external OCR adapter boundary (ADR 0007 §10, issue #352).

No OCR engine ships in this package, or in any connector — the same
posture ``pdf.py``'s own module docstring already states. A page (or, for a
future connector, any other locator-shaped unit) whose text layer is empty
or near-empty is diagnosed ``ocr_required`` with empty ``text`` in its
``ConnectorDocument`` (§5, §8, §10); an adapter is how an operator who has
their own OCR engine plugs it in, without this package ever depending on
one.

The interface is deliberately narrow, per §10's own wording: "given the raw
document bytes (or a rendered page image), return recovered text plus the
same locator shape (§7) any other connector produces — i.e., an OCR adapter
is itself just another producer of §5's normalized-document contract for
the pages it recovers." :class:`OcrAdapter` is a ``Protocol`` — the same
structural-typing convention :class:`~.protocol.Connector` and
:class:`~taguru_langchain.checkpoints.CheckpointStore` already use: no
forced base class, no ``runtime_checkable``, nothing here needs an
``isinstance`` check.

Only :class:`~.pdf.PdfConnector` calls out to a configured adapter today
(§10: "#352 implements this interface and the PDF connector's call-out to
it when configured"). The interface itself is not PDF-specific — it is
keyed on :class:`~taguru.Locator`, the same open ``kind``/``value`` shape
every connector's own ``locators`` already use — so a future connector
(PPTX, DOCX, HTML) can adopt the same adapter without a shape change here.
Wiring a second connector to it is left for a future issue; this one only
proves the interface against the PDF connector, which already has the
per-page granularity (§10's "the pages it recovers") an adapter call
naturally wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from taguru import Locator

from .document import Diagnostic


@dataclass(slots=True, frozen=True, kw_only=True)
class OcrRequest:
    """One call to an :class:`OcrAdapter`: the raw bytes a connector could
    not extract a usable text layer from, plus which locators need
    recovering.

    ``content`` is the same raw, pre-parse bytes a connector hashes into
    its own ``fingerprint_inputs.raw_content_sha256`` (§5) — never a
    re-rendered or re-encoded copy, so an adapter that itself rasterizes
    pages (rather than reading an embedded image) can do so from the exact
    bytes the connector fetched. ``locators`` names each unit recovery is
    wanted for (e.g. one :class:`~taguru.Locator` per unusable page); an
    empty tuple asks for the whole document, for a future connector kind
    with no finer-grained unit to name.
    """

    source: str
    content: bytes
    content_type: str | None
    locators: tuple[Locator, ...] = ()


@dataclass(slots=True, frozen=True, kw_only=True)
class OcrRecoveredUnit:
    """Recovered text for exactly one requested locator — the adapter's own
    twin of a connector's ``{"paragraph": N, "locator": {...}}`` line,
    minus the paragraph index a connector assigns only after splicing this
    text back into its own document (§7.4: a locator is derived from the
    source document's own structure, never from anything chunk-shaped)."""

    locator: Locator
    text: str


@dataclass(slots=True, frozen=True, kw_only=True)
class OcrResult:
    """An adapter's answer to one :class:`OcrRequest`.

    ``units`` covers whatever the adapter could recover — a subset of the
    requested locators is expected and normal (some pages may still have no
    usable text even after OCR); the calling connector decides what to do
    with a locator it asked for but did not get back. ``diagnostics`` lets
    an adapter report its own failures (§8's shared vocabulary, e.g.
    ``partial_extraction`` for a page it could not process) without raising
    — the same never-raise-for-an-ordinary-failure posture
    :meth:`~.protocol.Connector.read` itself takes.
    """

    units: tuple[OcrRecoveredUnit, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()


class OcrAdapter(Protocol):
    """The external OCR engine boundary a connector calls out to when one
    is configured (§10). Structural typing, no forced base class — an
    adapter is anything with this shape, including a fake built purely for
    tests."""

    @property
    def name(self) -> str:
        """The adapter's own identity, folded into the calling connector's
        ``parse_options_digest`` (alongside its own options) so that
        configuring, swapping, or removing an adapter changes the
        connector's effective fingerprint — the same reason
        :attr:`~.protocol.Connector.parser` exists, applied to an adapter
        instead of a connector."""
        ...

    @property
    def version(self) -> str:
        """The adapter's own version, folded into the calling connector's
        ``parse_options_digest`` the same way :attr:`name` is."""
        ...

    def recognize(self, request: OcrRequest) -> OcrResult:
        """Recovers whatever text ``request`` names.

        Never expected to raise for an ordinary recognition failure — an
        adapter that cannot process a page reports it via
        ``OcrResult.diagnostics`` instead of an exception, the same
        posture :meth:`~.protocol.Connector.read` documents for itself.
        A calling connector treats any exception raised here as this
        adapter's own failure, not a reason to fail the whole document: the
        pages it named stay ``ocr_required``, exactly as if no adapter had
        been configured at all.
        """
        ...
