"""The connector extension point (ADR 0007 §5, issue #347): the interface
any format connector — present or future — implements to produce
:class:`~taguru_langchain.ingest_connectors.document.ConnectorDocument`.

A ``Protocol``, not an ABC — the same extension-point convention
:class:`~taguru_langchain.checkpoints.CheckpointStore` already uses:
structural typing, no forced base class, no ``runtime_checkable`` (nothing
here needs an ``isinstance`` check).
"""

from __future__ import annotations

from typing import Protocol

from .document import ConnectorDocument


class Connector(Protocol):
    """Reads one reference (a path, URL, or object key) into a
    :class:`ConnectorDocument`."""

    @property
    def parser(self) -> str:
        """The connector's own identity, stamped into every document's
        ``fingerprint_inputs.parser`` (ADR 0007 §5)."""
        ...

    @property
    def parser_version(self) -> str:
        """The connector's own version, stamped into
        ``fingerprint_inputs.parser_version``."""
        ...

    @property
    def parse_options_digest(self) -> str:
        """The digest :meth:`read` would stamp into
        ``fingerprint_inputs.parse_options_digest`` for this instance's
        current effective config — computable without reading anything.
        Added for issue #351's S3 connector: its own connector-level
        checkpoint (ADR 0007 §6.3) must decide whether re-parsing a
        just-fetched object can be skipped BEFORE calling :meth:`read`,
        which needs this value as part of the candidate fingerprint it
        checks the checkpoint against."""
        ...

    def supports(self, reference: str) -> bool:
        """Whether this connector can read ``reference`` at all (extension,
        MIME, or content sniffing) — a caller dispatching across several
        connectors checks this before calling :meth:`read`."""
        ...

    def read(self, reference: str) -> ConnectorDocument:
        """Reads ``reference`` into a normalized document.

        Never raises for an ordinary parse failure — an unreadable,
        encrypted, corrupt, or unsupported object is reported as a
        :class:`ConnectorDocument` with an empty ``text`` and a non-empty
        ``diagnostics`` (ADR 0007 §5), not an exception. Raising is
        reserved for a programming error (e.g. calling ``read`` on a
        reference ``supports`` already rejected).
        """
        ...
