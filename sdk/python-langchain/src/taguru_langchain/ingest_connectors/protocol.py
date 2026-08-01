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
