"""The connector's own checkpoint (ADR 0007 §6.3, issue #347): "did I
already fetch and parse this object," independent of whether extraction or
import has run — the layer neither ``taguru extract``'s manifest/chunk
checkpoint nor :class:`~taguru_langchain.ingest.TaguruIngester`'s own
``CheckpointStore`` usage covers.

Reuses :class:`~taguru_langchain.checkpoints.CheckpointStore`'s 3-method
contract directly (issue #347's explicit requirement: no bespoke store).
:class:`ConnectorCheckpoint` is the thin layer on top that knows how to
serialize a :class:`~taguru_langchain.ingest_connectors.document.
ConnectorDocument` and gate reuse on its ``fingerprint_inputs`` — the same
composition ``TaguruIngester`` already does for its own checkpoint layer,
one level downstream.
"""

from __future__ import annotations

import json
import warnings

from ..checkpoints import CheckpointStore
from .document import ConnectorDocument, FingerprintInputs


class ConnectorCheckpoint:
    """A connector-native checkpoint over any :class:`CheckpointStore`.

    ``namespace`` prefixes every key (``f"{namespace}:{source}"``) so a
    connector checkpoint and :class:`~taguru_langchain.ingest.TaguruIngester`'s
    own chunk checkpoint can safely share one ``FilesystemCheckpointStore``
    directory — :func:`~taguru_langchain.checkpoints._checkpoint_file_name`
    derives its file name (and hash suffix) from the full key string, so a
    namespaced key can never collide with the bare source id the ingester's
    own checkpoint uses for the same source.

    Degrades exactly like every other checkpoint in this SDK (ADR 0007
    §6.3): a missing, corrupt, or fingerprint-mismatched entry means "treat
    as unseen," never a false hit. ``load``/``save`` failures are reported
    via ``warnings.warn`` and treated as a cache miss / an unsaved run,
    respectively; a ``delete`` failure is silently ignored — the same
    posture :class:`CheckpointStore`'s own contract already documents.
    """

    def __init__(self, store: CheckpointStore, *, namespace: str = "connector") -> None:
        self._store = store
        self._namespace = namespace

    def _key(self, source: str) -> str:
        return f"{self._namespace}:{source}"

    def load(self, source: str, fingerprint: FingerprintInputs) -> ConnectorDocument | None:
        """Returns the previously checkpointed document for ``source`` only
        when its own ``source`` field matches the one requested AND its
        ``fingerprint_inputs`` exactly matches ``fingerprint`` — i.e. the
        raw object is byte-identical to last time under the same
        parser/parser_version/options. The ``source`` check is not
        redundant with the namespaced key: a ``CheckpointStore``
        implementation is only obligated to key by the string it was given
        (``CheckpointStore``'s own contract, ``checkpoints.py``), not to
        guarantee no two keys ever resolve to the same stored bytes — two
        distinct sources whose fetched bytes happen to be byte-identical
        would otherwise pass the fingerprint check alone and swap
        checkpoints under a buggy or lossy custom store. Any mismatch,
        corruption, or store failure returns ``None``."""
        try:
            data = self._store.load(self._key(source))
        except Exception as error:
            warnings.warn(
                f"ConnectorCheckpoint.load({source!r}) raised {error!r}; treating as uncached",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        if data is None:
            return None
        try:
            raw = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        document = ConnectorDocument.from_dict(raw)
        if (
            document is None
            or document.source != source
            or document.fingerprint_inputs != fingerprint
        ):
            return None
        return document

    def save(self, document: ConnectorDocument) -> None:
        """Persists ``document`` in full — not only its fingerprint — so a
        later :meth:`load` hit can skip re-fetching AND re-parsing, the
        whole point of §6.3's connector-native checkpoint layer."""
        payload = json.dumps(document.to_dict(), ensure_ascii=False).encode("utf-8")
        try:
            self._store.save(self._key(document.source), payload)
        except Exception as error:
            warnings.warn(
                f"ConnectorCheckpoint.save({document.source!r}) raised {error!r}; "
                "the object will be re-fetched next run",
                RuntimeWarning,
                stacklevel=2,
            )

    def delete(self, source: str) -> None:
        """Removes any checkpointed document for ``source``. Failures are
        silently ignored — nothing correctness-critical depends on prompt
        cleanup, the same posture ``CheckpointStore.delete`` documents."""
        try:
            self._store.delete(self._key(source))
        except Exception:
            pass
