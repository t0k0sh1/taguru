"""The S3 (object-storage) connector (ADR 0007 §9, issue #351): lists a
bucket/prefix, dispatches each object to whichever installed format
connector (#347-#350) its extension or content-type names, and syncs the
result into a :class:`~taguru_langchain.ingest.TaguruIngester` — with a
two-layer checkpoint (skip the fetch when listing metadata is unchanged;
skip the model call/import when the fetched-and-parsed content is
unchanged), a deletion policy that never retracts by default, and a
``dry_run`` that touches nothing.

Two independent pieces:

- :class:`S3Connector` — dispatch. Given one listed object, fetches its
  raw bytes, hands them (via a suffix-named temporary file — the existing
  format connectors' own contract is "read a local path," never changed
  here) to whichever of the installed connectors ``supports`` it, then
  re-stamps the result's identity (``source``, ``metadata.origin_uri``/
  ``display_name``, each diagnostic's ``source``) onto the S3 source id —
  the delegate's OWN ``fingerprint_inputs`` (``parser``/``parser_version``/
  the raw content hash/its own effective options) is kept untouched,
  exactly as ADR 0007 §5's own JSON example does (``"source":
  "s3://reports/2026/q1.pdf#p12"``, ``"parser": "taguru-pdf-connector"``).
- :func:`sync_object_storage` — orchestration. Enumerates a store/prefix,
  applies both checkpoint layers, calls
  :func:`~taguru_langchain.ingest_connectors.bridge.ingest_connector_document`
  for anything that needs it, detects deletions against a persisted prefix
  inventory, and returns an :class:`S3SyncReport`.

Deliberately NOT a reimplementation of `src/ship.rs`'s Rust
``object_store``-backed path — see
:mod:`taguru_langchain.ingest_connectors.objectstore`'s own module
docstring for why the packaging boundary (ADR 0007 §3/§4) puts this here
instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, cast, get_args

from ..checkpoints import CheckpointStore
from ..ingest import TaguruIngester, _normalize_stop
from .bridge import ingest_connector_document
from .checkpoint import ConnectorCheckpoint
from .document import (
    MAX_TAG_BYTES,
    MAX_TAGS_PER_SOURCE,
    ConnectorDocument,
    ConnectorMetadata,
    Diagnostic,
    DiagnosticCode,
    FingerprintInputs,
    options_digest,
)
from .docx import DocxConnector
from .html import HtmlConnector
from .objectstore import (
    FetchedObject,
    FingerprintTier,
    ObjectMeta,
    ObjectNotFoundError,
    ObjectStore,
    PermanentStoreError,
    TransientStoreError,
    object_fingerprint,
)
from .pdf import PdfConnector
from .protocol import Connector
from .sources import SourceIdRegistry, check_source_id
from .text import TextFileConnector

PARSER_NAME: Final = "taguru-s3-connector"
PARSER_VERSION: Final = "1.0.0"

_DEFAULT_MAX_FILE_BYTES: Final = 64 * 1024 * 1024

_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()

# ADR 0007 §9: "extension first, MIME as fallback" — only consulted when a
# key's own extension does not already resolve to an installed connector.
_CONTENT_TYPE_SUFFIXES: Final[dict[str, str]] = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/xhtml+xml": ".xhtml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/markdown": ".md",
    "text/plain": ".txt",
}


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def _suffix_from_key(key: str) -> str:
    return Path(key).suffix.lower()


def _suffix_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    base = content_type.split(";", 1)[0].strip().lower()
    return _CONTENT_TYPE_SUFFIXES.get(base)


def _connector_for_suffix(connectors: Sequence[Connector], suffix: str) -> Connector | None:
    # `supports()` on every connector this package ships checks only the
    # reference's own extension (HtmlConnector also accepts a URL scheme,
    # irrelevant to a synthetic reference with none) — a fabricated
    # `"object<suffix>"` name is enough to ask "would you read this,"
    # without needing each connector's private suffix set.
    probe = f"object{suffix}"
    for connector in connectors:
        if connector.supports(probe):
            return connector
    return None


def _default_connectors() -> tuple[Connector, ...]:
    """The installed reference connectors, in ADR 0007's own PDF/HTML/DOCX
    order plus the original ``.md``/``.txt`` reference — an optional
    dependency's connector (``pdf``, ``docx``) is simply absent when its
    extra was never installed, so a bucket holding that format legitimately
    reports ``unsupported_format`` rather than raising at construction."""
    connectors: list[Connector] = [TextFileConnector(), HtmlConnector()]
    try:
        connectors.append(PdfConnector())
    except ImportError:
        pass
    try:
        connectors.append(DocxConnector())
    except ImportError:
        pass
    return tuple(connectors)


def _failure_document(
    source: str,
    display_name: str,
    code: DiagnosticCode,
    message: str,
    raw_content_sha256: str = _EMPTY_SHA256,
    *,
    content_type: str | None = None,
) -> ConnectorDocument:
    """A document that never reached a delegate connector at all
    (``source_id_too_long``/``content_too_large``/``unsupported_format``) —
    its ``fingerprint_inputs.parser`` is :class:`S3Connector`'s own
    identity, since no format connector was ever chosen to attribute it
    to."""
    return ConnectorDocument(
        source=source,
        text="",
        metadata=ConnectorMetadata(
            origin_uri=source, display_name=display_name, content_type=content_type
        ),
        fingerprint_inputs=FingerprintInputs(
            raw_content_sha256=raw_content_sha256,
            parser=PARSER_NAME,
            parser_version=PARSER_VERSION,
            parse_options_digest="",
        ),
        diagnostics=(Diagnostic(code=code, message=message, source=source),),
    )


def _reidentify(
    document: ConnectorDocument, *, source: str, display_name: str, content_type: str | None
) -> ConnectorDocument:
    """Replaces only the S3-identity-shaped fields — never
    ``fingerprint_inputs`` (the delegate's own parser/version/raw-hash/
    options stay exactly as it produced them, ADR 0007 §5's own example)."""
    metadata = replace(
        document.metadata,
        origin_uri=source,
        display_name=display_name,
        content_type=document.metadata.content_type or content_type,
    )
    diagnostics = tuple(replace(d, source=source) for d in document.diagnostics)
    return replace(document, source=source, metadata=metadata, diagnostics=diagnostics)


def _mapped_tags(store: ObjectStore, key: str) -> tuple[tuple[str, ...], int]:
    """Object tags → ``"k=v"`` (bare ``"k"`` for an empty value) strings,
    sorted for determinism, capped at ``MAX_TAGS_PER_SOURCE``/
    ``MAX_TAG_BYTES`` — dropped and counted, never truncated (ADR 0007
    §9, the same posture ``questions_dropped``/``sections_dropped``
    already take). Any failure reading tags at all (including the
    ``AccessDenied``-only-on-tagging degrade :meth:`ObjectStore.
    object_tags` itself already applies) yields no tags and no drops —
    never a failed object over a tags-only gap."""
    try:
        raw_tags = store.object_tags(key)
    except (TransientStoreError, PermanentStoreError, ObjectNotFoundError):
        return (), 0
    formatted = sorted(f"{name}={value}" if value else name for name, value in raw_tags.items())
    kept: list[str] = []
    dropped = 0
    for tag in formatted:
        if len(kept) >= MAX_TAGS_PER_SOURCE or len(tag.encode("utf-8")) > MAX_TAG_BYTES:
            dropped += 1
            continue
        kept.append(tag)
    return tuple(kept), dropped


@dataclass(slots=True, frozen=True, kw_only=True)
class ReadResult:
    document: ConnectorDocument
    tags_dropped: int = 0


class S3Connector:
    """Dispatches one object-storage object (S3, or a ``file://``-backed
    test/air-gapped bucket — issue #351's own "no minio/localstack, use
    file://" testing requirement) to whichever installed format connector
    handles it.

    ``connectors`` (default: every connector this package installed —
    :func:`_default_connectors`) is tried in order via each one's own
    ``supports()``; a caller may pass a customized subset/ordering (e.g. a
    ``PdfConnector(extract_outline=False)``). ``max_file_bytes`` (default
    64 MiB) caps the raw object size, checked against the LISTING's own
    size — before any fetch, mirroring every other connector's own
    pre-fetch size cap.
    """

    def __init__(
        self,
        store: ObjectStore,
        *,
        connectors: Sequence[Connector] | None = None,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self._store = store
        self._connectors = tuple(connectors) if connectors is not None else _default_connectors()
        self._max_file_bytes = max_file_bytes

    @property
    def parser(self) -> str:
        return PARSER_NAME

    @property
    def parser_version(self) -> str:
        return PARSER_VERSION

    @property
    def parse_options_digest(self) -> str:
        return options_digest(
            {
                "max_file_bytes": self._max_file_bytes,
                "connectors": [f"{c.parser}:{c.parser_version}" for c in self._connectors],
            }
        )

    def _key_from_reference(self, reference: str) -> str | None:
        prefix = f"{self._store.base_uri}/"
        if not reference.startswith(prefix):
            return None
        return reference[len(prefix) :]

    def supports(self, reference: str) -> bool:
        key = self._key_from_reference(reference)
        if key is None:
            return False
        suffix = _suffix_from_key(key)
        return bool(suffix) and _connector_for_suffix(self._connectors, suffix) is not None

    def read(self, reference: str) -> ConnectorDocument:
        """Protocol-conformant single-string entry point: fetches the
        object's CURRENT (latest) version and dispatches it. The richer,
        version-pinned, tag-aware, drop-counting path
        (:meth:`read_object`/:meth:`read_object_result`) is what
        :func:`sync_object_storage` actually drives; this method exists so
        :class:`S3Connector` is itself usable anywhere a plain
        :class:`~taguru_langchain.ingest_connectors.protocol.Connector`
        is expected."""
        key = self._key_from_reference(reference)
        if key is None:
            raise ValueError(
                f"{reference!r} does not belong to this store ({self._store.base_uri}/...)"
            )
        return self.read_object(ObjectMeta(key=key, size=-1, last_modified=None))

    def read_object(self, meta: ObjectMeta) -> ConnectorDocument:
        """As :meth:`read`, but takes the full listing :class:`ObjectMeta`
        (so a versioned bucket fetches the exact listed version, and the
        listing's own size is checked before any fetch)."""
        return self.read_object_result(meta).document

    def read_object_result(self, meta: ObjectMeta) -> ReadResult:
        source = f"{self._store.base_uri}/{meta.key}"
        display_name = Path(meta.key).name

        source_diagnostic = check_source_id(source)
        if source_diagnostic is not None:
            return ReadResult(
                document=_failure_document(
                    source, display_name, source_diagnostic.code, source_diagnostic.message
                )
            )

        if meta.size >= 0 and meta.size > self._max_file_bytes:
            return ReadResult(
                document=_failure_document(
                    source,
                    display_name,
                    "content_too_large",
                    f"{meta.size} bytes exceeds the {self._max_file_bytes}-byte object cap",
                )
            )

        def oversized(candidate: FetchedObject) -> ReadResult | None:
            # The listing's own size is authoritative when it exists (the
            # pre-fetch check above) — but `read()` (the plain
            # `Connector`-protocol entry point) synthesizes `size=-1` (no
            # listing at all), which skips it entirely, and content-type
            # fallback (below) always fetches before a connector is even
            # chosen. Re-checking the FETCHED bytes here, at every call
            # site right after a fetch, is what makes the cap hold
            # regardless of which path reached it.
            if len(candidate.body) <= self._max_file_bytes:
                return None
            return ReadResult(
                document=_failure_document(
                    source,
                    display_name,
                    "content_too_large",
                    f"{len(candidate.body)} bytes exceeds the "
                    f"{self._max_file_bytes}-byte object cap",
                    content_type=candidate.content_type,
                )
            )

        suffix = _suffix_from_key(meta.key)
        connector = _connector_for_suffix(self._connectors, suffix) if suffix else None
        fetched: FetchedObject | None = None

        if connector is None:
            # Extension alone did not resolve to an installed connector —
            # fetch once and fall back to content-type (ADR 0007 §9:
            # "extension first, MIME as fallback"). A key whose extension
            # names a KNOWN but not-installed format (e.g. `.pdf` without
            # the `pdf` extra) still gets one content-type check here
            # rather than assuming the extension is authoritative.
            fetched = self._store.get(meta.key, version_id=meta.version_id)
            too_large = oversized(fetched)
            if too_large is not None:
                return too_large
            content_suffix = _suffix_from_content_type(fetched.content_type)
            if content_suffix is not None:
                candidate = _connector_for_suffix(self._connectors, content_suffix)
                if candidate is not None:
                    connector = candidate
                    suffix = content_suffix

        if connector is None:
            raw_sha = (
                hashlib.sha256(fetched.body).hexdigest() if fetched is not None else _EMPTY_SHA256
            )
            extension = Path(meta.key).suffix.lower() or "(none)"
            message = f"no connector for extension {extension!r}"
            if fetched is not None:
                message += f" or content-type {fetched.content_type!r}"
            return ReadResult(
                document=_failure_document(
                    source,
                    display_name,
                    "unsupported_format",
                    message,
                    raw_sha,
                    content_type=fetched.content_type if fetched is not None else None,
                )
            )

        if fetched is None:
            fetched = self._store.get(meta.key, version_id=meta.version_id)
            too_large = oversized(fetched)
            if too_large is not None:
                return too_large

        tags, tags_dropped = _mapped_tags(self._store, meta.key)
        document = self._delegate(source, display_name, connector, suffix, fetched, tags)
        return ReadResult(document=document, tags_dropped=tags_dropped)

    def _delegate(
        self,
        source: str,
        display_name: str,
        connector: Connector,
        suffix: str,
        fetched: FetchedObject,
        tags: tuple[str, ...],
    ) -> ConnectorDocument:
        fd, staged_name = tempfile.mkstemp(suffix=suffix)
        staged = Path(staged_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(fetched.body)
            document = connector.read(str(staged))
        finally:
            staged.unlink(missing_ok=True)
        document = _reidentify(
            document, source=source, display_name=display_name, content_type=fetched.content_type
        )
        if tags:
            document = replace(document, metadata=replace(document.metadata, tags=tags))
        return document


Phase = Literal["discovered", "unchanged", "parsed", "extracted", "imported", "skipped", "failed"]
"""ADR 0007 §11's seven-state per-source phase vocabulary, verbatim."""


@dataclass(slots=True, frozen=True, kw_only=True)
class SourceEvent:
    """One phase transition for one source (ADR 0007 §11) — a source's
    full per-run history is the ordered subsequence of :class:`S3SyncReport`
    ``events`` sharing its ``source``."""

    source: str
    phase: Phase
    elapsed_ms: float
    bytes: int = 0
    parser: str | None = None
    diagnostic: Diagnostic | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class S3SyncReport:
    """One run's summary (ADR 0007 §11): counts are a tally of each
    source's LAST phase event only — a source that reached ``imported``
    counts once, under ``imported``, never separately under
    ``discovered``/``parsed`` too. ``deleted_detected``/``retracted`` and
    ``tags_dropped`` are additive to that seven-state vocabulary, specific
    to this connector (ADR 0007 §9)."""

    discovered: int = 0
    unchanged: int = 0
    parsed: int = 0
    extracted: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    deleted_detected: int = 0
    retracted: int = 0
    tags_dropped: int = 0
    events: tuple[SourceEvent, ...] = ()


_S3_OBJECT_CHECKPOINT_VERSION: Final = 1


class S3ObjectCheckpoint:
    """ADR 0007 §9's cheap gate: "does this object's LISTING metadata
    match what we saw last time" — answerable with no ``GET`` at all.
    Composes with (never replaces) :class:`~taguru_langchain.
    ingest_connectors.checkpoint.ConnectorCheckpoint` (§6.3): this layer
    only ever prevents an unnecessary FETCH; ``ConnectorCheckpoint`` still
    separately governs whether a fetched-and-parsed object's model
    call/import can be skipped. Degrades exactly like every other
    checkpoint in this SDK: a missing, corrupt, or store-failure entry
    means "treat as unseen," never a false hit."""

    def __init__(self, store: CheckpointStore, *, namespace: str = "s3-object") -> None:
        self._store = store
        self._namespace = namespace

    def _key(self, source: str) -> str:
        return f"{self._namespace}:{source}"

    def load(self, source: str) -> tuple[FingerprintTier, str] | None:
        try:
            data = self._store.load(self._key(source))
        except Exception:  # noqa: BLE001 - any store failure degrades to "uncached"
            return None
        if data is None:
            return None
        try:
            raw = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if (
            not isinstance(raw, dict)
            or raw.get("version") != _S3_OBJECT_CHECKPOINT_VERSION
            or raw.get("source") != source
        ):
            return None
        tier, value = raw.get("tier"), raw.get("value")
        if tier not in get_args(FingerprintTier) or not isinstance(value, str):
            return None
        return (cast(FingerprintTier, tier), value)

    def save(self, source: str, fingerprint: tuple[FingerprintTier, str]) -> None:
        tier, value = fingerprint
        payload = json.dumps(
            {
                "version": _S3_OBJECT_CHECKPOINT_VERSION,
                "source": source,
                "tier": tier,
                "value": value,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            self._store.save(self._key(source), payload)
        except Exception:  # noqa: BLE001 - best-effort: re-fetched next pass instead
            pass

    def delete(self, source: str) -> None:
        try:
            self._store.delete(self._key(source))
        except Exception:  # noqa: BLE001 - nothing correctness-critical depends on cleanup
            pass


_INVENTORY_NAMESPACE: Final = "s3-inventory"
_INVENTORY_VERSION: Final = 1


def _inventory_key(store: ObjectStore, prefix: str) -> str:
    return f"{_INVENTORY_NAMESPACE}:{store.base_uri}/{prefix}"


def _load_inventory(
    checkpoints: CheckpointStore, store: ObjectStore, prefix: str
) -> frozenset[str] | None:
    """``None`` — "no prior run to compare against" — on anything but a
    clean, version-matching, well-typed prior inventory: a missing or
    corrupt inventory must never masquerade as "nothing was ever here,"
    which would silently manufacture false deletions (ADR 0007 §9)."""
    try:
        data = checkpoints.load(_inventory_key(store, prefix))
    except Exception:  # noqa: BLE001 - a store failure is "no prior inventory," not a crash
        return None
    if data is None:
        return None
    try:
        raw = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("version") != _INVENTORY_VERSION:
        return None
    sources = raw.get("sources")
    if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
        return None
    return frozenset(sources)


def _save_inventory(
    checkpoints: CheckpointStore, store: ObjectStore, prefix: str, sources: frozenset[str]
) -> None:
    payload = json.dumps(
        {"version": _INVENTORY_VERSION, "sources": sorted(sources)}, ensure_ascii=False
    ).encode("utf-8")
    try:
        checkpoints.save(_inventory_key(store, prefix), payload)
    except Exception:  # noqa: BLE001 - best-effort: next pass just over-reports "discovered"
        pass


DeletionPolicy = Literal["report", "retract", "mirror"]

_DELETION_POLICIES: Final = frozenset(get_args(DeletionPolicy))


def _handle_deletions(
    *,
    store: ObjectStore,
    prefix: str,
    checkpoints: CheckpointStore,
    object_checkpoint: S3ObjectCheckpoint,
    connector_checkpoint: ConnectorCheckpoint,
    ingester: TaguruIngester,
    seen_sources: frozenset[str],
    listing_completed: bool,
    deletion_policy: DeletionPolicy,
    dry_run: bool,
) -> tuple[int, int]:
    if dry_run:
        # --dry-run touches nothing at all, including the inventory itself
        # (ADR 0007 §11) — the NEXT real pass still compares against
        # whatever the last real pass left behind.
        return 0, 0

    prior = _load_inventory(checkpoints, store, prefix)
    deleted: frozenset[str] = (prior - seen_sources) if prior is not None else frozenset()

    def retract(source: str) -> None:
        context.retract_source(source)
        # Without this, an object deleted then recreated elsewhere with
        # the SAME (size, last_modified) — the fingerprint tier a
        # non-versioned bucket falls back to — would read as "unchanged"
        # on the very next pass (Layer 1) and never be re-ingested, even
        # though the server no longer has its passage at all.
        object_checkpoint.delete(source)
        connector_checkpoint.delete(source)

    retracted = 0
    needs_context = deleted and deletion_policy in ("retract", "mirror")
    if needs_context or (deletion_policy == "mirror" and listing_completed):
        if ingester.client is None:
            raise ValueError(
                f"deletion_policy={deletion_policy!r} requires ingester.client "
                "(an async-only TaguruIngester has no synchronous Context to retract "
                "through)"
            )
        context = ingester.client.context(ingester.context)
        for source in deleted:
            retract(source)
            retracted += 1
        if deletion_policy == "mirror" and listing_completed:
            # Beyond the inventory diff (which only sees what THIS
            # connector itself previously listed), reconcile against the
            # server's own source list under this prefix — so `mirror`
            # self-heals even on a first run with no prior inventory at
            # all (ADR 0007 §9).
            mirror_prefix = f"{store.base_uri}/{prefix}"
            for source in context.iter_sources(prefix=mirror_prefix):
                if source in seen_sources or source in deleted:
                    continue
                retract(source)
                retracted += 1

    if listing_completed:
        _save_inventory(checkpoints, store, prefix, seen_sources)

    return len(deleted), retracted


def sync_object_storage(
    store: ObjectStore,
    prefix: str,
    *,
    ingester: TaguruIngester,
    checkpoints: CheckpointStore,
    connector: S3Connector | None = None,
    deletion_policy: DeletionPolicy = "report",
    dry_run: bool = False,
    should_stop: Callable[[], bool] | threading.Event | None = None,
) -> S3SyncReport:
    """Enumerates ``store.list(prefix)`` and syncs every object into
    ``ingester``, applying both checkpoint layers, ADR 0007 §9's deletion
    policy, and ``dry_run``.

    ``deletion_policy`` (default ``"report"``, #217's explicit
    default-no-destructive-sync requirement): ``"report"`` never retracts
    — a deleted object only shows up in the returned report's
    ``deleted_detected`` count and its own ``SourceEvent``s carry no
    special marker beyond simply no longer appearing as ``discovered``
    this run. ``"retract"``/``"mirror"`` are both explicit opt-in and
    require ``ingester.client`` (a synchronous core-SDK client — an
    async-only ``TaguruIngester`` has none, and raises instead of
    guessing). ``"mirror"`` additionally reconciles against the server's
    actual source list under this prefix, self-healing even without a
    prior inventory.

    ``dry_run`` tallies every listed object as would-be-``parsed`` or
    ``unchanged`` — never ``discovered``, since :class:`S3SyncReport`'s
    counts reflect each source's LAST phase only; the full ``discovered``
    → ``parsed``/``unchanged`` transition is still visible in ``events``.
    Nothing is fetched, parsed, ingested, or written to the checkpoint/
    inventory store. S3's own listing already carries ADR 0007 §9's
    fingerprint fields, so ``unchanged`` is exactly as reliable here as on
    a real run (ADR 0007 §11).
    """
    if deletion_policy not in _DELETION_POLICIES:
        raise ValueError(f"deletion_policy must be one of {sorted(_DELETION_POLICIES)}")

    stop = _normalize_stop(should_stop)
    active_connector = connector if connector is not None else S3Connector(store)
    object_checkpoint = S3ObjectCheckpoint(checkpoints)
    connector_checkpoint = ConnectorCheckpoint(checkpoints)
    registry = SourceIdRegistry()

    events: list[SourceEvent] = []
    last_phase: dict[str, Phase] = {}
    tags_dropped_total = 0
    seen_sources: set[str] = set()

    def record(
        source: str,
        phase: Phase,
        *,
        elapsed_ms: float,
        object_bytes: int = 0,
        parser: str | None = None,
        diagnostic: Diagnostic | None = None,
    ) -> None:
        events.append(
            SourceEvent(
                source=source,
                phase=phase,
                elapsed_ms=elapsed_ms,
                bytes=object_bytes,
                parser=parser,
                diagnostic=diagnostic,
            )
        )
        last_phase[source] = phase

    listing_completed = True
    try:
        for meta in store.list(prefix):
            if stop():
                listing_completed = False
                break

            source = f"{store.base_uri}/{meta.key}"
            if not registry.claim(source):
                continue  # a duplicate key from the store's own listing
            seen_sources.add(source)

            started = time.monotonic()

            record(source, "discovered", elapsed_ms=0.0)

            fingerprint = object_fingerprint(meta)
            if fingerprint is not None and object_checkpoint.load(source) == fingerprint:
                record(source, "unchanged", elapsed_ms=_elapsed_ms(started))
                continue

            if dry_run:
                record(source, "parsed", elapsed_ms=_elapsed_ms(started))
                continue

            try:
                result = active_connector.read_object_result(meta)
            except ObjectNotFoundError:
                record(
                    source,
                    "skipped",
                    elapsed_ms=_elapsed_ms(started),
                    diagnostic=Diagnostic(
                        code="unreadable", message="object no longer exists", source=source
                    ),
                )
                continue
            except (TransientStoreError, PermanentStoreError) as error:
                record(
                    source,
                    "failed",
                    elapsed_ms=_elapsed_ms(started),
                    diagnostic=Diagnostic(code="unreadable", message=str(error), source=source),
                )
                continue

            tags_dropped_total += result.tags_dropped
            document = result.document

            if not document.text:
                diagnostic = document.diagnostics[0] if document.diagnostics else None
                record(source, "failed", elapsed_ms=_elapsed_ms(started), diagnostic=diagnostic)
                continue

            record(
                source,
                "parsed",
                elapsed_ms=_elapsed_ms(started),
                object_bytes=len(document.text.encode("utf-8")),
                parser=document.fingerprint_inputs.parser,
            )

            # ADR 0007 §6.3: skip the model call/import entirely when this
            # exact fetched-and-parsed content was already ingested —
            # composes with, never replaces, `ingester`'s own per-chunk
            # checkpoint.
            if connector_checkpoint.load(source, document.fingerprint_inputs) is not None:
                if fingerprint is not None:
                    object_checkpoint.save(source, fingerprint)
                record(source, "unchanged", elapsed_ms=_elapsed_ms(started))
                continue

            outcome = ingest_connector_document(ingester, document, should_stop=should_stop)
            if outcome.interrupted:
                record(source, "skipped", elapsed_ms=_elapsed_ms(started))
                listing_completed = False
                break
            if not outcome.ok:
                record(
                    source,
                    "failed",
                    elapsed_ms=_elapsed_ms(started),
                    diagnostic=Diagnostic(
                        code="unreadable",
                        message=outcome.error or "ingest failed",
                        source=source,
                    ),
                )
                continue

            connector_checkpoint.save(document)
            if fingerprint is not None:
                object_checkpoint.save(source, fingerprint)
            record(
                source,
                "imported",
                elapsed_ms=_elapsed_ms(started),
                parser=document.fingerprint_inputs.parser,
            )
    except (TransientStoreError, PermanentStoreError) as error:
        # The LISTING itself failed — nothing was safely enumerated this
        # pass, so the inventory must not be overwritten (ADR 0007 §9: an
        # incomplete pass degrades to "no prior run to compare against"
        # next time, never a false deletion).
        listing_completed = False
        listing_source = f"{store.base_uri}/{prefix}"
        record(
            listing_source,
            "failed",
            elapsed_ms=0.0,
            diagnostic=Diagnostic(code="unreadable", message=str(error), source=listing_source),
        )

    deleted, retracted = _handle_deletions(
        store=store,
        prefix=prefix,
        checkpoints=checkpoints,
        object_checkpoint=object_checkpoint,
        connector_checkpoint=connector_checkpoint,
        ingester=ingester,
        seen_sources=frozenset(seen_sources),
        listing_completed=listing_completed,
        deletion_policy=deletion_policy,
        dry_run=dry_run,
    )

    tallies: dict[str, int] = {}
    for phase in last_phase.values():
        tallies[phase] = tallies.get(phase, 0) + 1

    return S3SyncReport(
        discovered=tallies.get("discovered", 0),
        unchanged=tallies.get("unchanged", 0),
        parsed=tallies.get("parsed", 0),
        extracted=tallies.get("extracted", 0),
        imported=tallies.get("imported", 0),
        skipped=tallies.get("skipped", 0),
        failed=tallies.get("failed", 0),
        deleted_detected=deleted,
        retracted=retracted,
        tags_dropped=tags_dropped_total,
        events=tuple(events),
    )
