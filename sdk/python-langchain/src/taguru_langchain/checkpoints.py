"""Durable per-chunk checkpoint stores for :class:`~taguru_langchain.ingest.TaguruIngester`
(issue #211, the Python twin of ``taguru extract``'s issue #210).

The SDK owns every correctness-critical piece of checkpoint/resume — the
fingerprint gate that invalidates a whole document's cache on any
output-shaping settings change, the unit-content-hash keying that survives
a differently-chunked resume, and the JSON schema on disk. A
:class:`CheckpointStore` is only durable blob storage keyed by source id;
implementing one for object storage or a database is three small methods,
and none of them can cause a false reuse even in principle — the ingester
parses, structurally compares the fingerprint, and degrades silently to
"nothing cached" on anything it does not recognize.

:class:`FilesystemCheckpointStore` is the batteries-included default: one
JSON file per document, the SDK analogue of ``taguru extract``'s
``.extract-checkpoints/`` directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from ._extract import ModelOutput

try:
    import fcntl
except ImportError:  # Windows has no flock — advisory locking is skipped
    # there (see FilesystemCheckpointStore._acquire_lock); `fcntl` stays
    # unbound on that platform and every reference to it is guarded by an
    # `is None` check, never reached at runtime on a platform where the
    # import itself failed.
    fcntl = None  # type: ignore[assignment]


class CheckpointStore(Protocol):
    """Durable blob storage for one document's checkpoint state, keyed by
    source id.

    Contract the ingester depends on:

    - ``save`` MUST be atomic: a concurrent or post-crash ``load`` returns
      either a previously saved value in full or the new one in full,
      never a partial write. :class:`FilesystemCheckpointStore` gets this
      via a temp-file-then-rename.
    - ``load`` returns ``None`` when nothing was ever saved for ``source``
      — the ordinary first-run/never-checkpointed case, not a failure.
    - A ``load``/``save``/``delete`` call MAY raise. ``load``/``save``
      failures are caught by the ingester and reported via
      ``warnings.warn``; the run proceeds as though nothing were cached
      (``load``) or the completed chunk still counts for this run
      (``save``). A ``delete`` failure is silently ignored — nothing
      correctness-critical depends on prompt cleanup.

    A store MAY additionally implement ``append(source: str, record:
    bytes) -> None`` (see :class:`_AppendCapableCheckpointStore`): appends
    one already-serialized record to whatever ``save`` last wrote, without
    rewriting the rest. This is a pure performance opt-in — a store that
    omits it keeps working exactly as before, just at ``save``'s O(document
    size) cost on every chunk instead of O(1). A store may also raise
    :class:`CheckpointLockedError` from ``save``/``append`` to report that
    another run already holds ``source``; :class:`FilesystemCheckpointStore`
    does this via an advisory per-source lock.
    """

    def load(self, source: str) -> bytes | None:
        """Return the last value saved for ``source``, or ``None`` if none
        was ever saved."""
        ...

    def save(self, source: str, data: bytes) -> None:
        """Atomically persist ``data`` as the sole value for ``source``,
        replacing whatever was saved before."""
        ...

    def delete(self, source: str) -> None:
        """Remove any saved value for ``source``. Deleting an already-empty
        source is not an error."""
        ...


@runtime_checkable
class _AppendCapableCheckpointStore(Protocol):
    """The optional extension :class:`CheckpointStore` implementations may
    add — checked via ``isinstance`` (``@runtime_checkable`` makes that a
    structural, attribute-presence check) rather than required on every
    store, so a custom store that only ever implements the base 3 methods
    keeps working unchanged."""

    def append(self, source: str, record: bytes) -> None:
        """Append one already-serialized record — produced by
        :meth:`_DocumentCheckpoints.unit_bytes` — to whatever ``save`` last
        wrote for ``source``. Never called before at least one ``save`` has
        established the file this run."""
        ...


class CheckpointLockedError(RuntimeError):
    """Raised by a store's ``save``/``append`` when another run already
    holds ``source``'s advisory lock (see
    :class:`FilesystemCheckpointStore`). Deliberately a distinct type from
    a generic store failure: the ingester reacts to lock CONTENTION by
    disabling further checkpoint writes for this document for the rest of
    the run (and skipping its end-of-run delete, so it never touches state
    that belongs to whoever holds the lock) instead of re-warning on the
    same conflict for every remaining chunk."""

    def __init__(self, source: str) -> None:
        super().__init__(f"checkpoint for {source!r} is locked by another run")
        self.source = source


def _flatten_source(source: str) -> str:
    return source.replace("/", "__").replace("\\", "__").replace(":", "__")


def _truncate_utf8_prefix(text: str, max_bytes: int) -> str:
    """The largest prefix of ``text`` whose UTF-8 encoding is at most
    ``max_bytes`` long, backing off to a valid character boundary rather
    than splitting a multi-byte codepoint — mirrors src/extract.rs's
    ``floor_char_boundary`` truncation exactly."""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _checkpoint_file_name(source: str) -> str:
    """Same flatten-then-hash-suffix scheme as ``taguru extract``'s
    ``checkpoint_file_name`` (src/extract.rs): path separators and ``:``
    flatten to ``__`` so the checkpoint directory stays flat and the name
    stays human-readable, then a 16-hex-character content-hash suffix is
    ALWAYS appended — flattening alone is not injective (``"a/b"``,
    ``"a:b"``, and ``"a__b"`` all flatten to ``"a__b"``), so without the
    suffix, distinct short source ids could collide on the same file and
    silently share (and overwrite) each other's checkpoint progress. A
    flattened name over 120 UTF-8 bytes also truncates to a ≤96-byte prefix
    so long source paths never blow a filesystem's name-length limit; the
    hash suffix alone is what keeps such names apart."""
    name = _flatten_source(source)
    prefix = _truncate_utf8_prefix(name, 96) if len(name.encode("utf-8")) > 120 else name
    suffix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{suffix}.json"


def _fsync_dir(directory: Path) -> None:
    """Best-effort parent-directory fsync: the rename in ``_write_atomic``
    is itself an entry in the parent directory's own data, so without this
    a crash could forget the rename even though the file's own contents
    reached disk. Some platforms (Windows) cannot open a directory for
    fsync at all — silently skipped there; the file's own fsync is still
    the durability floor."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    """Writes via a temporary file, fsync, and rename — a crash mid-write
    leaves the previous version intact, and power loss right after this
    call returns cannot tear or lose the new one. Mirrors src/storage.rs's
    ``write_atomic`` exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        # The staged file never became `path` — clean up its temporary
        # name rather than leave partial-write litter behind. A failure
        # after a successful `os.replace` never reaches here (`staged` no
        # longer exists under its temporary name at that point).
        staged.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


class FilesystemCheckpointStore:
    """One JSON file per document under ``directory`` — the SDK analogue of
    ``taguru extract``'s ``.extract-checkpoints/``. ``directory`` is never
    created until the first ``save``; a source with nothing checkpointed
    yet never gets a file.

    Guards against two concurrent runs on the SAME source silently
    last-writer-winning each other's checkpoint (issue #<concurrent-run
    clobber>): the first ``save``/``append`` for a source acquires an
    advisory, exclusive, non-blocking ``flock`` on a sidecar
    ``<checkpoint-file>.lock`` file, held for this instance's lifetime (or
    until :meth:`delete`/:meth:`close`). A second run's write attempt on
    the same source raises :class:`CheckpointLockedError` instead of
    racing the first run's writes. Process death releases the OS-level
    lock automatically, so a crashed run never wedges the next one; a
    platform with no ``fcntl`` (Windows) silently skips locking rather than
    refusing to run there.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        self._locks: dict[str, IO[bytes]] = {}

    def path_for(self, source: str) -> Path:
        """The file ``source`` maps to — public so an operator can inspect
        or remove one document's checkpoints by hand (the SDK has no
        ``--force`` equivalent; deleting this file is the manual
        invalidation path)."""
        return self._directory / _checkpoint_file_name(source)

    def _lock_path(self, source: str) -> Path:
        path = self.path_for(source)
        return path.with_name(path.name + ".lock")

    def _acquire_lock(self, source: str) -> None:
        """Idempotent: a source already held by THIS instance (the common
        case — every write after a document's first goes through here
        again) is a no-op, never a re-acquisition attempt against our own
        lock."""
        if fcntl is None or source in self._locks:
            return
        lock_path = self._lock_path(source)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise CheckpointLockedError(source) from error
        self._locks[source] = handle

    def _release_lock(self, source: str) -> None:
        handle = self._locks.pop(source, None)
        if handle is not None:
            # Closing the fd releases the OS-level flock by itself; no
            # explicit LOCK_UN needed. The sidecar file itself is left in
            # place (cheap, and its mere existence carries no state) —
            # only the in-process handle holding the lock open goes away.
            handle.close()

    def load(self, source: str) -> bytes | None:
        try:
            return self.path_for(source).read_bytes()
        except FileNotFoundError:
            return None

    def save(self, source: str, data: bytes) -> None:
        self._acquire_lock(source)
        _write_atomic(self.path_for(source), data)

    def append(self, source: str, record: bytes) -> None:
        """Appends one already-serialized JSONL line — the O(1)-per-chunk
        counterpart to ``save``'s O(document-size) full rewrite. Not
        wrapped in ``_write_atomic``'s temp-file-then-rename: an append
        can't be made atomic that way without paying the full-rewrite cost
        it exists to avoid. A crash mid-append can leave a torn final
        line, which :meth:`_DocumentCheckpoints.from_bytes` already
        tolerates by discarding it on the next load."""
        self._acquire_lock(source)
        path = self.path_for(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as handle:
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())

    def delete(self, source: str) -> None:
        self.path_for(source).unlink(missing_ok=True)
        self._release_lock(source)

    def close(self) -> None:
        """Release every advisory lock this instance still holds, without
        touching the checkpoint files themselves — the explicit
        "this run is done with these sources" signal a real process exit
        gives for free by closing all its file descriptors."""
        for source in list(self._locks):
            self._release_lock(source)


def _derive_model_identity(llm: BaseChatModel) -> str | None:
    """Best-effort model identity for the checkpoint fingerprint, prefixed
    with the wrapper's class name so two providers exposing the same model
    string (e.g. two different ``"llama3"`` backends) can never collide.
    LangChain's own ``BaseChatModel._get_ls_params`` resolves a model name
    the same best-effort way but documents that derivation as not part of
    a stable contract — this is the SDK's own equivalent, checked once at
    :class:`~taguru_langchain.ingest.TaguruIngester` construction time so
    an ingester with an unidentifiable model fails fast (``checkpoint_model_id``
    is then required) rather than checkpointing under an unstable or
    degenerate key."""
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return f"{type(llm).__name__}:{value}"
    return None


@dataclass(frozen=True)
class _CheckpointFingerprint:
    """The same compatibility inputs a batch's manifest entry would check,
    minus anything that shapes only how many corrective turns a chunk
    takes rather than the validity of an accepted output. Any field
    mismatch (content edited, model/prompt/questions/fact_budget/
    structured_output/lossy changed) treats the whole checkpoint file as
    absent — a settings change can never silently reuse an incompatible
    cached chunk. Mirrors src/extract.rs's ``CheckpointFingerprint``
    (minus ``max_output_tokens``, which has no Python equivalent, and
    ``max_attempts``/``chunk_bytes``, which Rust excludes for the same
    reason: a validated output does not depend on retry budget, and a
    chunk-size change just makes today's pieces hash differently — a safe
    cache miss, not a correctness hazard)."""

    sha256: str
    model: str
    prompt_version: int
    context: str
    questions_n: int
    no_passage: bool
    description: str
    fact_budget: int
    structured_output: str
    lossy: bool
    schema_digest: str = ""
    """The fetched schema document's digest (``""`` = no schema). A
    checkpoint file written before this field existed defaults to ``""``
    on load (``from_dict``'s ``.get``) — matches a schema-less rerun, the
    same "new field defaults to the value that changes today's behavior
    least" precedent every other field here already sets. Mirrors
    src/extract.rs's ``CheckpointFingerprint.schema_digest``."""

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "context": self.context,
            "questions_n": self.questions_n,
            "no_passage": self.no_passage,
            "description": self.description,
            "fact_budget": self.fact_budget,
            "structured_output": self.structured_output,
            "lossy": self.lossy,
            "schema_digest": self.schema_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> _CheckpointFingerprint | None:
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                sha256=str(data["sha256"]),
                model=str(data["model"]),
                prompt_version=int(data["prompt_version"]),
                context=str(data["context"]),
                questions_n=int(data["questions_n"]),
                no_passage=bool(data["no_passage"]),
                description=str(data["description"]),
                fact_budget=int(data["fact_budget"]),
                structured_output=str(data["structured_output"]),
                lossy=bool(data["lossy"]),
                schema_digest=str(data.get("schema_digest", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class _CheckpointUnit:
    """One durable unit of extraction work — a top-level chunk (Python has
    no length-ladder split rung yet, but the schema and the unit-hash
    keying already anticipate one, exactly like extract.rs's
    ``CheckpointUnit`` anticipated issue #179's amendment before #210
    implemented it). Keyed by the unit's OWN content hash rather than
    ``chunk_index`` alone, so a resumed run whose chunking differs from a
    prior one never misattributes a piece's output to the wrong text.
    ``user``/``answer`` are stored (not just ``output``) so a reused unit
    can still participate in Stage 2 cross-chunk correction exactly like a
    freshly-extracted one — ``_correct_cross_chunk_issues`` rebuilds a
    chunk's own conversation from these."""

    chunk_index: int
    """The chunk's coordinates in ITS run, kept only for the same "chunk
    i/n" reporting ``ChunkCompleted`` already carries — not part of the
    cache key."""
    output: ModelOutput
    user: str
    """The exact user turn that produced ``output``."""
    answer: str
    """The model's own final accepted answer text."""

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "output": self.output.model_dump(mode="json"),
            "user": self.user,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data: object) -> _CheckpointUnit | None:
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                chunk_index=int(data["chunk_index"]),
                output=ModelOutput.model_validate(data["output"]),
                user=str(data["user"]),
                answer=str(data["answer"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class _DocumentCheckpoints:
    """One document's durable checkpoint state: the settings it was
    extracted under, and every unit completed so far, keyed by content
    hash.

    Serialized as JSON Lines: one header line (``{"fingerprint": {...}}``),
    then one line per unit (``{"unit": "<hash>", ...unit fields}``) —
    issue #<O(N^2) checkpoint saves>. This is what lets
    :meth:`unit_bytes`'s single new line be handed to an append-capable
    store instead of ``to_bytes()``'s full rewrite: an N-chunk document
    then costs O(document size) total on disk, not O(N * document size).
    A pre-migration checkpoint (one JSON object with a top-level ``units``
    key) is still accepted on load — see :meth:`_parse` — so an in-flight
    old-format checkpoint keeps resuming across the migration."""

    fingerprint: _CheckpointFingerprint
    units: dict[str, _CheckpointUnit] = field(default_factory=dict)
    established_on_disk: bool = field(default=False, compare=False, repr=False)
    """True once THIS RUN has written the file at least once via a full
    ``save()`` — only after that may later units in the same run be
    streamed with ``append()`` instead of repeating the full rewrite.
    Always starts ``False`` on a freshly loaded/empty instance, even when
    ``units`` is already non-empty from a resumed run: the very first NEW
    unit recorded this run still needs a full, format-establishing
    ``save()`` (the on-disk file may predate the JSONL migration, or may
    not exist at all yet)."""
    locked_out: bool = field(default=False, compare=False, repr=False)
    """True once a write for this document reported
    :class:`CheckpointLockedError` — another run already holds this
    source. Further writes are skipped outright (rather than repeating,
    and re-warning on, the same conflict every remaining chunk), and the
    end-of-run delete is skipped too, so this run never touches state that
    belongs to whoever holds the lock."""

    @classmethod
    def empty(cls, fingerprint: _CheckpointFingerprint) -> _DocumentCheckpoints:
        return cls(fingerprint=fingerprint, units={})

    @classmethod
    def from_bytes(
        cls, data: bytes | None, fingerprint: _CheckpointFingerprint
    ) -> _DocumentCheckpoints:
        """Missing, unreadable, corrupt, or fingerprint-mismatched
        checkpoints all degrade to "nothing cached" — never an error, and
        never a false reuse of an incompatible output. A single parse
        failure anywhere in the file (including one corrupted unit)
        invalidates the WHOLE file rather than salvaging the units that
        happen to still parse, mirroring Rust's single-``serde_json``-parse
        posture: a partially-trustworthy checkpoint file is treated
        exactly like an absent one. The one deliberate exception is a torn
        FINAL line with no unit lines after it (a crash mid-``append``) —
        see :meth:`_parse_jsonl` — which is discarded rather than
        invalidating everything already durably appended before it."""
        parsed = cls._parse(data, fingerprint)
        return parsed if parsed is not None else cls.empty(fingerprint)

    @classmethod
    def _parse(
        cls, data: bytes | None, fingerprint: _CheckpointFingerprint
    ) -> _DocumentCheckpoints | None:
        if data is None:
            return None
        legacy = cls._sniff_legacy_object(data)
        if legacy is not None:
            return cls._parse_legacy(legacy, fingerprint)
        return cls._parse_jsonl(data, fingerprint)

    @staticmethod
    def _sniff_legacy_object(data: bytes) -> dict[str, object] | None:
        """The pre-migration format was one JSON object with a top-level
        ``units`` key; JSONL's own header line is ALSO a bare ``{...}``
        object (just ``{"fingerprint": ...}``, no ``units`` key), so the
        two can only be told apart by attempting a whole-document parse
        and checking for that key — never by a cheaper prefix check."""
        stripped = data.lstrip()
        if not stripped.startswith(b"{"):
            return None
        try:
            raw = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return raw if isinstance(raw, dict) and "units" in raw else None

    @classmethod
    def _parse_legacy(
        cls, raw: dict[str, object], fingerprint: _CheckpointFingerprint
    ) -> _DocumentCheckpoints | None:
        loaded_fingerprint = _CheckpointFingerprint.from_dict(raw.get("fingerprint"))
        if loaded_fingerprint != fingerprint:
            return None
        raw_units = raw.get("units", {})
        if not isinstance(raw_units, dict):
            return None
        units: dict[str, _CheckpointUnit] = {}
        for key, value in raw_units.items():
            unit = _CheckpointUnit.from_dict(value)
            if unit is None:
                return None
            units[key] = unit
        return cls(fingerprint=fingerprint, units=units)

    @classmethod
    def _parse_jsonl(
        cls, data: bytes, fingerprint: _CheckpointFingerprint
    ) -> _DocumentCheckpoints | None:
        # A well-formed file (even with zero units) always ends in "\n" —
        # every line this class itself ever writes carries one. A file
        # NOT ending in "\n" was torn mid-write (a crash mid-`append`,
        # mid-`save`, or even mid-header); its incomplete last line is
        # discarded rather than treated as a corruption that invalidates
        # every unit durably written before it.
        raw_lines = data.split(b"\n")
        if raw_lines:
            # Well-formed data: this is the trailing empty string left by
            # the final line's own "\n". Torn data: this is the partial,
            # unparseable tail itself. Either way it is never a line to
            # parse — every OTHER element already sits between two
            # newlines (or the start of the file and one), so it is
            # complete regardless of which case this was.
            raw_lines.pop()
        if not raw_lines:
            return None
        try:
            header = json.loads(raw_lines[0])
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(header, dict):
            return None
        loaded_fingerprint = _CheckpointFingerprint.from_dict(header.get("fingerprint"))
        if loaded_fingerprint != fingerprint:
            return None
        units: dict[str, _CheckpointUnit] = {}
        for line in raw_lines[1:]:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not the tolerated case above (that was already popped
                # off before this loop even starts) — a corrupt line
                # anywhere else still invalidates the whole file.
                return None
            if not isinstance(record, dict) or "unit" not in record:
                return None
            unit = _CheckpointUnit.from_dict(record)
            if unit is None:
                return None
            units[str(record["unit"])] = unit  # duplicate unit lines: last one wins
        return cls(fingerprint=fingerprint, units=units)

    @staticmethod
    def _header_line(fingerprint: _CheckpointFingerprint) -> bytes:
        return (json.dumps({"fingerprint": fingerprint.to_dict()}) + "\n").encode("utf-8")

    @staticmethod
    def _unit_line(unit_hash: str, unit: _CheckpointUnit) -> bytes:
        payload: dict[str, object] = {"unit": unit_hash, **unit.to_dict()}
        return (json.dumps(payload) + "\n").encode("utf-8")

    def to_bytes(self) -> bytes:
        """Full JSONL rewrite: the header line plus every accumulated
        unit's own line — what a first write (or a store with no
        ``append``) always persists."""
        lines = [self._header_line(self.fingerprint)]
        lines.extend(self._unit_line(key, unit) for key, unit in self.units.items())
        return b"".join(lines)

    def unit_bytes(self, unit_hash: str) -> bytes:
        """The single JSONL line for one already-``record``ed unit — what
        an append-capable store persists instead of ``to_bytes()``'s full
        rewrite."""
        return self._unit_line(unit_hash, self.units[unit_hash])

    def lookup(self, unit_hash: str) -> _CheckpointUnit | None:
        return self.units.get(unit_hash)

    def record(self, unit_hash: str, unit: _CheckpointUnit) -> None:
        self.units[unit_hash] = unit
