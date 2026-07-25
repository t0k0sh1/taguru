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
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from ._extract import ModelOutput


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
    yet never gets a file."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)

    def path_for(self, source: str) -> Path:
        """The file ``source`` maps to — public so an operator can inspect
        or remove one document's checkpoints by hand (the SDK has no
        ``--force`` equivalent; deleting this file is the manual
        invalidation path)."""
        return self._directory / _checkpoint_file_name(source)

    def load(self, source: str) -> bytes | None:
        try:
            return self.path_for(source).read_bytes()
        except FileNotFoundError:
            return None

    def save(self, source: str, data: bytes) -> None:
        _write_atomic(self.path_for(source), data)

    def delete(self, source: str) -> None:
        self.path_for(source).unlink(missing_ok=True)


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
    hash. Serialized as one small JSON object, rewritten in full
    (``_write_atomic``) after every new unit lands — small enough that a
    read-modify-write-whole-file is simpler, and just as durable, as an
    append-only log."""

    fingerprint: _CheckpointFingerprint
    units: dict[str, _CheckpointUnit] = field(default_factory=dict)

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
        exactly like an absent one."""
        parsed = cls._parse(data, fingerprint)
        return parsed if parsed is not None else cls.empty(fingerprint)

    @classmethod
    def _parse(
        cls, data: bytes | None, fingerprint: _CheckpointFingerprint
    ) -> _DocumentCheckpoints | None:
        if data is None:
            return None
        try:
            raw = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
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

    def to_bytes(self) -> bytes:
        payload = {
            "fingerprint": self.fingerprint.to_dict(),
            "units": {key: unit.to_dict() for key, unit in self.units.items()},
        }
        return json.dumps(payload).encode("utf-8")

    def lookup(self, unit_hash: str) -> _CheckpointUnit | None:
        return self.units.get(unit_hash)

    def record(self, unit_hash: str, unit: _CheckpointUnit) -> None:
        self.units[unit_hash] = unit
