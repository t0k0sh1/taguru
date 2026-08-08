"""FilesystemCheckpointStore and _DocumentCheckpoints: file naming,
atomicity, and the fingerprint/degradation contract in isolation from
TaguruIngester."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from taguru_langchain._extract import ModelOutput
from taguru_langchain.checkpoints import (
    CheckpointLockedError,
    FilesystemCheckpointStore,
    _checkpoint_file_name,
    _CheckpointFingerprint,
    _CheckpointUnit,
    _DocumentCheckpoints,
)


def test_filesystem_store_round_trips_units(tmp_path: Path) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    assert store.load("docs/aomine.md") is None
    store.save("docs/aomine.md", b'{"fingerprint": {}, "units": {}}')
    assert store.load("docs/aomine.md") == b'{"fingerprint": {}, "units": {}}'
    store.save("docs/aomine.md", b'{"fingerprint": {}, "units": {"x": 1}}')
    assert store.load("docs/aomine.md") == b'{"fingerprint": {}, "units": {"x": 1}}'


def test_checkpoint_file_name_flattens_path_separators() -> None:
    name = _checkpoint_file_name("docs/a:b\\c.md")
    assert name.startswith("docs__a__b__c.md-")
    assert name.endswith(".json")


def test_checkpoint_file_name_always_carries_a_hash_suffix() -> None:
    # Distinct short source ids that flatten to the same string must not
    # collide: "a/b", "a:b", and "a__b" all flatten to "a__b", so only an
    # unconditional hash suffix keeps them apart (issue #227).
    name_a = _checkpoint_file_name("a/b")
    name_b = _checkpoint_file_name("a:b")
    name_c = _checkpoint_file_name("a__b")
    assert len({name_a, name_b, name_c}) == 3
    for name in (name_a, name_b, name_c):
        assert name.startswith("a__b-")
        assert name.endswith(".json")


def test_checkpoint_file_name_truncates_long_sources_with_a_hash_suffix() -> None:
    # 50 Japanese characters * 3 bytes/char = 150 UTF-8 bytes, over the
    # 120-byte flatten threshold, and straddling the 96-byte truncation
    # point mid-codepoint (96 is not a multiple of 3) — exercises the
    # UTF-8 boundary backoff, not just byte-length truncation.
    source = "文書/" + "あ" * 50 + ".md"
    name = _checkpoint_file_name(source)
    assert name.endswith(".json")
    stem = name.removesuffix(".json")
    prefix, _, suffix = stem.rpartition("-")
    assert len(prefix.encode("utf-8")) <= 96
    assert len(suffix) == 16
    # The prefix is a clean, valid-UTF-8 truncation of the flattened
    # source (no split codepoint) — it must itself be a text prefix of
    # "文書__" + 50 "あ"s, never a mangled partial character.
    assert ("文書__" + "あ" * 50).startswith(prefix)


def test_checkpoint_file_name_truncation_keeps_distinct_sources_distinct() -> None:
    # Two sources sharing the same >120-byte prefix must still map to
    # different files — the hash suffix, not the truncated prefix alone,
    # provides uniqueness.
    base = "d" * 130
    name_a = _checkpoint_file_name(base + "-a")
    name_b = _checkpoint_file_name(base + "-b")
    assert name_a != name_b


def test_atomic_write_leaves_no_tmp_litter(tmp_path: Path) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    store.save("docs/aomine.md", b"{}")
    # The checkpoint file itself, plus the sidecar advisory-lock file
    # `save` creates on a source's first write (issue #<concurrent-run
    # clobber>) — no OTHER litter (staged temp files, ...) survives.
    checkpoint_name = _checkpoint_file_name("docs/aomine.md")
    assert set(os.listdir(tmp_path)) == {checkpoint_name, f"{checkpoint_name}.lock"}


def test_a_failed_replace_keeps_the_old_file_and_cleans_up_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    store.save("docs/aomine.md", b"original")

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        store.save("docs/aomine.md", b"replacement")

    assert store.load("docs/aomine.md") == b"original"
    checkpoint_name = _checkpoint_file_name("docs/aomine.md")
    assert set(os.listdir(tmp_path)) == {checkpoint_name, f"{checkpoint_name}.lock"}


def test_load_never_creates_the_directory_and_save_creates_it_on_demand(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    store = FilesystemCheckpointStore(directory)
    assert store.load("docs/aomine.md") is None
    assert not directory.exists()
    store.save("docs/aomine.md", b"{}")
    assert directory.exists()


def test_delete_of_a_missing_file_is_a_no_op(tmp_path: Path) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    store.delete("docs/never-saved.md")  # must not raise


# -- _DocumentCheckpoints: the fingerprint gate and degrade-to-empty posture --


def _fingerprint(
    *,
    sha256: str = "deadbeef",
    model: str = "fake-model",
    prompt_version: int = 2,
    context: str = "sake",
    questions_n: int = 2,
    no_passage: bool = False,
    description: str = "",
    fact_budget: int = 0,
    structured_output: str = "",
    lossy: bool = False,
) -> _CheckpointFingerprint:
    return _CheckpointFingerprint(
        sha256=sha256,
        model=model,
        prompt_version=prompt_version,
        context=context,
        questions_n=questions_n,
        no_passage=no_passage,
        description=description,
        fact_budget=fact_budget,
        structured_output=structured_output,
        lossy=lossy,
    )


def test_document_checkpoints_round_trips_through_bytes() -> None:
    fingerprint = _fingerprint()
    unit = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u", answer="a")
    checkpoints = _DocumentCheckpoints(fingerprint=fingerprint, units={"hash1": unit})
    restored = _DocumentCheckpoints.from_bytes(checkpoints.to_bytes(), fingerprint)
    assert restored.lookup("hash1") == unit


def test_document_checkpoints_degrades_to_empty_on_fingerprint_mismatch() -> None:
    fingerprint = _fingerprint()
    unit = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u", answer="a")
    checkpoints = _DocumentCheckpoints(fingerprint=fingerprint, units={"hash1": unit})
    other = _fingerprint(model="a-different-model")
    restored = _DocumentCheckpoints.from_bytes(checkpoints.to_bytes(), other)
    assert restored.units == {}
    assert restored.fingerprint == other


def test_document_checkpoints_degrades_to_empty_on_missing_data() -> None:
    fingerprint = _fingerprint()
    assert _DocumentCheckpoints.from_bytes(None, fingerprint).units == {}


@pytest.mark.parametrize(
    "data",
    [b"not json at all", b"[1, 2, 3]", b'{"units": {}}', b'{"fingerprint": {}, "units": "nope"}'],
    ids=["invalid_json", "wrong_top_level_type", "missing_fingerprint", "units_wrong_type"],
)
def test_document_checkpoints_degrades_to_empty_on_malformed_bytes(data: bytes) -> None:
    fingerprint = _fingerprint()
    assert _DocumentCheckpoints.from_bytes(data, fingerprint).units == {}


def test_document_checkpoints_a_single_corrupt_unit_invalidates_the_whole_file() -> None:
    fingerprint = _fingerprint()
    good_unit = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u", answer="a")
    payload = {
        "fingerprint": fingerprint.to_dict(),
        "units": {
            "good": good_unit.to_dict(),
            "bad": {"chunk_index": 0},  # missing output/user/answer
        },
    }
    data = json.dumps(payload).encode("utf-8")
    restored = _DocumentCheckpoints.from_bytes(data, fingerprint)
    # The whole file degrades to empty — the still-good "good" unit is NOT
    # salvaged, matching Rust's single-serde-parse posture.
    assert restored.units == {}


# -- _DocumentCheckpoints: JSON Lines format (issue #<O(N^2) checkpoint saves>) --


def test_document_checkpoints_round_trips_two_units_through_jsonl_bytes() -> None:
    fingerprint = _fingerprint()
    unit1 = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u1", answer="a1")
    unit2 = _CheckpointUnit(chunk_index=1, output=ModelOutput(), user="u2", answer="a2")
    checkpoints = _DocumentCheckpoints(fingerprint=fingerprint, units={"h1": unit1, "h2": unit2})
    restored = _DocumentCheckpoints.from_bytes(checkpoints.to_bytes(), fingerprint)
    assert restored.lookup("h1") == unit1
    assert restored.lookup("h2") == unit2


def test_document_checkpoints_tolerates_a_torn_trailing_line_from_a_crash_mid_append() -> None:
    fingerprint = _fingerprint()
    unit1 = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u1", answer="a1")
    unit2 = _CheckpointUnit(chunk_index=1, output=ModelOutput(), user="u2", answer="a2")
    checkpoints = _DocumentCheckpoints(fingerprint=fingerprint, units={"h1": unit1, "h2": unit2})
    data = checkpoints.to_bytes()
    # Simulate a crash mid-`append`: the last line's own trailing newline
    # (and a few bytes before it) never made it to disk.
    torn = data[:-6]
    assert not torn.endswith(b"\n")
    restored = _DocumentCheckpoints.from_bytes(torn, fingerprint)
    # The complete "h1" unit (and the header) survive; the torn "h2" line
    # is discarded rather than invalidating everything already durably
    # appended before it.
    assert restored.lookup("h1") == unit1
    assert restored.lookup("h2") is None


def test_document_checkpoints_still_loads_the_legacy_single_object_format_with_real_units() -> None:
    fingerprint = _fingerprint()
    unit = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u", answer="a")
    legacy_payload = {"fingerprint": fingerprint.to_dict(), "units": {"h1": unit.to_dict()}}
    data = json.dumps(legacy_payload).encode("utf-8")
    # An in-flight pre-migration checkpoint keeps resuming exactly as
    # before the JSONL migration — never treated as corrupt just because
    # it predates it.
    restored = _DocumentCheckpoints.from_bytes(data, fingerprint)
    assert restored.lookup("h1") == unit


def test_document_checkpoints_jsonl_duplicate_unit_lines_let_the_last_one_win() -> None:
    fingerprint = _fingerprint()
    stale = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="stale", answer="stale")
    fresh = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="fresh", answer="fresh")
    header = json.dumps({"fingerprint": fingerprint.to_dict()}).encode("utf-8")
    stale_line = json.dumps({"unit": "h1", **stale.to_dict()}).encode("utf-8")
    fresh_line = json.dumps({"unit": "h1", **fresh.to_dict()}).encode("utf-8")
    data = header + b"\n" + stale_line + b"\n" + fresh_line + b"\n"
    restored = _DocumentCheckpoints.from_bytes(data, fingerprint)
    assert restored.lookup("h1") == fresh


def test_document_checkpoints_a_corrupt_middle_line_in_jsonl_invalidates_the_whole_file() -> None:
    fingerprint = _fingerprint()
    good = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u", answer="a")
    header = json.dumps({"fingerprint": fingerprint.to_dict()}).encode("utf-8")
    bad_line = json.dumps({"unit": "bad", "chunk_index": 0}).encode(
        "utf-8"
    )  # no output/user/answer
    good_line = json.dumps({"unit": "good", **good.to_dict()}).encode("utf-8")
    # The bad line has a well-formed trailing newline AND a line after it
    # — genuine corruption, not the crash-mid-append tail the test above
    # tolerates.
    data = header + b"\n" + bad_line + b"\n" + good_line + b"\n"
    restored = _DocumentCheckpoints.from_bytes(data, fingerprint)
    assert restored.units == {}


# -- FilesystemCheckpointStore: append() and the advisory per-source lock ------
# (issue #<O(N^2) checkpoint saves> / issue #<concurrent-run clobber>) ---------


def test_filesystem_store_append_adds_only_the_new_record_not_a_full_rewrite(
    tmp_path: Path,
) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    fingerprint = _fingerprint()
    unit1 = _CheckpointUnit(chunk_index=0, output=ModelOutput(), user="u1", answer="a1")
    checkpoints = _DocumentCheckpoints(fingerprint=fingerprint, units={"h1": unit1})
    store.save("docs/aomine.md", checkpoints.to_bytes())
    size_after_save = store.path_for("docs/aomine.md").stat().st_size

    checkpoints.record(
        "h2", _CheckpointUnit(chunk_index=1, output=ModelOutput(), user="u2", answer="a2")
    )
    new_record = checkpoints.unit_bytes("h2")
    store.append("docs/aomine.md", new_record)
    size_after_append = store.path_for("docs/aomine.md").stat().st_size

    # The file grew by exactly the new record's own bytes — proof `append`
    # never touched (let alone rewrote) what `save` already wrote.
    assert size_after_append - size_after_save == len(new_record)
    restored = _DocumentCheckpoints.from_bytes(store.load("docs/aomine.md"), fingerprint)
    assert set(restored.units) == {"h1", "h2"}


def test_filesystem_store_save_raises_when_another_holder_already_locked_the_source(
    tmp_path: Path,
) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    lock_path = Path(f"{store.path_for('docs/aomine.md')}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    external = open(lock_path, "a+b")
    fcntl.flock(external.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(CheckpointLockedError, match="docs/aomine.md"):
            store.save("docs/aomine.md", b"{}")
    finally:
        external.close()


def test_filesystem_store_append_raises_when_another_holder_already_locked_the_source(
    tmp_path: Path,
) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    lock_path = Path(f"{store.path_for('docs/aomine.md')}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    external = open(lock_path, "a+b")
    fcntl.flock(external.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(CheckpointLockedError, match="docs/aomine.md"):
            store.append("docs/aomine.md", b'{"unit": "h1"}\n')
    finally:
        external.close()


def test_filesystem_store_a_second_save_for_a_source_it_already_locked_is_not_a_re_acquisition(
    tmp_path: Path,
) -> None:
    """The common case — every write after a document's first goes through
    `_acquire_lock` again — must be a no-op against the SAME instance's own
    lock, never mistaken for contention."""
    store = FilesystemCheckpointStore(tmp_path)
    store.save("docs/aomine.md", b"{}")
    store.save("docs/aomine.md", b"{}")  # must not raise CheckpointLockedError


def test_filesystem_store_delete_releases_the_held_lock(tmp_path: Path) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    store.save("docs/aomine.md", b"{}")
    lock_path = Path(f"{store.path_for('docs/aomine.md')}.lock")

    external = open(lock_path, "a+b")
    try:
        with pytest.raises(OSError):
            fcntl.flock(external.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        store.delete("docs/aomine.md")

        fcntl.flock(external.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise now
        fcntl.flock(external.fileno(), fcntl.LOCK_UN)
    finally:
        external.close()


def test_filesystem_store_close_releases_every_held_lock(tmp_path: Path) -> None:
    store = FilesystemCheckpointStore(tmp_path)
    store.save("a.md", b"{}")
    store.save("b.md", b"{}")

    store.close()

    for source in ("a.md", "b.md"):
        lock_path = Path(f"{store.path_for(source)}.lock")
        external = open(lock_path, "a+b")
        try:
            fcntl.flock(external.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
            fcntl.flock(external.fileno(), fcntl.LOCK_UN)
        finally:
            external.close()
