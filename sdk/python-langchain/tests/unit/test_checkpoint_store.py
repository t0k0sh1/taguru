"""FilesystemCheckpointStore and _DocumentCheckpoints: file naming,
atomicity, and the fingerprint/degradation contract in isolation from
TaguruIngester."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from taguru_langchain._extract import ModelOutput
from taguru_langchain.checkpoints import (
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
    assert _checkpoint_file_name("docs/a:b\\c.md") == "docs__a__b__c.md.json"


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
    entries = os.listdir(tmp_path)
    assert entries == ["docs__aomine.md.json"]


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
    assert os.listdir(tmp_path) == ["docs__aomine.md.json"]


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
