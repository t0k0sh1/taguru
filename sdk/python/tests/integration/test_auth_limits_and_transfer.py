"""Auth, key scopes, server limits, and export/import round trips."""

from __future__ import annotations

import pytest

from taguru import (
    AuthenticationError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    Taguru,
)

from .conftest import SpawnedServer
from .test_full_loop import AOMINE_DOC, seed


def test_wrong_or_missing_token_is_401(server: SpawnedServer) -> None:
    with Taguru(server.base_url, "wrong-token", retries=0) as client:
        with pytest.raises(AuthenticationError) as unauthorized:
            client.contexts.list()
        assert unauthorized.value.code == "unauthorized"
    with Taguru(server.base_url, api_key="", retries=0) as client:
        with pytest.raises(AuthenticationError):
            client.contexts.list()


def test_probes_stay_token_free(server: SpawnedServer) -> None:
    with Taguru(server.base_url, api_key="", retries=0) as client:
        client.live()
        client.health()
        assert "taguru" in client.metrics()


def test_read_scoped_key_reads_but_cannot_write(
    client: Taguru, reader_client: Taguru, fresh_name: str
) -> None:
    seed(client, fresh_name)
    ctx = reader_client.context(fresh_name)

    assert ctx.recall("青嶺酒造").total > 0  # the retrieval loop is granted
    with pytest.raises(PermissionDeniedError):
        ctx.add_associations([{"subject": "s", "label": "l", "object": "o", "weight": 1.0}])
    with pytest.raises(PermissionDeniedError):
        reader_client.contexts.create("reader-made")
    with pytest.raises(PermissionDeniedError):
        reader_client.contexts.delete(fresh_name)  # delete is admin-only
    with pytest.raises(PermissionDeniedError):
        reader_client.flush()
    client.contexts.delete(fresh_name)


def test_rate_limit_answers_429_with_retry_after(spawn_server, tmp_path) -> None:
    server = spawn_server({"TAGURU_API_TOKEN": "rl-token", "TAGURU_RATE_LIMIT_PER_MIN": "3"})
    with Taguru(server.base_url, "rl-token", retries=0) as client:
        client.wait_until_ready(timeout=30)
        with pytest.raises(RateLimitError) as excinfo:
            for _ in range(10):
                client.contexts.list()
        assert excinfo.value.retry_after is not None
        assert excinfo.value.retry_after > 0
        # Probes stay exempt even while the key is throttled.
        client.health()


def test_body_cap_answers_413(spawn_server) -> None:
    server = spawn_server({"TAGURU_API_TOKEN": "cap-token", "TAGURU_MAX_BODY_BYTES": "1024"})
    with Taguru(server.base_url, "cap-token", retries=0) as client:
        client.wait_until_ready(timeout=30)
        client.contexts.create("cap")
        big = [
            {"subject": f"s{i}", "label": "l", "object": "o" * 40, "weight": 1.0}
            for i in range(200)
        ]
        with pytest.raises(PayloadTooLargeError) as capped:
            client.context("cap").add_associations(big)
        # The cap breach speaks the JSON error shape, code included.
        assert capped.value.code == "payload_too_large"


def test_export_import_round_trip(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)
    ctx.add_aliases(concepts={"Aomine": "青嶺酒造"})
    stream = ctx.export()
    assert stream.count('"taguru_batch"') >= 1

    restored_name = f"{fresh_name}-restored"
    result = client.import_batches(stream.replace(f'"{fresh_name}"', f'"{restored_name}"'))
    assert all(outcome.context == restored_name for outcome in result.batches)
    assert any(outcome.created for outcome in result.batches)

    restored = client.context(restored_name)
    assert restored.query(subject="青嶺酒造", label="杜氏").matches[0].object == "高瀬"
    assert restored.lookup_passages(["docs/aomine.md"]).passages["docs/aomine.md"] == AOMINE_DOC
    assert restored.resolve("Aomine")[0].name == "青嶺酒造"

    # Re-importing the same stream is a per-source replace: weights must not
    # double-count.
    before = restored.query(subject="青嶺酒造", label="杜氏").matches[0]
    client.import_batches(stream.replace(f'"{fresh_name}"', f'"{restored_name}"'))
    after = restored.query(subject="青嶺酒造", label="杜氏").matches[0]
    assert after.weight == before.weight
    assert after.count == before.count

    client.contexts.delete(fresh_name)
    client.contexts.delete(restored_name)


def test_export_stream_and_file(client: Taguru, fresh_name: str, tmp_path) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)
    streamed = b"".join(chunk for chunk in ctx.export_stream())
    assert streamed.decode("utf-8") == ctx.export()

    target = tmp_path / "backup.jsonl"
    ctx.export_to_file(target)
    assert target.read_bytes() == streamed
    client.contexts.delete(fresh_name)


def test_import_file(client: Taguru, fresh_name: str, tmp_path) -> None:
    batch = (
        f'{{"taguru_batch": 1, "context": "{fresh_name}", "source": "f.md", '
        f'"create": {{"description": "from file"}}}}\n'
        '{"passage": "ファイルからの本文。"}\n'
        '{"subject": "a", "label": "b", "object": "c", "weight": 1.0}\n'
    )
    path = tmp_path / "batch.jsonl"
    path.write_text(batch, encoding="utf-8")
    result = client.import_file(path)
    assert result.batches[0].created
    assert result.batches[0].associations == 1
    assert result.batches[0].passage_stored
    client.contexts.delete(fresh_name)
