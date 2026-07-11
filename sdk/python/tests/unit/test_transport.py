"""Envelope handling, raw-body routes, tolerant decoding, auth headers."""

from __future__ import annotations

import httpx
import pytest

from taguru import DirectoryEntry, TaguruError

from .conftest import async_client, err_response, ok_response, sync_client

DIRECTORY_ROW = {
    "name": "sake",
    "description": "酒蔵の知識",
    "pinned": False,
    "loaded": True,
    "dice_floor": None,
    "semantic_floor": 0.35,
    "stats": {
        "associations": 6,
        "concepts": 5,
        "labels": 4,
        "sources": 1,
        "footprint_bytes": 4096,
        "top_concepts": [{"label": "青嶺酒造", "count": 4}],
        "label_sample": ["代表銘柄"],
    },
    "usage": {
        "reads": 1,
        "empty_reads": 0,
        "writes": 2,
        "last_read_epoch": 100,
        "last_write_epoch": 90,
    },
}


def test_envelope_unwraps_and_decodes() -> None:
    client = sync_client(lambda _req: ok_response(DIRECTORY_ROW))
    entry = client.contexts.get("sake")
    assert isinstance(entry, DirectoryEntry)
    assert entry.stats.top_concepts[0].label == "青嶺酒造"
    assert entry.usage.writes == 2
    assert entry.dice_floor is None


def test_unknown_fields_are_ignored() -> None:
    """The protocol promises additive evolution; new fields must not break us."""
    row = {**DIRECTORY_ROW, "brand_new_field": {"nested": True}}
    row["stats"] = {**DIRECTORY_ROW["stats"], "another_new_stat": 7}  # type: ignore[dict-item]
    client = sync_client(lambda _req: ok_response(row))
    entry = client.contexts.get("sake")
    assert entry.name == "sake"


def test_describe_null_result_is_none_not_error() -> None:
    client = sync_client(lambda _req: ok_response(None))
    assert client.context("sake").describe("unknown") is None


def test_non_envelope_2xx_raises_protocol_error() -> None:
    client = sync_client(lambda _req: httpx.Response(200, json={"weird": True}))
    with pytest.raises(TaguruError, match="envelope"):
        client.context("sake").recall("cue")


def test_raw_text_routes_bypass_envelope() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            return httpx.Response(200, text="ok")
        if req.url.path == "/metrics":
            return httpx.Response(200, text="taguru_requests_total 1\n")
        if req.url.path == "/protocol":
            return httpx.Response(200, text="# Taguru client protocol\n")
        raise AssertionError(req.url.path)

    client = sync_client(handler)
    client.health()
    assert "taguru_requests_total" in client.metrics()
    assert client.protocol().startswith("# Taguru client protocol")


def test_export_returns_raw_ndjson() -> None:
    ndjson = '{"taguru_batch":1,"context":"sake","source":"a"}\n{"passage":"text"}\n'
    client = sync_client(lambda _req: httpx.Response(200, text=ndjson))
    assert client.context("sake").export() == ndjson


def test_import_normalizes_single_outcome_to_list() -> None:
    outcome = {
        "context": "sake",
        "source": "a",
        "created": True,
        "retracted": 0,
        "associations": 2,
        "aliases": 0,
        "passage_stored": True,
        "passage_dropped": False,
        "questions_stored": 0,
        "questions_dropped": 0,
        "sections_stored": 0,
        "sections_dropped": 0,
        "association_paragraphs_dropped": 0,
    }
    client = sync_client(lambda _req: ok_response(outcome))
    outcomes = client.import_batches('{"taguru_batch":1}')
    assert len(outcomes) == 1
    assert outcomes[0].context == "sake"

    client = sync_client(lambda _req: ok_response({"batches": [outcome, outcome]}))
    outcomes = client.import_batches('{"taguru_batch":1}')
    assert [o.source for o in outcomes] == ["a", "a"]


def test_bearer_header_present_only_with_api_key() -> None:
    seen: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get("authorization"))
        return ok_response({"total": 0, "matches": []})

    with_key = sync_client(handler, api_key="secret")
    with_key.context("sake").recall("cue")
    without_key = sync_client(handler, api_key="")
    without_key.context("sake").recall("cue")
    assert seen == ["Bearer secret", None]


def test_context_names_are_path_encoded() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.raw_path.decode())
        return ok_response({"total": 0, "matches": []})

    client = sync_client(handler)
    client.context("日本 酒/テスト").recall("cue")
    assert paths == [
        "/contexts/%E6%97%A5%E6%9C%AC%20%E9%85%92%2F%E3%83%86%E3%82%B9%E3%83%88/recall"
    ]


def test_query_sends_one_or_many_and_drops_none() -> None:
    bodies: list[bytes] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(req.content)
        return ok_response({"total": 0, "matches": []})

    client = sync_client(handler)
    client.context("sake").query(label=["住所", "職歴"], subject="高瀬")
    assert bodies[0] == '{"subject":"高瀬","label":["住所","職歴"]}'.encode()


async def test_async_client_mirrors_sync() -> None:
    client = async_client(lambda _req: ok_response(DIRECTORY_ROW))
    entry = await client.contexts.get("sake")
    assert entry.name == "sake"
    await client.close()


def test_error_message_and_body_survive() -> None:
    client = sync_client(lambda _req: err_response(404, "context 'x' does not exist"), retries=0)
    with pytest.raises(TaguruError) as excinfo:
        client.contexts.get("x")
    assert excinfo.value.body == {
        "status": "error",
        "error": "context 'x' does not exist",
        "time": 0.001,
    }
