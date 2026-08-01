"""S3Connector/sync_object_storage (ADR 0007 §9, issue #351): dispatch by
extension/content-type, S3 identity re-stamping, tag mapping, the
two-layer checkpoint (fetch-skip + ingest-skip), deletion policy, and
``dry_run`` — exercised against :class:`~tests._s3.FakeObjectStore` (no
filesystem, no network) and the same in-memory ``FakeServer`` every other
bridge test in this suite already uses."""

from __future__ import annotations

import dataclasses
import io
import json
from datetime import datetime, timezone

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from taguru import AsyncTaguru, Taguru

from taguru_langchain import TaguruIngester
from taguru_langchain.ingest_connectors import (
    ObjectNotFoundError,
    PermanentStoreError,
    S3Connector,
    S3ObjectCheckpoint,
    TransientStoreError,
    sync_object_storage,
)
from taguru_langchain.ingest_connectors.document import MAX_NAME_BYTES, MAX_TAG_BYTES

from .._s3 import FakeObjectStore
from .conftest import FakeServer
from .test_checkpoints import RecordingCheckpointStore

EMPTY_ANSWER = json.dumps({"associations": [], "aliases": [], "questions": []})


def _ingester(
    sync_client: Taguru | None, async_client: AsyncTaguru | None, **overrides: object
) -> TaguruIngester:
    return TaguruIngester(
        context="sake",
        llm=FakeListChatModel(responses=[EMPTY_ANSWER]),
        client=sync_client,
        async_client=async_client,
        **overrides,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# S3Connector — dispatch, identity re-stamping, size/source-id/tag handling
# ---------------------------------------------------------------------------


def test_dispatches_a_text_object_and_restamps_its_identity() -> None:
    store = FakeObjectStore("reports")
    store.put("notes.md", b"paragraph one.\n\nparagraph two.")
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.source == "s3://reports/notes.md"
    assert document.metadata.origin_uri == "s3://reports/notes.md"
    assert document.metadata.display_name == "notes.md"
    assert document.fingerprint_inputs.parser == "taguru-text-connector"
    assert document.diagnostics == ()


def test_dispatches_an_html_object() -> None:
    store = FakeObjectStore("reports")
    store.put("page.html", b"<html><body><p>hello world</p></body></html>")
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.text == "hello world"
    assert document.fingerprint_inputs.parser == "taguru-html-connector"


def test_dispatches_a_pdf_object() -> None:
    pytest.importorskip("pypdf")
    from .._pdfs import text_pdf  # noqa: E402

    store = FakeObjectStore("reports")
    store.put("q1.pdf", text_pdf(["quarterly results"]))
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.text == "quarterly results"
    assert document.fingerprint_inputs.parser == "taguru-pdf-connector"
    assert document.locators[0].locator.kind == "page"


def test_dispatches_a_docx_object() -> None:
    pytest.importorskip("docx")
    from .._docx import docx_bytes, para  # noqa: E402

    store = FakeObjectStore("reports")
    store.put("q1.docx", docx_bytes(para("quarterly results")))
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.text == "quarterly results"
    assert document.fingerprint_inputs.parser == "taguru-docx-connector"


def test_unsupported_extension_still_gets_one_content_type_check() -> None:
    """A key whose extension names no installed connector still gets ONE
    fetch to consult content-type (ADR 0007 §9's fallback) before giving
    up — a mislabeled key legitimately might be something else; the size
    cap (checked from the LISTING, before any fetch) is what actually
    bounds the cost of that one attempt."""
    store = FakeObjectStore()
    store.put("archive.zip", b"PK\x03\x04...")
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.text == ""
    assert document.diagnostics[0].code == "unsupported_format"
    assert store.get_calls == {"archive.zip": 1}


def test_missing_extension_falls_back_to_content_type() -> None:
    store = FakeObjectStore()
    store.put("data", b"plain body", content_type="text/plain")
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.text == "plain body"
    assert document.fingerprint_inputs.parser == "taguru-text-connector"
    assert store.get_calls == {"data": 1}


def test_missing_extension_and_unresolvable_content_type_is_unsupported() -> None:
    store = FakeObjectStore()
    store.put("data", b"\x00\x01binary", content_type="application/octet-stream")
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.diagnostics[0].code == "unsupported_format"
    assert store.get_calls == {"data": 1}  # fetched once to consult content-type, never again


def test_size_cap_still_applies_on_the_content_type_fallback_fetch() -> None:
    """The content-type-fallback path (no extension match) fetches BEFORE
    a connector is even chosen — that fetch must still respect
    `max_file_bytes`, the same as the listing-driven pre-fetch check and
    `read()`'s own post-fetch check. A prior fix only guarded the LATTER
    two call sites, silently reading an oversized, extension-and-
    content-type-unresolvable object fully into memory before reporting
    `unsupported_format`. Uses `read()` (size=-1, no listing) so the
    listing-driven pre-fetch check — which would otherwise catch this
    100-byte object on its own — cannot mask the fallback-fetch site's own
    gap."""
    store = FakeObjectStore("reports")
    store.put("data", b"x" * 100, content_type="application/octet-stream")
    connector = S3Connector(store, max_file_bytes=10)

    document = connector.read("s3://reports/data")

    assert document.diagnostics[0].code == "content_too_large"


def test_content_too_large_is_reported_without_any_fetch() -> None:
    store = FakeObjectStore()
    store.put("big.txt", b"x" * 100)
    connector = S3Connector(store, max_file_bytes=10)

    document = connector.read_object(next(store.list("")))

    assert document.diagnostics[0].code == "content_too_large"
    assert store.get_calls == {}


def test_oversized_source_id_is_reported_without_any_fetch() -> None:
    store = FakeObjectStore("b")
    long_key = "x" * (MAX_NAME_BYTES + 10) + ".txt"
    store.put(long_key, b"hello")
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.diagnostics[0].code == "source_id_too_long"
    assert store.get_calls == {}


def test_object_tags_are_mapped_onto_metadata_sorted() -> None:
    store = FakeObjectStore()
    store.put("a.md", b"hello", tags={"team": "", "project": "quarterly"})
    connector = S3Connector(store)

    document = connector.read_object(next(store.list("")))

    assert document.metadata.tags == ("project=quarterly", "team")


def test_excess_tags_are_dropped_and_counted() -> None:
    store = FakeObjectStore()
    tags = {f"tag{i:03d}": "v" for i in range(35)}
    store.put("a.md", b"hello", tags=tags)
    connector = S3Connector(store)

    result = connector.read_object_result(next(store.list("")))

    assert len(result.document.metadata.tags) == 32
    assert result.tags_dropped == 3


def test_oversized_tag_value_is_dropped_and_counted() -> None:
    store = FakeObjectStore()
    store.put("a.md", b"hello", tags={"ok": "fine", "huge": "v" * MAX_TAG_BYTES})
    connector = S3Connector(store)

    result = connector.read_object_result(next(store.list("")))

    assert result.document.metadata.tags == ("ok=fine",)
    assert result.tags_dropped == 1


def test_read_via_the_plain_connector_protocol_entry_point() -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"hello")
    connector = S3Connector(store)

    assert connector.supports("s3://reports/a.md")
    assert not connector.supports("s3://other-bucket/a.md")
    document = connector.read("s3://reports/a.md")
    assert document.source == "s3://reports/a.md"


def test_read_enforces_max_file_bytes_even_without_a_listing() -> None:
    """`read()` synthesizes `size=-1` (no `ObjectMeta` from a real listing),
    which skips the pre-fetch size check entirely — the cap must still
    hold via a post-fetch check, or `read()` could pull an object of any
    size fully into memory."""
    store = FakeObjectStore("reports")
    store.put("big.txt", b"x" * 100)
    connector = S3Connector(store, max_file_bytes=10)

    document = connector.read("s3://reports/big.txt")

    assert document.diagnostics[0].code == "content_too_large"


def test_parse_options_digest_is_stable_and_reflects_config() -> None:
    store = FakeObjectStore()
    a = S3Connector(store, max_file_bytes=1024)
    b = S3Connector(store, max_file_bytes=1024)
    c = S3Connector(store, max_file_bytes=2048)
    assert a.parse_options_digest == b.parse_options_digest
    assert a.parse_options_digest != c.parse_options_digest


# ---------------------------------------------------------------------------
# sync_object_storage — orchestration: checkpoint layers, deletion, dry_run
# ---------------------------------------------------------------------------


def test_a_full_pass_imports_every_discovered_object(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha content")
    store.put("b.md", b"beta content")
    checkpoints = RecordingCheckpointStore()

    report = sync_object_storage(
        store, "", ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )

    # Counts tally each source's LAST phase only (ADR 0007 §11) — a source
    # that reached "imported" never counts under "discovered" too.
    assert report.imported == 2
    assert report.discovered == 0
    assert report.failed == 0
    assert len(fake_server.imported) == 2


def test_a_second_pass_with_no_change_skips_the_fetch_entirely(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha content", etag="e1")
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)
    assert store.get_calls == {"a.md": 1}

    report = sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)

    assert report.unchanged == 1
    assert report.imported == 0
    assert store.get_calls == {"a.md": 1}  # no second fetch at all
    assert len(fake_server.imported) == 1  # no second import either


def test_a_metadata_only_change_skips_the_re_import_but_not_the_fetch(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """A listing-metadata bump (e.g. a tag/touch) with byte-identical
    content re-fetches (Layer 1 misses) but does not re-ingest (Layer 2,
    ConnectorCheckpoint, still hits) — ADR 0007 §6.3."""
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha content", etag="e1")
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)

    store.put("a.md", b"alpha content", etag="e2")  # same bytes, new etag
    report = sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)

    assert report.unchanged == 1
    assert store.get_calls == {"a.md": 2}  # re-fetched
    assert len(fake_server.imported) == 1  # but not re-imported


def test_genuinely_changed_content_is_re_imported(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha content", etag="e1")
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)
    store.put("a.md", b"alpha content, revised", etag="e2")
    report = sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)

    assert report.imported == 1
    assert len(fake_server.imported) == 2


def test_deletion_default_policy_only_reports_never_retracts(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    store.put("b.md", b"beta")
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)
    store.delete("b.md")
    report = sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)

    assert report.deleted_detected == 1
    assert report.retracted == 0
    assert fake_server.retracted == []


def test_deletion_retract_policy_withdraws_only_the_detected_deletion(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    store.put("b.md", b"beta")
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)
    store.delete("b.md")
    report = sync_object_storage(
        store, "", ingester=ingester, checkpoints=checkpoints, deletion_policy="retract"
    )

    assert report.deleted_detected == 1
    assert report.retracted == 1
    assert fake_server.retracted == ["s3://reports/b.md"]


def test_retract_clears_both_checkpoints_so_a_recreated_object_is_reimported(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """Without clearing `S3ObjectCheckpoint`/`ConnectorCheckpoint` on
    retract, a non-versioned bucket's object deleted then recreated
    elsewhere with the SAME (size, last_modified) — the fingerprint tier
    that combination falls back to — would read as `unchanged` on the
    very next pass and never be re-ingested, even though the server no
    longer has its passage at all."""
    store = FakeObjectStore("reports")
    same_when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.put("a.md", b"alpha", last_modified=same_when)
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)
    store.delete("a.md")
    sync_object_storage(
        store, "", ingester=ingester, checkpoints=checkpoints, deletion_policy="retract"
    )
    assert fake_server.retracted == ["s3://reports/a.md"]

    # Recreated with the exact same size and last_modified a non-versioned
    # bucket's fingerprint falls back to — the false-"unchanged" scenario.
    store.put("a.md", b"alpha", last_modified=same_when)
    report = sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)

    assert report.imported == 1
    assert len(fake_server.imported) == 2


def test_deletion_mirror_policy_self_heals_even_without_a_prior_inventory(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    """`mirror` reconciles against the server's own source list under this
    prefix even on the FIRST pass, when no inventory has ever been written
    yet — a phantom source the connector's own listing never even
    mentions still gets retracted (ADR 0007 §9)."""
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    fake_server.sources = ["s3://reports/a.md", "s3://reports/phantom.md"]
    checkpoints = RecordingCheckpointStore()

    report = sync_object_storage(
        store,
        "",
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        deletion_policy="mirror",
    )

    assert report.retracted == 1
    assert fake_server.retracted == ["s3://reports/phantom.md"]


def test_retract_without_a_sync_client_raises_instead_of_guessing(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    checkpoints = RecordingCheckpointStore()
    # The first pass needs a real sync client to ingest "a.md" at all.
    sync_object_storage(
        store, "", ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )
    store.delete("a.md")

    # The second pass has nothing left to fetch/ingest (the store is now
    # empty) — only `_handle_deletions` runs, and it is what must refuse an
    # async-only ingester rather than silently skipping the retract.
    with pytest.raises(ValueError, match="requires ingester.client"):
        sync_object_storage(
            store,
            "",
            ingester=_ingester(None, async_client),
            checkpoints=checkpoints,
            deletion_policy="retract",
        )


def test_dry_run_touches_nothing(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    checkpoints = RecordingCheckpointStore()

    report = sync_object_storage(
        store,
        "",
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        dry_run=True,
    )

    assert report.parsed == 1  # "would fetch/parse," ADR 0007 §11
    assert store.get_calls == {}
    assert fake_server.imported == []
    assert checkpoints.state == {}  # no checkpoint and no inventory written


def test_dry_run_reports_unchanged_when_the_listing_metadata_already_matches(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha", etag="e1")
    checkpoints = RecordingCheckpointStore()
    ingester = _ingester(sync_client, async_client)

    sync_object_storage(store, "", ingester=ingester, checkpoints=checkpoints)
    report = sync_object_storage(
        store, "", ingester=ingester, checkpoints=checkpoints, dry_run=True
    )

    assert report.unchanged == 1
    assert report.parsed == 0


def test_a_permanent_fetch_failure_fails_only_that_object(
    sync_client: Taguru, async_client: AsyncTaguru, fake_server: FakeServer
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    store.put("b.md", b"beta")
    checkpoints = RecordingCheckpointStore()

    # Only the FIRST object listed ("a.md", alphabetically first) should
    # fail — the real (unpatched) FakeObjectStore.get behavior backs every
    # later call, proving one object's permanent failure doesn't abort the
    # rest of the enumeration.
    real_get = FakeObjectStore.get
    calls = {"n": 0}

    def flaky_get(key: str, *, version_id: str | None = None) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermanentStoreError("access denied")
        return real_get(store, key, version_id=version_id)

    store.get = flaky_get  # type: ignore[method-assign]

    report = sync_object_storage(
        store, "", ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )

    assert report.failed == 1
    assert report.imported == 1


def test_a_vanished_object_is_skipped_not_failed(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    checkpoints = RecordingCheckpointStore()

    def missing(key: str, *, version_id: str | None = None) -> object:
        raise ObjectNotFoundError("gone")

    store.get = missing  # type: ignore[method-assign]

    report = sync_object_storage(
        store, "", ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )

    assert report.skipped == 1
    assert report.failed == 0


def test_a_listing_failure_fails_the_pass_and_never_writes_the_inventory(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    store.fail_list_with = TransientStoreError("timeout")
    checkpoints = RecordingCheckpointStore()

    report = sync_object_storage(
        store, "", ingester=_ingester(sync_client, async_client), checkpoints=checkpoints
    )

    assert report.failed == 1
    assert not any(key.startswith("s3-inventory:") for key in checkpoints.state)


def test_should_stop_mid_pass_leaves_the_inventory_unwritten(
    sync_client: Taguru, async_client: AsyncTaguru
) -> None:
    store = FakeObjectStore("reports")
    store.put("a.md", b"alpha")
    store.put("b.md", b"beta")
    checkpoints = RecordingCheckpointStore()

    report = sync_object_storage(
        store,
        "",
        ingester=_ingester(sync_client, async_client),
        checkpoints=checkpoints,
        should_stop=lambda: True,
    )

    assert report.discovered == 0
    assert not any(key.startswith("s3-inventory:") for key in checkpoints.state)


# ---------------------------------------------------------------------------
# S3ObjectCheckpoint — the cheap "skip the fetch" layer, on its own
# ---------------------------------------------------------------------------


def test_s3_object_checkpoint_round_trips() -> None:
    checkpoint = S3ObjectCheckpoint(RecordingCheckpointStore())
    checkpoint.save("s3://b/k", ("etag", "abc"))
    assert checkpoint.load("s3://b/k") == ("etag", "abc")


def test_s3_object_checkpoint_is_none_when_never_saved() -> None:
    checkpoint = S3ObjectCheckpoint(RecordingCheckpointStore())
    assert checkpoint.load("s3://b/never-seen") is None


def test_s3_object_checkpoint_is_none_on_corrupt_bytes() -> None:
    store = RecordingCheckpointStore(seed={"s3-object:s3://b/k": b"not json {"})
    checkpoint = S3ObjectCheckpoint(store)
    assert checkpoint.load("s3://b/k") is None


def test_s3_object_checkpoint_delete_removes_the_entry() -> None:
    checkpoint = S3ObjectCheckpoint(RecordingCheckpointStore())
    checkpoint.save("s3://b/k", ("etag", "abc"))
    checkpoint.delete("s3://b/k")
    assert checkpoint.load("s3://b/k") is None


# ---------------------------------------------------------------------------
# Credential boundary (ADR 0007 §9): never checkpoint, batch, log, or
# metadata — a real boto3 client, built from fake-but-distinctive env-chain
# credentials, stubbed at the HTTP layer (never a live bucket).
# ---------------------------------------------------------------------------


def test_credentials_never_appear_in_checkpoint_report_or_document_bytes(
    sync_client: Taguru,
    async_client: AsyncTaguru,
    fake_server: FakeServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("boto3")
    import boto3
    from botocore.response import StreamingBody
    from botocore.stub import Stubber

    fake_access_key = "AKIAFAKEFAKEFAKEFAKE"
    fake_secret = "Fake/Secret+Access+Key+Value+Never+Persisted"
    fake_session_token = "fake-session-token-never-persisted"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", fake_access_key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", fake_secret)
    monkeypatch.setenv("AWS_SESSION_TOKEN", fake_session_token)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # Built via `boto3.Session()` with NO explicit credential arguments —
    # exactly the standard chain S3ObjectStore itself uses — so the fake
    # secrets above flow through the same path a real deployment's env
    # vars would, not a shortcut that bypasses credential resolution.
    client = boto3.client("s3")
    stubber = Stubber(client)
    stubber.add_response("get_bucket_versioning", {}, {"Bucket": "reports"})
    stubber.add_response(
        "list_objects_v2",
        {
            "Contents": [{"Key": "a.md", "Size": 5, "ETag": '"abc"'}],
            "IsTruncated": False,
        },
        {"Bucket": "reports", "Prefix": ""},
    )
    stubber.add_response(
        "get_object",
        {
            "Body": StreamingBody(io.BytesIO(b"hello world"), 11),
            "ContentType": "text/markdown",
        },
        {"Bucket": "reports", "Key": "a.md"},
    )
    stubber.add_response("get_object_tagging", {"TagSet": []}, {"Bucket": "reports", "Key": "a.md"})

    from taguru_langchain.ingest_connectors import S3ObjectStore

    store = S3ObjectStore("reports", client=client)
    checkpoints = RecordingCheckpointStore()

    with stubber:
        report = sync_object_storage(
            store,
            "",
            ingester=_ingester(sync_client, async_client),
            checkpoints=checkpoints,
        )

    assert report.imported == 1
    haystacks: list[str] = [
        json.dumps(str(checkpoints.state)),
        json.dumps([dataclasses.asdict(event) for event in report.events], default=str),
        *fake_server.imported,
    ]
    for secret in (fake_access_key, fake_secret, fake_session_token):
        for haystack in haystacks:
            assert secret not in haystack
