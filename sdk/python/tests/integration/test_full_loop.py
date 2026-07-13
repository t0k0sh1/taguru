"""The full ingest → retrieval loop against the real server binary."""

from __future__ import annotations

import pytest

from taguru import (
    ConflictError,
    EmbeddingUnavailableError,
    NotFoundError,
    Taguru,
    ValidationError,
    citation_key,
)

AOMINE_DOC = """青嶺酒造は1907年創業の架空の酒蔵である。代表銘柄は「青嶺」。

杜氏は高瀬である。高瀬は寒仕込みを重視する。

青嶺酒造は大量生産を行わない。"""


def seed(client: Taguru, name: str) -> None:
    client.contexts.create(name, description="青嶺酒造という架空の酒蔵の知識")
    ctx = client.context(name)
    ctx.add_associations(
        [
            {
                "subject": "青嶺酒造",
                "label": "創業年",
                "object": "1907年",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 0,
            },
            {
                "subject": "青嶺酒造",
                "label": "代表銘柄",
                "object": "青嶺",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 0,
            },
            {
                "subject": "青嶺酒造",
                "label": "杜氏",
                "object": "高瀬",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 1,
            },
            {
                "subject": "高瀬",
                "label": "重視する",
                "object": "寒仕込み",
                "weight": 1.0,
                "source": "docs/aomine.md",
                "paragraph": 1,
            },
            {
                "subject": "青嶺酒造",
                "label": "行う",
                "object": "大量生産",
                "weight": -1.0,
                "source": "docs/aomine.md",
                "paragraph": 2,
            },
        ]
    )
    ctx.store_passages({"docs/aomine.md": AOMINE_DOC})


def test_context_lifecycle(client: Taguru, fresh_name: str) -> None:
    assert not client.contexts.exists(fresh_name)
    assert client.contexts.create(fresh_name, description="d", pinned=False)
    with pytest.raises(ConflictError) as conflict:
        client.contexts.create(fresh_name)
    assert conflict.value.code == "already_exists"
    entry = client.contexts.get(fresh_name)
    assert entry.name == fresh_name
    assert entry.description == "d"

    meta = client.contexts.update(fresh_name, description="d2", dice_floor=0.25)
    assert meta.description == "d2"
    assert meta.dice_floor == 0.25

    names = [e.name for e in client.contexts.iter(limit=2)]
    assert fresh_name in names

    renamed = f"{fresh_name}-renamed"
    assert client.contexts.rename(fresh_name, renamed)
    with pytest.raises(NotFoundError) as renamed_away:
        client.contexts.get(fresh_name)
    assert renamed_away.value.code == "no_context"
    entry = client.contexts.get(renamed)
    assert entry.description == "d2"
    assert entry.dice_floor == 0.25

    assert client.contexts.delete(renamed)
    with pytest.raises(NotFoundError) as missing:
        client.contexts.get(renamed)
    assert missing.value.code == "no_context"


def test_associations_accumulate_weight_and_validate(client: Taguru, fresh_name: str) -> None:
    client.contexts.create(fresh_name)
    ctx = client.context(fresh_name)
    op = {"subject": "s", "label": "l", "object": "o", "weight": 1.0, "source": "a"}
    assert ctx.add_associations([op]) == 1
    assert ctx.add_associations([{**op, "source": "b"}]) == 1

    page = ctx.query(subject="s", label="l")
    assert page.total == 1
    match = page.matches[0]
    # Two 1.0 assertions: averaged weight 1.0, count 2, two attributions.
    assert match.weight == 1.0
    assert match.count == 2
    assert {a.source for a in match.attributions} == {"a", "b"}

    with pytest.raises(ValidationError) as bad_weight:
        ctx.add_associations([{**op, "weight": float("1e300")}])
    assert bad_weight.value.code == "invalid_argument"
    with pytest.raises(ValidationError):
        ctx.add_associations([{**op, "subject": ""}])
    with pytest.raises(ValidationError) as too_many:
        ctx.add_associations([{**op, "subject": f"s{index}"} for index in range(10_001)])
    assert too_many.value.code == "over_limit"
    client.contexts.delete(fresh_name)


def test_graph_reads(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)

    recall = ctx.recall("青嶺酒造")
    assert recall.total >= 4

    both = ctx.query(subject="青嶺酒造", label=["杜氏", "代表銘柄"])
    assert {m.object for m in both.matches} == {"高瀬", "青嶺"}

    negated = ctx.query(subject="青嶺酒造", label="行う")
    assert negated.matches[0].weight == -1.0

    outline = ctx.describe("青嶺酒造")
    assert outline is not None
    assert any(usage.label == "杜氏" for usage in outline.as_subject)
    assert ctx.describe("存在しない概念") is None

    walked = ctx.explore(["青嶺酒造"], max_depth=2)
    assert any(rec.association.object == "寒仕込み" for rec in walked.matches)
    two_hop = next(rec for rec in walked.matches if rec.association.object == "寒仕込み")
    assert two_hop.distance == 2
    assert two_hop.path[0] == "青嶺酒造"

    activated = ctx.activate(["青嶺酒造"], limit=10)
    assert activated.matches
    strengths = [a.strength for a in activated.matches]
    assert strengths == sorted(strengths, reverse=True)

    audit = ctx.unreachable_from(["青嶺酒造"])
    assert audit.total == 0

    labels = ctx.list_labels()
    assert "杜氏" in labels.labels
    assert list(ctx.iter_labels()) == labels.labels
    client.contexts.delete(fresh_name)


def test_resolve_tiers_and_floors(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)

    exact = ctx.resolve("青嶺酒造")
    assert exact[0].name == "青嶺酒造"
    assert exact[0].kind == "exact"
    assert exact[0].tier == "lexical"
    assert exact[0].score == 1.0

    # Full-width/typo-ish entry still lands via the normalized bigram index.
    fuzzy = ctx.resolve("青嶺酒蔵")
    assert any(c.name == "青嶺酒造" for c in fuzzy)
    assert all(c.kind in {"exact", "alias", "containment", "fuzzy"} for c in fuzzy)

    # A lower dice_floor widens fuzzy matching for one call.
    assert len(ctx.resolve("青嶺", dice_floor=0.1)) >= len(ctx.resolve("青嶺", dice_floor=0.9))

    label = ctx.resolve_label("杜氏")
    assert label[0].name == "杜氏"
    client.contexts.delete(fresh_name)


def test_aliases_roundtrip_and_reregistration_semantics(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)

    assert ctx.add_aliases(concepts={"Aomine Brewery": "青嶺酒造"}, labels={"brewer": "杜氏"}) == 2
    page = ctx.get_aliases()
    assert page.concepts == {"Aomine Brewery": "青嶺酒造"}
    assert page.labels == {"brewer": "杜氏"}
    assert [e.namespace for e in ctx.iter_aliases()] == ["concept", "label"]

    resolved = ctx.resolve("Aomine Brewery")
    assert resolved[0].name == "青嶺酒造"  # aliases are entry-only
    assert resolved[0].kind == "alias"

    # Re-registering the identical pair succeeds as a no-op — this pins the
    # semantics that justify classifying add_aliases as retry-SAFE.
    assert ctx.add_aliases(concepts={"Aomine Brewery": "青嶺酒造"}) == 1
    assert ctx.get_aliases().concepts == {"Aomine Brewery": "青嶺酒造"}

    with pytest.raises(ConflictError):
        # An alias can never shadow an existing concept name.
        ctx.add_aliases(concepts={"高瀬": "青嶺酒造"})

    assert ctx.remove_aliases(concepts=["Aomine Brewery"], labels=["brewer"]) == 2
    assert ctx.get_aliases().concepts == {}
    client.contexts.delete(fresh_name)


def test_sources_and_citations(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)

    stored = ctx.store_passages(
        {"docs/extra.md": "第一段落。\n\n第二段落。"},
        questions={
            "docs/extra.md": [
                {"paragraph": 1, "question": "二番目は?"},
                {"paragraph": 9, "question": "範囲外"},
            ]
        },
        sections={
            "docs/extra.md": [
                {"paragraph": 0, "section": "冒頭"},
                {"paragraph": 8, "section": "範囲外"},
            ]
        },
    )
    assert stored.stored == 1
    assert stored.questions_stored == 1
    assert stored.questions_dropped == 1
    assert stored.sections_stored == 1
    assert stored.sections_dropped == 1

    sources = ctx.list_sources()
    assert set(sources.sources) == {"docs/aomine.md", "docs/extra.md"}
    assert list(ctx.iter_sources()) == sources.sources

    lookup = ctx.lookup_passages(["docs/aomine.md", "missing.md"])
    assert lookup.passages["docs/aomine.md"] == AOMINE_DOC
    assert lookup.missing == ["missing.md"]

    hits = ctx.search_passages("杜氏は高瀬である", limit=3)
    assert hits
    top = hits[0]
    assert top.source == "docs/aomine.md"
    assert "高瀬" in top.text
    assert top.lanes.bm25 is not None  # lexical lane evidence
    assert top.lanes.vector is None  # no embedding provider in tests

    cited = ctx.cite_passage("docs/aomine.md", 1)
    assert "杜氏は高瀬" in cited.text
    assert cited.source == "docs/aomine.md"
    with pytest.raises(NotFoundError):
        ctx.cite_passage("missing.md", 0)
    with pytest.raises(NotFoundError):
        ctx.cite_passage("docs/aomine.md", 99)

    retracted = ctx.retract_source("docs/extra.md")
    assert retracted.passage_removed
    assert "docs/extra.md" not in ctx.list_sources().sources
    client.contexts.delete(fresh_name)


def test_embeddings_refresh_501_without_provider(client: Taguru, fresh_name: str) -> None:
    client.contexts.create(fresh_name)
    with pytest.raises(EmbeddingUnavailableError) as excinfo:
        client.context(fresh_name).refresh_embeddings()
    assert excinfo.value.reason == "not_configured"
    assert excinfo.value.code == "embeddings_unconfigured"
    client.contexts.delete(fresh_name)


def test_vocabulary_audit_surfaces_lexical_twins(client: Taguru, fresh_name: str) -> None:
    client.contexts.create(fresh_name)
    ctx = client.context(fresh_name)
    ctx.add_associations(
        [
            {"subject": "株式会社青嶺", "label": "kind", "object": "会社", "weight": 1.0},
            {"subject": "青嶺株式会社", "label": "kind", "object": "会社", "weight": 1.0},
        ]
    )
    audit = ctx.audit_vocabulary(dice_floor=0.4)
    assert any(
        {pair.a, pair.b} == {"株式会社青嶺", "青嶺株式会社"} for pair in audit.lexical_concepts
    )
    client.contexts.delete(fresh_name)


def test_compact_reports_shed_bytes(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)
    ctx.retract_source("docs/aomine.md")
    outcome = ctx.compact()
    assert outcome.bytes_after <= outcome.bytes_before
    client.contexts.delete(fresh_name)


def test_flush_names_dirty_contexts(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    flushed = client.flush()
    assert fresh_name in flushed
    client.contexts.delete(fresh_name)


def test_retrieve_end_to_end(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)

    result = ctx.retrieve("青嶺酒造", text_fallback_query="杜氏は高瀬である")
    assert result.resolved["青嶺酒造"][0].kind == "exact"
    assert result.outline["青嶺酒造"] is not None
    assert any(a.object == "高瀬" for a in result.associations)
    # Citations came back verbatim, keyed by (source, paragraph).
    assert result.citations
    key = citation_key("docs/aomine.md", 1)
    if key in result.citations:
        assert "高瀬" in result.citations[key].text
    # Graph answered → text lane stayed off by default.
    assert result.passage_hits == []

    # qa_recall-style question: who is the 杜氏? Answerable from the result.
    toji = [a for a in result.associations if a.label == "杜氏"]
    assert toji and toji[0].object == "高瀬"
    client.contexts.delete(fresh_name)


async def test_async_client_full_smoke(server, fresh_name: str) -> None:
    from taguru import AsyncTaguru

    from .conftest import ADMIN_TOKEN

    async with AsyncTaguru(server.base_url, ADMIN_TOKEN) as aclient:
        await aclient.wait_until_ready(timeout=30)
        await aclient.contexts.create(fresh_name, description="async smoke")
        ctx = aclient.context(fresh_name)
        await ctx.add_associations(
            [{"subject": "s", "label": "l", "object": "o", "weight": 1.0, "source": "a"}]
        )
        await ctx.store_passages({"a": "本文。"})
        page = await ctx.recall("s")
        assert page.total == 1
        names = [e.name async for e in aclient.contexts.iter()]
        assert fresh_name in names
        exported = await ctx.export()
        assert '"taguru_batch"' in exported
        await aclient.contexts.delete(fresh_name)


def test_retract_association_withdraws_one_edge(client: Taguru, fresh_name: str) -> None:
    seed(client, fresh_name)
    ctx = client.context(fresh_name)

    outcome = ctx.retract_association("青嶺酒造", "代表銘柄", "青嶺")
    assert outcome.retracted
    assert outcome.attributions_removed == 1

    # Found-nothing honesty on the second call; the document's other
    # facts are untouched.
    again = ctx.retract_association("青嶺酒造", "代表銘柄", "青嶺")
    assert not again.retracted
    assert again.attributions_removed == 0
    toji = ctx.query(subject="青嶺酒造", label="杜氏")
    assert toji.matches[0].object == "高瀬"
    dead = ctx.query(subject="青嶺酒造", label="代表銘柄")
    assert dead.matches[0].weight == 0.0
    assert dead.matches[0].count == 0
    client.contexts.delete(fresh_name)
