"""Groups CRUD and cross-context search against the real server binary."""

from __future__ import annotations

import json

import pytest

from taguru import ConflictError, NotFoundError, PermissionDeniedError, Taguru, ValidationError


def seeded_pair(client: Taguru, base: str) -> tuple[str, str]:
    """Two contexts holding one distinct fact (graph + passage) each."""
    sake, tea = f"{base}-sake", f"{base}-tea"
    client.contexts.create(sake, description="酒蔵の知識")
    client.contexts.create(tea, description="茶園の知識")
    client.context(sake).add_associations(
        [
            {
                "subject": "青嶺酒造",
                "label": "代表銘柄",
                "object": "青嶺",
                "weight": 1.0,
                "source": "sake.md",
                "paragraph": 0,
            }
        ]
    )
    client.context(tea).add_associations(
        [
            {
                "subject": "青嶺茶園",
                "label": "代表銘柄",
                "object": "露霜",
                "weight": 1.0,
                "source": "tea.md",
                "paragraph": 0,
            }
        ]
    )
    client.context(sake).store_passages({"sake.md": "青嶺酒造の代表銘柄は「青嶺」である。"})
    client.context(tea).store_passages({"tea.md": "青嶺茶園の代表銘柄は「露霜」である。"})
    return sake, tea


def test_group_lifecycle(client: Taguru, fresh_name: str) -> None:
    sake, tea = seeded_pair(client, fresh_name)
    group, child = f"{fresh_name}-g", f"{fresh_name}-child"

    assert not client.groups.exists(group)
    assert client.groups.create(group, description="蔵元一式", contexts=[sake])
    with pytest.raises(ConflictError) as conflict:
        client.groups.create(group)
    assert conflict.value.code == "already_exists"

    entry = client.groups.get(group)
    assert entry.name == group
    assert entry.description == "蔵元一式"
    assert entry.contexts == [sake]
    assert entry.groups == []

    # Deltas: removals are idempotent no-ops, additions demand existence.
    entry = client.groups.update(group, add_contexts=[tea], remove_contexts=["never-a-member"])
    assert entry.contexts == sorted([sake, tea])
    with pytest.raises(NotFoundError) as missing_member:
        client.groups.update(group, add_contexts=[f"{fresh_name}-missing"])
    assert missing_member.value.code == "no_context"

    # Nesting: a child group rides the row's `groups` list.
    assert client.groups.create(child, contexts=[tea])
    entry = client.groups.update(group, add_groups=[child])
    assert entry.groups == [child]

    names = [row.name for row in client.groups.iter(limit=2)]
    assert group in names and child in names

    # Deleting the bundling leaves members (and the child group) alone.
    assert client.groups.delete(group)
    with pytest.raises(NotFoundError) as gone:
        client.groups.get(group)
    assert gone.value.code == "no_group"
    assert client.groups.exists(child)
    assert client.contexts.exists(sake)

    client.groups.delete(child)
    client.contexts.delete(sake)
    client.contexts.delete(tea)


def test_group_writes_need_the_write_role(
    client: Taguru, reader_client: Taguru, fresh_name: str
) -> None:
    with pytest.raises(PermissionDeniedError) as denied:
        reader_client.groups.create(f"{fresh_name}-g")
    assert denied.value.code == "forbidden"


def test_cross_context_search_tags_every_match(client: Taguru, fresh_name: str) -> None:
    sake, tea = seeded_pair(client, fresh_name)
    group = f"{fresh_name}-g"
    client.groups.create(group, contexts=[sake, tea])

    # recall: named contexts, every match tagged with its origin.
    page = client.recall("代表銘柄", contexts=[sake, tea])
    assert page.total == 2
    assert {match.context for match in page.matches} == {sake, tea}
    assert {match.object for match in page.matches} == {"青嶺", "露霜"}

    # query: a group resolves to every context it reaches; overlaps with
    # directly named contexts dedupe silently.
    page = client.query(label="代表銘柄", groups=[group], contexts=[sake])
    assert page.total == 2
    assert {match.context for match in page.matches} == {sake, tea}

    # search_passages: rank-interleaved, hits tagged; score is per-context.
    hits = client.search_passages("代表銘柄は青嶺", contexts=[sake, tea], limit=4)
    assert {hit.context for hit in hits} == {sake, tea}
    assert all(hit.text for hit in hits)

    # An empty target list is refused, an unknown group answers no_group.
    with pytest.raises(ValidationError) as empty:
        client.recall("青嶺", contexts=[])
    assert empty.value.code == "invalid_argument"
    with pytest.raises(NotFoundError) as missing:
        client.recall("青嶺", groups=[f"{fresh_name}-missing"])
    assert missing.value.code == "no_group"

    client.groups.delete(group)
    client.contexts.delete(sake)
    client.contexts.delete(tea)


def test_group_export_import_round_trip(client: Taguru, fresh_name: str) -> None:
    sake, tea = seeded_pair(client, fresh_name)
    group = f"{fresh_name}-g"
    client.groups.create(group, description="蔵元一式", contexts=[sake, tea])

    line = client.groups.export(group)
    record = json.loads(line)
    assert isinstance(record["taguru_group"], int)
    assert record["name"] == group
    assert record["contexts"] == sorted([sake, tea])

    # The record is the group's complete truth: import restores it whole.
    client.groups.delete(group)
    assert not client.groups.exists(group)
    client.import_batches(line)
    assert client.groups.get(group).contexts == sorted([sake, tea])

    client.groups.delete(group)
    client.contexts.delete(sake)
    client.contexts.delete(tea)
