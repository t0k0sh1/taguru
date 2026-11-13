"""Groups CRUD and cross-context search against the real server binary."""

from __future__ import annotations

import json
import uuid

import pytest

from taguru import (
    ConflictError,
    GroupImportOutcome,
    NotFoundError,
    PermissionDeniedError,
    Taguru,
    ValidationError,
)

GHOST_ID = "ead6ef03-d61e-460c-933d-6d450c50a1e5"


def seeded_pair(client: Taguru, base: str) -> tuple[str, str, str, str]:
    """Two contexts holding one distinct fact (graph + passage) each.

    Returns ``(sake_name, tea_name, sake_id, tea_id)`` — group members
    and cross-search bodies take the ids (#965), as does every
    ``/contexts/{id}/…`` call (#964); the names are display labels."""
    sake, tea = f"{base}-sake", f"{base}-tea"
    sake_id = client.contexts.create(sake, description="酒蔵の知識").id
    tea_id = client.contexts.create(tea, description="茶園の知識").id
    client.context(sake_id).add_associations(
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
    client.context(tea_id).add_associations(
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
    client.context(sake_id).store_passages({"sake.md": "青嶺酒造の代表銘柄は「青嶺」である。"})
    client.context(tea_id).store_passages({"tea.md": "青嶺茶園の代表銘柄は「露霜」である。"})
    return sake, tea, sake_id, tea_id


def group_id_named(client: Taguru, name: str) -> str:
    """The id of the one ``group`` called ``name`` — the directory walk a caller
    holding only a name makes (names are not unique, #995)."""
    ids = [row.id for row in client.groups.iter() if row.name == name]
    assert len(ids) == 1, ids
    return ids[0]


def test_group_lifecycle(client: Taguru, fresh_name: str) -> None:
    sake, tea, sake_id, tea_id = seeded_pair(client, fresh_name)
    group, child = f"{fresh_name}-g", f"{fresh_name}-child"

    created = client.groups.create(group, description="蔵元一式", context_ids=[sake_id])
    group_id = created.id
    assert created.name == group
    assert created.context_ids == [sake_id]
    assert client.groups.exists(group_id)
    assert group_id_named(client, group) == group_id
    # A name is a label: creating it again makes a second group.
    twin = client.groups.create(group)
    assert twin.id != group_id
    assert client.groups.delete(twin.id)
    # A caller's own id: the same record again answers the standing row,
    # a different one is a conflict.
    own_id = str(uuid.uuid4())
    pinned = client.groups.create(f"{group}-pinned", group_id=own_id)
    assert pinned.id == own_id
    assert client.groups.create(f"{group}-pinned", group_id=own_id).id == own_id
    with pytest.raises(ConflictError):
        client.groups.create(f"{group}-other", group_id=own_id)
    assert client.groups.delete(own_id)
    # A name where the id belongs is a 400, never a lookup.
    with pytest.raises(ValidationError) as by_name:
        client.groups.get(group)
    assert by_name.value.code == "invalid_argument"

    entry = client.groups.get(group_id)
    assert entry.id == group_id
    assert entry.name == group
    assert entry.description == "蔵元一式"
    assert entry.context_ids == [sake_id]
    assert entry.groups == []

    # Deltas: removals are idempotent no-ops, additions demand existence.
    entry = client.groups.update(group_id, add_context_ids=[tea_id], remove_context_ids=[GHOST_ID])
    assert entry.context_ids == sorted([sake_id, tea_id])
    with pytest.raises(NotFoundError) as missing_member:
        client.groups.update(group_id, add_context_ids=[GHOST_ID])
    assert missing_member.value.code == "no_context"

    # Nesting: a child group rides the row's `groups` list (by name).
    child_id = client.groups.create(child, context_ids=[tea_id]).id
    entry = client.groups.update(group_id, add_groups=[child])
    assert entry.groups == [child]

    ids = [row.id for row in client.groups.iter(limit=2)]
    assert group_id in ids and child_id in ids

    # Rename: a display-name change — the id and the membership stay.
    renamed_group = f"{fresh_name}-g-renamed"
    assert client.groups.rename(group_id, renamed_group)
    entry = client.groups.get(group_id)
    assert entry.name == renamed_group
    assert entry.context_ids == sorted([sake_id, tea_id])
    assert entry.groups == [child]

    # Deleting the bundling leaves members (and the child group) alone.
    assert client.groups.delete(group_id)
    with pytest.raises(NotFoundError) as gone:
        client.groups.get(group_id)
    assert gone.value.code == "no_group"
    assert client.groups.exists(child_id)
    assert client.contexts.exists(sake_id)

    client.groups.delete(child_id)
    client.contexts.delete(sake_id)
    client.contexts.delete(tea_id)


def test_group_writes_need_the_write_role(
    client: Taguru, reader_client: Taguru, fresh_name: str
) -> None:
    with pytest.raises(PermissionDeniedError) as denied:
        reader_client.groups.create(f"{fresh_name}-g")
    assert denied.value.code == "forbidden"


def test_cross_context_search_tags_every_match(client: Taguru, fresh_name: str) -> None:
    sake, tea, sake_id, tea_id = seeded_pair(client, fresh_name)
    group = f"{fresh_name}-g"
    group_id = client.groups.create(group, context_ids=[sake_id, tea_id]).id

    # recall: named contexts, every match tagged with its origin.
    page = client.recall("代表銘柄", context_ids=[sake_id, tea_id])
    assert page.total == 2
    assert {match.context_id for match in page.matches} == {sake_id, tea_id}
    assert {match.context_name for match in page.matches} == {sake, tea}
    assert {match.object for match in page.matches} == {"青嶺", "露霜"}

    # query: a group resolves to every context it reaches; overlaps with
    # directly named contexts dedupe silently.
    page = client.query(label="代表銘柄", groups=[group], context_ids=[sake_id])
    assert page.total == 2
    assert {match.context_id for match in page.matches} == {sake_id, tea_id}
    assert {match.context_name for match in page.matches} == {sake, tea}

    # search_passages: rank-interleaved, hits tagged; score is per-context.
    page = client.search_passages("代表銘柄は青嶺", context_ids=[sake_id, tea_id], limit=4)
    assert {hit.context_id for hit in page.hits} == {sake_id, tea_id}
    assert all(hit.text for hit in page.hits)
    # The plan lists both targets in effective order (#151).
    assert [entry.context_id for entry in page.plan.contexts] == [sake_id, tea_id]

    # An empty target list is refused, an unknown group answers no_group.
    with pytest.raises(ValidationError) as empty:
        client.recall("青嶺", context_ids=[])
    assert empty.value.code == "invalid_argument"
    with pytest.raises(NotFoundError) as missing:
        client.recall("青嶺", groups=[f"{fresh_name}-missing"])
    assert missing.value.code == "no_group"

    client.groups.delete(group_id)
    client.contexts.delete(sake_id)
    client.contexts.delete(tea_id)


def test_group_export_import_round_trip(client: Taguru, fresh_name: str) -> None:
    sake, tea, sake_id, tea_id = seeded_pair(client, fresh_name)
    group = f"{fresh_name}-g"
    group_id = client.groups.create(group, description="蔵元一式", context_ids=[sake_id, tea_id]).id

    line = client.groups.export(group_id)
    record = json.loads(line)
    assert record["type"] == "group"
    assert record["version"] == "2026-10-01"
    assert record["id"] == group
    assert record["context_ids"] == sorted([sake_id, tea_id])

    # The record is the group's complete truth: import restores it whole.
    client.groups.delete(group_id)
    assert not client.groups.exists(group_id)
    result = client.import_batches(line)
    assert result.groups == [
        GroupImportOutcome(name=group, outcome="created", contexts=2, groups=0)
    ]
    restored = group_id_named(client, group)
    assert client.groups.get(restored).context_ids == sorted([sake_id, tea_id])

    client.groups.delete(restored)
    client.contexts.delete(sake_id)
    client.contexts.delete(tea_id)
