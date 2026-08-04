"""``get_schema`` decodes into ``SchemaDocument``/``TypeDef``/``RelationDef``
(ADR 0009 §5) — the same shape ``PUT /contexts/{name}/schema`` accepts and
``taguru extract --schema``/both LangChain ingesters consume."""

from __future__ import annotations

from typing import Any

from taguru import NotFoundError, RelationDef, SchemaDocument, TypeDef

from .conftest import err_response, ok_response, sync_client

SCHEMA_DOCUMENT: dict[str, Any] = {
    "schema": 1,
    "mode": "strict",
    "closed_labels": False,
    "types": {
        "Brewery": {"is_a": ["Organization"]},
        "Organization": {"is_a": []},
        "Person": {"is_a": []},
    },
    "relations": {
        "杜氏": {"domain": ["Brewery"], "range": ["Person"]},
    },
}


def test_get_schema_decodes_into_reexported_models() -> None:
    client = sync_client(lambda _req: ok_response(SCHEMA_DOCUMENT))
    document = client.context("aomine").get_schema()

    assert isinstance(document, SchemaDocument)
    assert document.schema == 1
    assert document.mode == "strict"
    assert document.closed_labels is False

    assert set(document.types) == {"Brewery", "Organization", "Person"}
    brewery = document.types["Brewery"]
    assert isinstance(brewery, TypeDef)
    assert brewery.is_a == ["Organization"]
    assert document.types["Organization"].is_a == []

    relation = document.relations["杜氏"]
    assert isinstance(relation, RelationDef)
    assert relation.domain == ["Brewery"]
    assert relation.range == ["Person"]


def test_get_schema_raises_not_found_when_the_context_has_no_schema() -> None:
    client = sync_client(
        lambda _req: err_response(404, "context 'aomine' has no schema document", code="no_schema")
    )
    try:
        client.context("aomine").get_schema()
    except NotFoundError:
        pass
    else:
        raise AssertionError("expected NotFoundError")
