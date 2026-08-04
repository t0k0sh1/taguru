"""The schema client surface: ``get_schema`` decodes into
``SchemaDocument``/``TypeDef``/``RelationDef`` (ADR 0009 §5) — the same
shape ``put_schema`` sends and ``taguru extract --schema``/both LangChain
ingesters consume — and ``audit_schema``/``validate_schema`` decode the
shared ``SchemaAudit`` shape (§10)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from taguru import NotFoundError, RelationDef, SchemaAudit, SchemaDocument, TypeDef

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


SCHEMA_AUDIT: dict[str, Any] = {
    "total": 1,
    "violations": [
        {
            "association": {
                "subject": "青嶺酒造",
                "label": "杜氏",
                "object": "広島",
                "weight": 1.0,
                "count": 1,
                "attributions": [],
            },
            "issues": [
                {
                    "path": "edge(青嶺酒造, 杜氏, 広島)",
                    "kind": "range",
                    "expected": "one of [Person]",
                    "actual": "Prefecture",
                }
            ],
        }
    ],
    "untyped_concepts": {"total": 2, "names": ["広島", "青嶺"]},
    "undeclared_types": {"total": 0, "names": []},
    "unknown_labels": {"total": 0, "names": []},
    "reserved_alias_conflicts": {"total": 1, "aliases": {"種類": "schema:type"}},
}


def test_put_schema_sends_the_document_and_decodes_the_installed_one() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return ok_response(SCHEMA_DOCUMENT)

    client = sync_client(handler)
    # A plain mapping and the decoded dataclass must serialize identically.
    installed = client.context("aomine").put_schema(SCHEMA_DOCUMENT)
    assert seen["method"] == "PUT"
    assert seen["path"] == "/contexts/aomine/schema"
    assert seen["body"] == SCHEMA_DOCUMENT
    assert isinstance(installed, SchemaDocument)

    reinstalled = client.context("aomine").put_schema(installed)
    assert seen["body"] == SCHEMA_DOCUMENT
    assert reinstalled.mode == "strict"


def test_audit_and_validate_schema_decode_the_shared_audit_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return ok_response(SCHEMA_AUDIT)

    client = sync_client(handler)
    audit = client.context("aomine").audit_schema(limit=10)
    assert seen["path"] == "/contexts/aomine/schema/audit"
    assert seen["body"] == {"limit": 10}
    assert isinstance(audit, SchemaAudit)
    assert audit.total == 1
    violation = audit.violations[0]
    assert violation.association.object == "広島"
    assert violation.issues[0].kind == "range"
    assert audit.untyped_concepts.names == ["広島", "青嶺"]
    assert audit.reserved_alias_conflicts.aliases == {"種類": "schema:type"}

    validated = client.context("aomine").validate_schema(SCHEMA_DOCUMENT, limit=10)
    assert seen["path"] == "/contexts/aomine/schema/validate"
    assert seen["body"] == {"document": SCHEMA_DOCUMENT, "limit": 10}
    assert validated.total == 1
