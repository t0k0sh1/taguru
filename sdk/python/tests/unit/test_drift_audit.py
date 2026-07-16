"""``audit_drift`` decodes into ``DriftAudit``/``UnsourcedEdge`` — both must
be importable from the top-level package, not just from ``_models``."""

from __future__ import annotations

from typing import Any

from taguru import Association, DriftAudit, UnsourcedEdge

from .conftest import ok_response, sync_client

DRIFT_AUDIT: dict[str, Any] = {
    "total": 1,
    "unsourced": [
        {
            "unsourced_weight": 1.0,
            "unsourced_count": 1,
            "association": {
                "subject": "青嶺酒造",
                "label": "kind",
                "object": "会社",
                "weight": 1.0,
                "count": 1,
                "attributions": [],
            },
        }
    ],
    "dead_concept_aliases": {"タカセ": "高瀬"},
    "dead_label_aliases": {},
    "twins": None,
}


def test_audit_drift_decodes_into_reexported_models() -> None:
    client = sync_client(lambda _req: ok_response(DRIFT_AUDIT))
    audit = client.context("aomine").audit_drift()

    assert isinstance(audit, DriftAudit)
    assert audit.total == 1
    assert audit.dead_concept_aliases == {"タカセ": "高瀬"}
    assert audit.twins is None

    assert len(audit.unsourced) == 1
    edge = audit.unsourced[0]
    assert isinstance(edge, UnsourcedEdge)
    assert edge.unsourced_weight == 1.0
    assert isinstance(edge.association, Association)
    assert edge.association.subject == "青嶺酒造"
