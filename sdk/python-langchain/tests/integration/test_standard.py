"""LangChain's standard retriever conformance suite, run against the real server."""

from __future__ import annotations

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests

from taguru_langchain import TaguruRetriever


class TestTaguruRetrieverStandard(RetrieversIntegrationTests):
    """Inherited: k constructor param, k invoke kwarg, invoke/ainvoke return
    Documents. The connection comes from TAGURU_URL/TAGURU_API_TOKEN, which
    the server fixture exports — the same zero-config path applications use."""

    @pytest.fixture(autouse=True)
    def _seeded_id(self, seeded: str) -> None:
        # The retriever's `context` field takes the id (#964); the
        # session-scoped fixture returns the seeded context's.
        self._context_id = seeded

    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return TaguruRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, object]:
        return {"context": self._context_id}

    @property
    def retriever_query_example(self) -> str:
        return "青嶺酒造"
