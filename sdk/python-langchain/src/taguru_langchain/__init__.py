"""LangChain integration for the Taguru long-term semantic memory server.

Two entry points:

- :class:`TaguruRetriever` — the documented retrieval loop (graph lane +
  text lane, RRF-merged) as a LangChain ``BaseRetriever``.
- :class:`TaguruIngester` — the LangChain twin of ``taguru extract``: a chat
  model decomposes Documents into associations under the protocol's ingest
  discipline, applied via ``POST /import`` (per-source replace, idempotent).

Deliberately NOT provided: a VectorStore facade (Taguru's retrieval is
structural-first; forcing it behind ``similarity_search`` would misrepresent
resolve/activate semantics), a Memory class (deprecated upstream in favor of
LangGraph state), and agent Tools (the MCP bridge ``taguru-mcp`` already
serves the identical tool vocabulary — pair it with ``langchain-mcp-adapters``).
"""

from __future__ import annotations

from .ingest import IngestOutcome, TaguruIngester
from .retrievers import TaguruRetriever

__version__ = "0.4.0"

__all__ = ["TaguruRetriever", "TaguruIngester", "IngestOutcome", "__version__"]
