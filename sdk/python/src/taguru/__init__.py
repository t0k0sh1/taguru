"""Python client SDK for the Taguru long-term semantic memory server.

Quick start::

    from taguru import Taguru

    client = Taguru()  # TAGURU_URL / TAGURU_API_TOKEN, else localhost:8248
    ctx = client.context("sake")
    hits = ctx.search_passages("酒蔵の創業年", limit=5)

``AsyncTaguru`` is the identical async surface. The behavioral contract is
the server's own protocol document: ``client.protocol()`` (GET /protocol).
"""

from __future__ import annotations

from ._async.client import AsyncContext, AsyncContexts, AsyncGroups, AsyncTaguru
from ._errors import (
    AuthenticationError,
    ConflictError,
    EmbeddingUnavailableError,
    NotFoundError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    ServiceUnavailableError,
    StorageFullError,
    TaguruError,
    TransportError,
    UnexpectedStatusError,
    ValidationError,
)
from ._models import (
    Activation,
    ActivationPage,
    AliasEntry,
    AliasPage,
    Association,
    Attribution,
    BatchApplyResult,
    Citation,
    CompactOutcome,
    ConceptDescription,
    ContextMeta,
    ContextPage,
    ContextStats,
    ContextUsage,
    CrossAssociation,
    CrossMatchPage,
    CrossPassageHit,
    DirectoryEntry,
    ExplorePage,
    GroupEntry,
    GroupPage,
    ImportOutcome,
    LabelPage,
    LabelUsage,
    LaneEvidence,
    MatchPage,
    PassageHit,
    PassageLanes,
    PassageLookup,
    Recollection,
    RefreshBreakdown,
    RefreshOutcome,
    RetractOutcome,
    RetrievalResult,
    SourcePage,
    StoredPassages,
    TieredResolution,
    TwinPair,
    VocabularyAudit,
)
from ._shared import citation_key
from ._sync.client import Context, Contexts, Groups, Taguru
from ._types import AssocOp, QuestionSpec, SectionSpec

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # clients
    "Taguru",
    "AsyncTaguru",
    "Context",
    "AsyncContext",
    "Contexts",
    "AsyncContexts",
    "Groups",
    "AsyncGroups",
    "citation_key",
    # request types
    "AssocOp",
    "QuestionSpec",
    "SectionSpec",
    # errors
    "TaguruError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "PayloadTooLargeError",
    "RequestTimeoutError",
    "RateLimitError",
    "ServerError",
    "ServiceUnavailableError",
    "StorageFullError",
    "EmbeddingUnavailableError",
    "TransportError",
    "UnexpectedStatusError",
    # models
    "Activation",
    "ActivationPage",
    "AliasEntry",
    "AliasPage",
    "Association",
    "Attribution",
    "BatchApplyResult",
    "Citation",
    "CompactOutcome",
    "ConceptDescription",
    "ContextMeta",
    "ContextPage",
    "ContextStats",
    "ContextUsage",
    "CrossAssociation",
    "CrossMatchPage",
    "CrossPassageHit",
    "DirectoryEntry",
    "ExplorePage",
    "GroupEntry",
    "GroupPage",
    "ImportOutcome",
    "LabelPage",
    "LabelUsage",
    "LaneEvidence",
    "MatchPage",
    "PassageHit",
    "PassageLanes",
    "PassageLookup",
    "Recollection",
    "RefreshBreakdown",
    "RefreshOutcome",
    "RetractOutcome",
    "RetrievalResult",
    "SourcePage",
    "StoredPassages",
    "TieredResolution",
    "TwinPair",
    "VocabularyAudit",
]
