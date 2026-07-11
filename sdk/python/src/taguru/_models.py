"""Response models, mirroring the server's wire shapes field-for-field.

All models are frozen dataclasses decoded tolerantly (unknown fields are
ignored — the protocol promises additive-only evolution; absent optional
fields are omitted on the wire, not null, and decode to ``None``).

Weight semantics worth knowing: ``Association.weight`` is the per-assertion
average (sum/count); each ``Attribution.weight`` inside it is that source's
raw cumulative sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "LabelUsage",
    "ContextStats",
    "ContextUsage",
    "ContextMeta",
    "DirectoryEntry",
    "ContextPage",
    "Attribution",
    "Association",
    "MatchPage",
    "Recollection",
    "ExplorePage",
    "Activation",
    "ActivationPage",
    "TieredResolution",
    "ConceptDescription",
    "LabelPage",
    "AliasPage",
    "AliasEntry",
    "SourcePage",
    "StoredPassages",
    "PassageLookup",
    "LaneEvidence",
    "PassageLanes",
    "PassageHit",
    "Citation",
    "RetractOutcome",
    "RefreshBreakdown",
    "RefreshOutcome",
    "TwinPair",
    "VocabularyAudit",
    "CompactOutcome",
    "ImportOutcome",
    "BatchApplyResult",
    "RetrievalResult",
]


@dataclass(slots=True, frozen=True)
class LabelUsage:
    label: str
    count: int


@dataclass(slots=True, frozen=True)
class ContextStats:
    associations: int
    concepts: int
    labels: int
    sources: int
    footprint_bytes: int
    top_concepts: list[LabelUsage]
    label_sample: list[str]


@dataclass(slots=True, frozen=True)
class ContextUsage:
    reads: int
    empty_reads: int
    writes: int
    last_read_epoch: int
    last_write_epoch: int


@dataclass(slots=True, frozen=True)
class ContextMeta:
    description: str
    pinned: bool
    dice_floor: float | None = None
    semantic_floor: float | None = None


@dataclass(slots=True, frozen=True)
class DirectoryEntry:
    name: str
    description: str
    pinned: bool
    loaded: bool
    stats: ContextStats
    usage: ContextUsage
    dice_floor: float | None = None
    semantic_floor: float | None = None


@dataclass(slots=True, frozen=True)
class ContextPage:
    """One page of the context directory. ``total`` is the whole population."""

    total: int
    contexts: list[DirectoryEntry]


@dataclass(slots=True, frozen=True)
class Attribution:
    """One source's contribution to an association. ``weight`` is raw cumulative."""

    source: str
    weight: float
    count: int
    paragraph: int | None = None
    section: str | None = None


@dataclass(slots=True, frozen=True)
class Association:
    """One (subject, label, object) edge. ``weight`` is the per-assertion average."""

    subject: str
    label: str
    object: str
    weight: float
    count: int
    attributions: list[Attribution] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class MatchPage:
    """Ranked matches. ``total`` above ``len(matches)`` means truncation."""

    total: int
    matches: list[Association]


@dataclass(slots=True, frozen=True)
class Recollection:
    distance: int
    path: list[str]
    association: Association


@dataclass(slots=True, frozen=True)
class ExplorePage:
    total: int
    matches: list[Recollection]


@dataclass(slots=True, frozen=True)
class Activation:
    """``strength`` is an ordering within one call — never compare across calls."""

    strength: float
    path: list[str]
    association: Association


@dataclass(slots=True, frozen=True)
class ActivationPage:
    total: int
    matches: list[Activation]


@dataclass(slots=True, frozen=True)
class TieredResolution:
    """One resolve candidate. ``tier`` is ``"lexical"`` or ``"semantic"``.

    ``kind`` (lexical only) is ``"exact"``/``"alias"``/``"containment"``/
    ``"fuzzy"`` — never adopt a containment/fuzzy hit on score alone; read
    ``gloss`` first.
    """

    name: str
    score: float
    tier: str
    kind: str | None = None
    gloss: str | None = None


@dataclass(slots=True, frozen=True)
class ConceptDescription:
    concept: str
    as_subject: list[LabelUsage]
    as_object: list[LabelUsage]


@dataclass(slots=True, frozen=True)
class LabelPage:
    total: int
    labels: list[str]


@dataclass(slots=True, frozen=True)
class AliasPage:
    """One page of aliases; the cursor spans both namespaces (concepts first)."""

    total: int
    concepts: dict[str, str]
    labels: dict[str, str]


@dataclass(slots=True, frozen=True)
class AliasEntry:
    """A flattened alias row as yielded by ``iter_aliases``."""

    namespace: str  # "concept" | "label"
    alias: str
    canonical: str


@dataclass(slots=True, frozen=True)
class SourcePage:
    total: int
    sources: list[str]


@dataclass(slots=True, frozen=True)
class StoredPassages:
    """A dropped question/section named a paragraph its passage's split lacks."""

    stored: int
    questions_stored: int
    questions_dropped: int
    sections_stored: int
    sections_dropped: int


@dataclass(slots=True, frozen=True)
class PassageLookup:
    passages: dict[str, str]
    missing: list[str]


@dataclass(slots=True, frozen=True)
class LaneEvidence:
    rank: int
    score: float


@dataclass(slots=True, frozen=True)
class PassageLanes:
    """Per-lane evidence; a lane the server didn't run is omitted (None)."""

    bm25: LaneEvidence | None = None
    vector: LaneEvidence | None = None


@dataclass(slots=True, frozen=True)
class PassageHit:
    """One paragraph hit from ``search_passages``. ``text`` is that paragraph alone."""

    source: str
    paragraph: int
    score: float
    text: str
    lanes: PassageLanes = field(default_factory=PassageLanes)


@dataclass(slots=True, frozen=True)
class Citation:
    """One verbatim paragraph. ``section`` is null outside every stored section."""

    text: str
    source: str
    section: str | None = None


@dataclass(slots=True, frozen=True)
class RetractOutcome:
    associations_touched: int
    passage_removed: bool


@dataclass(slots=True, frozen=True)
class RefreshBreakdown:
    embedded: int
    total: int
    skipped_over_limit: int | None = None


@dataclass(slots=True, frozen=True)
class RefreshOutcome:
    embedded: int
    total: int
    glosses: RefreshBreakdown | None = None
    passages: RefreshBreakdown | None = None


@dataclass(slots=True, frozen=True)
class TwinPair:
    a: str
    b: str
    score: float


@dataclass(slots=True, frozen=True)
class VocabularyAudit:
    """Fork candidates, not verdicts — adjudicate each pair."""

    lexical_concepts: list[TwinPair]
    lexical_labels: list[TwinPair]
    semantic_concepts: list[TwinPair]
    semantic_labels: list[TwinPair]
    semantic_note: str | None = None


@dataclass(slots=True, frozen=True)
class CompactOutcome:
    bytes_before: int
    bytes_after: int
    dead_edges: int
    aliases_dropped: int


@dataclass(slots=True, frozen=True)
class ImportOutcome:
    """Outcome of one applied batch (one source's retract-then-apply)."""

    context: str
    source: str
    created: bool
    retracted: int
    associations: int
    aliases: int
    passage_stored: bool
    passage_dropped: bool
    questions_stored: int
    questions_dropped: int
    sections_stored: int
    sections_dropped: int
    association_paragraphs_dropped: int


@dataclass(slots=True, frozen=True)
class BatchApplyResult:
    """Outcome of ``add_associations_batched``: chunks are independent writes."""

    applied: int
    chunks: int


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Everything one ``retrieve()`` pass gathered.

    ``resolved`` keeps every candidate (with glosses) so a lookalike anchor is
    never hidden from the calling LLM. ``citations`` is keyed by
    ``(source, paragraph)`` — build keys with :func:`taguru.citation_key`.
    """

    resolved: dict[str, list[TieredResolution]]
    outline: dict[str, ConceptDescription | None]
    associations: list[Association]
    activations: list[Activation]
    citations: dict[tuple[str, int], Citation]
    passage_hits: list[PassageHit]
