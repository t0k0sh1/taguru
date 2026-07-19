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
    "ContextRevision",
    "DirectoryEntry",
    "ContextPage",
    "GroupEntry",
    "GroupPage",
    "Attribution",
    "Association",
    "MatchPage",
    "CrossAssociation",
    "CrossMatchPage",
    "CrossPassageHit",
    "Recollection",
    "ExplorePage",
    "Activation",
    "ActivationPage",
    "TieredResolution",
    "NearestResolution",
    "NearestGloss",
    "NearestSpellings",
    "LexicalExplain",
    "SemanticExplain",
    "ResolveRanking",
    "ResolveExplanation",
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
    "TermContribution",
    "Bm25Explain",
    "VectorExplain",
    "RankingExplain",
    "SearchExplanation",
    "Citation",
    "RetractOutcome",
    "RetractAssociationOutcome",
    "RefreshBreakdown",
    "RefreshOutcome",
    "TwinPair",
    "VocabularyAudit",
    "UnsourcedEdge",
    "DriftAudit",
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
    dead_edges: int
    dead_attributions: int
    arena_slack: int
    unsourced_edges: int
    unsourced_weight: float
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
class ContextRevision:
    """Change counters a retrieval cache keys on: graph writes, passage
    writes, and config/embedding changes count independently, so a
    consumer watches only the lanes it depends on. Compare for equality
    only, and re-check after a server restart."""

    graph: int
    passages: int
    config: int


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
    #: ``None`` only from a server that predates the field.
    revision: ContextRevision | None = None


@dataclass(slots=True, frozen=True)
class ContextPage:
    """One page of the context directory. ``total`` is the whole population."""

    total: int
    contexts: list[DirectoryEntry]


@dataclass(slots=True, frozen=True)
class GroupEntry:
    """One group row: member contexts bundled many-to-many, plus child groups.

    For a context-scoped key ``contexts`` carries only the members the grant
    allows; ``groups`` (child names — labels, not content) is never filtered.
    """

    name: str
    description: str
    contexts: list[str]
    groups: list[str]
    #: Change token over the transitive member contexts' revisions —
    #: same equality-only contract as ``ContextRevision``. Empty only
    #: from a server that predates the field.
    fingerprint: str = ""


@dataclass(slots=True, frozen=True)
class GroupPage:
    """One page of the group directory. ``total`` is the whole population."""

    total: int
    groups: list[GroupEntry]


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
class MatchPlan:
    """The execution plan of a graph search (#151): the contexts actually
    consulted, in effective order. For the cross variants that is the
    RESOLVED target list — groups expanded, the key's grants applied —
    which the tagged matches alone cannot reconstruct when a target came
    up empty. ``None`` from servers predating the field, and always for
    the non-search :class:`MatchPage` producers (``unreachable_from`` —
    an audit, not a search)."""

    contexts: list[str]


@dataclass(slots=True, frozen=True)
class MatchPage:
    """Ranked matches. ``total`` above ``len(matches)`` means truncation."""

    total: int
    matches: list[Association]
    plan: MatchPlan | None = None


@dataclass(slots=True, frozen=True)
class CrossAssociation(Association):
    """An :class:`Association` tagged with the context it came from.

    The tag is what makes a cross-context match actionable — every follow-up
    (citations, lookups, activate) is a per-context call.
    """

    context: str = ""


@dataclass(slots=True, frozen=True)
class CrossMatchPage:
    """:class:`MatchPage` across contexts: same truncation contract, every
    match tagged. Weights share one scale (evidence mass), so the cut past
    ``total`` means the same thing across contexts."""

    total: int
    matches: list[CrossAssociation]
    plan: MatchPlan | None = None


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
class NearestResolution:
    """One stored spelling near a cue. ``kind`` names the lexical relation
    (``"exact"``/``"alias"``/``"containment"``/``"fuzzy"``)."""

    name: str
    score: float
    kind: str


@dataclass(slots=True, frozen=True)
class NearestGloss:
    """A concept whose gloss embedding sits nearest the cue's."""

    name: str
    cosine: float


@dataclass(slots=True, frozen=True)
class NearestSpellings:
    """Nearest stored spellings for a ``not_in_vocabulary`` verdict — the
    fix (register an alias) is one step away. ``semantic_note`` explains a
    missing semantic list (no provider, no gloss)."""

    lexical: list[NearestResolution]
    semantic: list[NearestGloss]
    semantic_note: str | None = None


@dataclass(slots=True, frozen=True)
class LexicalExplain:
    """The lexical tier's account: the Dice/coverage ``score`` the resolver
    gave (cue → canonical) next to the ``floor`` in effect. ``confident`` is
    whether the tier's best candidate cleared 0.5 — the predicate deciding
    if the semantic tier joins at all."""

    floor: float
    confident: bool
    score: float | None = None
    kind: str | None = None


@dataclass(slots=True, frozen=True)
class SemanticExplain:
    """The semantic tier's account: whether it ``entered`` this call, the
    ``reason`` when it could not, and the expected name's gloss ``cosine``
    against the ``floor`` when the sweep ran. ``cap`` is the fixed count the
    tier serves (not a request knob)."""

    entered: bool
    reason: str | None = None
    floor: float | None = None
    cosine: float | None = None
    rank: int | None = None
    cap: int | None = None


@dataclass(slots=True, frozen=True)
class ResolveRanking:
    """Where the canonical stands against the served list: its ``rank`` and
    ``tier`` when present, and a ``limit_to_reach`` verified by rerunning the
    real serve computation."""

    limit: int
    served: bool
    rank: int | None = None
    tier: str | None = None
    score: float | None = None
    limit_to_reach: int | None = None


@dataclass(slots=True, frozen=True)
class ResolveExplanation:
    """Why a name did (or didn't) resolve for a cue. ``verdict`` is machine-
    readable, ``summary`` human-readable, the rest is evidence: which tiers
    ran, what they scored, and how the expected name ranked. A diagnosed
    miss is a success — every explain call is a 200."""

    verdict: str
    summary: str
    cue: str
    expected: str
    in_vocabulary: bool
    canonical: str | None = None
    expected_kind: str | None = None
    lexical: LexicalExplain | None = None
    semantic: SemanticExplain | None = None
    ranking: ResolveRanking | None = None
    nearest: NearestSpellings | None = None


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
class CrossPassageHit(PassageHit):
    """A :class:`PassageHit` tagged with its context. ``score`` compares
    within one context only — the cross-context order is rank interleaving."""

    context: str = ""


@dataclass(slots=True, frozen=True)
class LanePlan:
    """One lane's verdict for a whole search call (#151): it ran (the vector
    lane also names the effective cosine ``floor`` it swept under — the
    resolved override → context setting → server default chain), or it did
    not and ``reason`` says why, in the same prose ``explain_search`` uses."""

    ran: bool
    reason: str | None = None
    floor: float | None = None


@dataclass(slots=True, frozen=True)
class SearchLanesPlan:
    """Both lanes' verdicts, mirroring the per-hit ``lanes`` shape."""

    bm25: LanePlan
    vector: LanePlan


@dataclass(slots=True, frozen=True)
class SearchContextPlan:
    """One searched context's account within a :class:`SearchPlan`."""

    context: str
    lanes: SearchLanesPlan


@dataclass(slots=True, frozen=True)
class SearchPlan:
    """The execution plan of one passage search (#151): one entry per
    context actually searched, in effective order. What the per-hit lane
    evidence cannot say — "the semantic lane never ran here, and this is
    why" — lives here, so a lexical-only answer is distinguishable from a
    fused one without a separate explain call."""

    contexts: list[SearchContextPlan]


@dataclass(slots=True, frozen=True)
class PassagePage:
    """``search_passages``' response: the :class:`SearchPlan` beside the
    hits it accounts for."""

    plan: SearchPlan
    hits: list[PassageHit]


@dataclass(slots=True, frozen=True)
class CrossPassagePage:
    """:class:`PassagePage` across contexts: the same wrap, every hit
    tagged."""

    plan: SearchPlan
    hits: list[CrossPassageHit]


@dataclass(slots=True, frozen=True)
class TermContribution:
    """One query term against the target paragraph: ``df`` paragraphs carry
    it corpus-wide (its ``idf`` follows), the target carries it ``tf`` times,
    adding ``contribution`` to the BM25 score. ``tf`` 0 with a high ``df`` is
    the "matched only ubiquitous bigrams" signature."""

    term: str
    tf: float
    df: int
    idf: float
    contribution: float


@dataclass(slots=True, frozen=True)
class Bm25Explain:
    """The lexical lane's evidence: the target's ``rank`` in that lane, its
    BM25 ``score``, and the score's per-term addends."""

    score: float
    terms: list[TermContribution]
    rank: int | None = None


@dataclass(slots=True, frozen=True)
class VectorExplain:
    """The vector lane's evidence — or the ``reason`` there is none.
    ``cosine`` is the target's best across its rows, floor or no floor."""

    ran: bool
    reason: str | None = None
    floor: float | None = None
    cosine: float | None = None
    rank: int | None = None


@dataclass(slots=True, frozen=True)
class RankingExplain:
    """Where the target stands in the fused ranking ``search_passages``
    truncates: its ``rank`` against ``ranked`` scored candidates, the
    ``cutoff_score`` the ``limit`` served down to, and a ``limit_to_reach``
    verified by rerunning the real serve computation."""

    fused: bool
    ranked: int
    limit: int
    served: bool
    rank: int | None = None
    score: float | None = None
    cutoff_score: float | None = None
    limit_to_reach: int | None = None


@dataclass(slots=True, frozen=True)
class SearchExplanation:
    """Why a source did (or didn't) appear for a query. ``verdict`` is
    machine-readable, ``summary`` human-readable, the rest is evidence: the
    query's ``query_terms``, the lexical/vector lanes, and the ranking. A
    diagnosed miss is a success — every explain call is a 200."""

    verdict: str
    summary: str
    source: str
    paragraph: int | None = None
    paragraphs: int | None = None
    paragraph_named: bool | None = None
    query_terms: list[str] | None = None
    paragraph_terms: list[str] | None = None
    bm25: Bm25Explain | None = None
    vector: VectorExplain | None = None
    ranking: RankingExplain | None = None


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
class RetractAssociationOutcome:
    """``retracted=False`` means the triple named no live edge — nothing
    changed. ``attributions_removed`` counts the per-source records
    unlinked with the edge (0 for one carrying only unsourced weight)."""

    retracted: bool
    attributions_removed: int


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
class UnsourcedEdge:
    """One edge carrying weight no named source explains. ``unsourced_weight``
    can be negative."""

    unsourced_weight: float
    unsourced_count: int
    association: Association


@dataclass(slots=True, frozen=True)
class DriftAudit:
    """Graph-vs-archive drift: unsourced weight, dead-canonical aliases, and
    (opt-in) the same fork candidates :class:`VocabularyAudit` finds."""

    total: int
    unsourced: list[UnsourcedEdge]
    dead_concept_aliases: dict[str, str]
    dead_label_aliases: dict[str, str]
    twins: VocabularyAudit | None = None


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
class GroupImportOutcome:
    """Outcome of restoring one ``taguru_group`` record via import.

    A restore is a replace of the whole record; ``outcome`` is one of
    ``"created"``, ``"replaced"``, or ``"unchanged"``. ``contexts``/``groups``
    are the member counts of the record as restored.
    """

    name: str
    outcome: str
    contexts: int
    groups: int


@dataclass(slots=True, frozen=True)
class ImportResult:
    """What ``POST /import`` accomplished: per-batch outcomes plus any group restores."""

    batches: list[ImportOutcome]
    groups: list[GroupImportOutcome]


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
    search_plan: SearchPlan | None = None
    """The fallback search's :class:`SearchPlan` — ``None`` when no text
    fallback ran, so "the search never happened" and "the semantic lane
    was skipped" stay distinguishable."""
