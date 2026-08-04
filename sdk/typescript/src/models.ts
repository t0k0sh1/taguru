/**
 * Wire types, mirroring the server's shapes field-for-field (snake_case, same
 * as the Python SDK). Two kinds of absence are distinguished deliberately:
 * `field?: T` for keys the server OMITS when absent (serde
 * `skip_serializing_if`) and `field: T | null` for keys that are always
 * present but nullable. Unknown extra fields are tolerated natively.
 *
 * Weight semantics worth knowing: `Association.weight` is the per-assertion
 * average (sum/count); each `Attribution.weight` inside it is that source's
 * raw cumulative sum.
 */

// -- request-shaped inputs ----------------------------------------------------

/**
 * One association to assert. `paragraph` locates the fact within `source`
 * (its blank-line paragraph index) and is ignored by the server without a
 * `source`.
 */
export interface AssocOp {
  subject: string;
  label: string;
  object: string;
  weight: number;
  source?: string;
  paragraph?: number;
}

/** A doc2query question attached to one paragraph of a stored passage. */
export interface QuestionSpec {
  paragraph: number;
  question: string;
}

/** A section label governing paragraphs from `paragraph` onward. */
export interface SectionSpec {
  paragraph: number;
  section: string;
}

/**
 * A typed citation locator attached to one paragraph of a stored passage
 * (ADR 0007 §7). Independent of {@link SectionSpec} — unlike a section, a
 * locator does not extend to the next paragraph.
 */
export interface LocatorSpec {
  paragraph: number;
  locator: Locator;
}

/**
 * Resumes a `recall`/`query`/`unreachableFrom`/`auditDrift` page past its
 * last match: copy `weight`/`subject`/`label`/`object` verbatim from the
 * last match of the previous page (for `auditDrift`, that's the
 * `association` nested inside the last `unsourced` entry).
 */
export interface MatchCursor {
  weight: number;
  subject: string;
  label: string;
  object: string;
}

/**
 * Narrows a match down to the four fields `MatchCursor` wants.
 * `Association` structurally satisfies `MatchCursor` (it has `weight`,
 * `subject`, `label`, and `object`, so TypeScript happily accepts passing
 * the previous page's last match straight back as `after`), but it also
 * carries `count` and `attributions` — and the server's `MatchCursor`
 * rejects any field it doesn't recognize, so a match passed verbatim 400s.
 * Route every `after` built from a match through this first; `recall`,
 * `query`, `unreachableFrom`, and `auditDrift` do.
 */
export function matchCursor(match: MatchCursor): MatchCursor {
  return { weight: match.weight, subject: match.subject, label: match.label, object: match.object };
}

/**
 * `MatchCursor` plus `context`, for cross-context `recall`/`query`
 * (`contexts`/`groups`). `context` is the tiebreak two different target
 * contexts can't share on their own: each can independently hold an edge at
 * the identical `(subject, label, object)`.
 */
export interface CrossMatchCursor {
  weight: number;
  context: string;
  subject: string;
  label: string;
  object: string;
}

/** `matchCursor`, for the cross-context match shape (`context` included). */
export function crossMatchCursor(match: CrossMatchCursor): CrossMatchCursor {
  return {
    weight: match.weight,
    context: match.context,
    subject: match.subject,
    label: match.label,
    object: match.object,
  };
}

/**
 * Resumes an `explore` page past its last recollection: copy
 * `distance`/`subject`/`label`/`object` verbatim from it.
 */
export interface ExploreCursor {
  distance: number;
  subject: string;
  label: string;
  object: string;
}

/**
 * Explicit ceilings for one `assembleEvidence` call. Each field falls back
 * to the server default (40 items / 65536 bytes / 4000 tokens) when
 * omitted.
 */
export interface BudgetRequest {
  max_items?: number;
  max_bytes?: number;
  max_tokens?: number;
}

/**
 * Opt into a configured reranker for one `assembleEvidence` call. Omitting
 * `rerank` entirely leaves every provider untouched; a failure degrades to
 * the deterministic order and is reported in `EvidencePlan.reranker.reason`.
 */
export interface RerankRequest {
  model?: string;
}

/** One name or an OR-set of names, for query() positions. */
export type OneOrMany = string | string[];

/**
 * An enum-like wire field: the known values, plus any value a newer
 * server may add. ADR 0005 §5 requires every enum-like field to be
 * open — a closed string-literal union silently lies the moment the
 * server ships a value it doesn't list, where Python's plain `str`
 * fields (no `Literal`/`Enum` types anywhere in `_models.py`) already
 * pass such a value through untouched. `(string & {})` keeps the known
 * literals' autocomplete while widening acceptance to any string.
 */
export type Open<T extends string> = T | (string & {});

// -- directory ----------------------------------------------------------------

export interface LabelUsage {
  label: string;
  count: number;
}

export interface ContextStats {
  associations: number;
  concepts: number;
  labels: number;
  sources: number;
  footprint_bytes: number;
  dead_edges: number;
  dead_attributions: number;
  arena_slack: number;
  unsourced_edges: number;
  unsourced_weight: number;
  top_concepts: LabelUsage[];
  label_sample: string[];
}

export interface ContextUsage {
  reads: number;
  empty_reads: number;
  writes: number;
  last_read_epoch: number;
  last_write_epoch: number;
}

export interface ContextMeta {
  description: string;
  pinned: boolean;
  dice_floor: number | null;
  semantic_floor: number | null;
}

/**
 * Change counters a retrieval cache keys on: graph writes, passage writes,
 * and config/embedding changes count independently, so a consumer watches
 * only the lanes it depends on. Compare for equality only, and re-check
 * after a server restart.
 */
export interface ContextRevision {
  graph: number;
  passages: number;
  config: number;
}

export interface DirectoryEntry {
  name: string;
  description: string;
  pinned: boolean;
  loaded: boolean;
  dice_floor: number | null;
  semantic_floor: number | null;
  stats: ContextStats;
  usage: ContextUsage;
  /** Absent only from a server that predates the field. */
  revision?: ContextRevision;
}

/** One page of the context directory. `total` is the whole population. */
export interface ContextPage {
  total: number;
  contexts: DirectoryEntry[];
}

// -- groups ---------------------------------------------------------------------

/**
 * One group row: member contexts bundled many-to-many, plus child groups. For
 * a context-scoped key `contexts` carries only the members the grant allows;
 * `groups` (child names — labels, not content) is never filtered.
 */
export interface GroupEntry {
  name: string;
  description: string;
  contexts: string[];
  groups: string[];
  /**
   * Change token over the transitive member contexts' revisions — same
   * equality-only contract as {@link ContextRevision}. Absent only from a
   * server that predates the field.
   */
  fingerprint?: string;
}

/** One page of the group directory. `total` is the whole population. */
export interface GroupPage {
  total: number;
  groups: GroupEntry[];
}

// -- graph shapes ---------------------------------------------------------------

/**
 * A typed citation locator (page/slide/sheet/table position, ADR 0007 §7).
 * `kind` is an open string (`"page"`, `"slide"`, `"sheet"`, `"table"`, and
 * whatever a future connector needs); `value` is free text since its
 * natural representation varies by kind (`"12"`, `"A1:C4"`). Independent
 * of `section` — unlike it, a locator does not extend to the next
 * paragraph.
 */
export interface Locator {
  kind: string;
  value: string;
}

/** One source's contribution to an association. `weight` is raw cumulative. */
export interface Attribution {
  source: string;
  weight: number;
  count: number;
  paragraph: number | null;
  section: string | null;
  locator: Locator | null;
}

/** One (subject, label, object) edge. `weight` is the per-assertion average. */
export interface Association {
  subject: string;
  label: string;
  object: string;
  weight: number;
  count: number;
  attributions: Attribution[];
}

/**
 * The execution plan of a graph search (#151): the contexts actually
 * consulted, in effective order. For the cross variants that is the RESOLVED
 * target list — groups expanded, the key's grants applied — which the tagged
 * matches alone cannot reconstruct when a target came up empty. Absent from
 * servers predating the field, and always for the non-search `MatchPage`
 * producers (`unreachableFrom` — an audit, not a search).
 */
export interface MatchPlan {
  contexts: string[];
}

/** Ranked matches. `total` above `matches.length` means truncation. */
export interface MatchPage {
  total: number;
  matches: Association[];
  plan?: MatchPlan;
}

/**
 * An `Association` tagged with the context it came from. The tag is what
 * makes a cross-context match actionable — every follow-up (citations,
 * lookups, activate) is a per-context call.
 */
export interface CrossAssociation extends Association {
  context: string;
}

/**
 * `MatchPage` across contexts: same truncation contract, every match tagged.
 * Weights share one scale (evidence mass), so the cut past `total` means the
 * same thing across contexts.
 */
export interface CrossMatchPage {
  total: number;
  matches: CrossAssociation[];
  plan?: MatchPlan;
}

export interface Recollection {
  distance: number;
  path: string[];
  association: Association;
}

export interface ExplorePage {
  total: number;
  matches: Recollection[];
}

/** `strength` is an ordering within one call — never compare across calls. */
export interface Activation {
  strength: number;
  path: string[];
  association: Association;
}

export interface ActivationPage {
  total: number;
  matches: Activation[];
}

/**
 * One resolve candidate. `kind` (lexical tier only) is
 * "exact"/"alias"/"containment"/"fuzzy" — never adopt a containment/fuzzy hit
 * on score alone; read `gloss` first. `types` (top candidates only) are the
 * candidate's declared entity types, present only when the context has an
 * installed schema and the candidate carries a type assertion.
 */
export interface TieredResolution {
  name: string;
  score: number;
  tier: Open<"lexical" | "semantic">;
  kind?: Open<"exact" | "alias" | "containment" | "fuzzy">;
  gloss?: string;
  types?: string[];
}

// -- resolve explain -----------------------------------------------------------

/**
 * One stored spelling near a cue. `kind` names the lexical relation — never
 * adopt a containment/fuzzy neighbor as the fix without reading the gloss.
 */
export interface NearestResolution {
  name: string;
  score: number;
  kind: Open<"exact" | "alias" | "containment" | "fuzzy">;
}

/** A concept whose gloss embedding sits nearest the cue's. */
export interface NearestGloss {
  name: string;
  cosine: number;
}

/**
 * Nearest stored spellings for a `not_in_vocabulary` verdict — the fix
 * (register an alias) is one step away. `semantic_note` explains a missing
 * semantic list (no provider, no gloss).
 */
export interface NearestSpellings {
  lexical: NearestResolution[];
  semantic: NearestGloss[];
  semantic_note?: string;
}

/**
 * The lexical tier's account: the Dice/coverage `score` the resolver gave
 * (cue → canonical) next to the `floor` in effect. `confident` is whether the
 * tier's best candidate cleared 0.5 — the predicate deciding if the semantic
 * tier joins at all.
 */
export interface LexicalExplain {
  score?: number;
  kind?: Open<"exact" | "alias" | "containment" | "fuzzy">;
  floor: number;
  confident: boolean;
}

/**
 * The semantic tier's account: whether it `entered` this call, the `reason`
 * when it could not, and the expected name's gloss `cosine` against the
 * `floor` when the sweep ran. `cap` is the fixed count the tier serves (not a
 * request knob).
 */
export interface SemanticExplain {
  entered: boolean;
  reason?: string;
  floor?: number;
  cosine?: number;
  rank?: number;
  cap?: number;
}

/**
 * Where the canonical stands against the served list: its `rank` and `tier`
 * when present, and a `limit_to_reach` verified by rerunning the real serve
 * computation.
 */
export interface ResolveRanking {
  rank?: number;
  tier?: Open<"lexical" | "semantic">;
  score?: number;
  limit: number;
  served: boolean;
  limit_to_reach?: number;
}

/**
 * Why a name did (or didn't) resolve for a cue. `verdict` is machine-readable,
 * `summary` human-readable, the rest is evidence: which tiers ran, what they
 * scored, and how the expected name ranked. A diagnosed miss is a success —
 * every explain call is a 200.
 */
export interface ResolveExplanation {
  verdict: Open<
    | "not_in_vocabulary"
    | "served"
    | "cue_resolved_exactly"
    | "below_floor"
    | "below_cutoff"
    | "semantic_not_run"
    | "semantic_below_floor"
  >;
  summary: string;
  cue: string;
  expected: string;
  in_vocabulary: boolean;
  canonical?: string;
  expected_kind?: Open<"exact" | "alias">;
  lexical?: LexicalExplain;
  semantic?: SemanticExplain;
  ranking?: ResolveRanking;
  nearest?: NearestSpellings;
}

export interface ConceptDescription {
  concept: string;
  as_subject: LabelUsage[];
  as_object: LabelUsage[];
  /**
   * The concept's declared types; empty for a schema-free context or an
   * untyped concept (the two are indistinguishable here on purpose).
   */
  types: string[];
}

export interface LabelPage {
  total: number;
  labels: string[];
}

// -- schema ------------------------------------------------------------------------

/**
 * One entity type's `is_a` parents (ADR 0009 §6.2) — an undeclared
 * parent is legal, a leaf ancestor with nothing further above it.
 */
export interface TypeDef {
  is_a: string[];
}

/**
 * One relation label's domain/range constraint (ADR 0009 §6). An empty
 * side is unconstrained, never "matches anything."
 */
export interface RelationDef {
  domain: string[];
  range: string[];
}

/**
 * A context's schema document — the same shape `{stem}.schema.json`/
 * `GET /contexts/{name}/schema` persist and serve, mirrored
 * field-for-field from `src/schema.rs`'s `SchemaDocument` (ADR 0009 §5.3).
 */
export interface SchemaDocument {
  schema: number;
  mode: Open<"off" | "warn" | "strict">;
  closed_labels: boolean;
  types: Record<string, TypeDef>;
  relations: Record<string, RelationDef>;
}

// -- aliases ---------------------------------------------------------------------

/** One page of aliases; the cursor spans both namespaces (concepts first). */
export interface AliasPage {
  total: number;
  concepts: Record<string, string>;
  labels: Record<string, string>;
}

/**
 * A flattened alias row as yielded by `iterAliases`. Deliberately kept
 * closed, unlike this file's other enum-like fields (ADR 0005 §5):
 * `namespace` is never decoded from the wire — `iterAliases` (`client.ts`)
 * synthesizes it client-side from `AliasPage`'s two fixed keys
 * (`concepts`/`labels`), so a server can never surprise this SDK with a
 * value it didn't itself mint.
 */
export interface AliasEntry {
  namespace: "concept" | "label";
  alias: string;
  canonical: string;
}

// -- passages / sources ------------------------------------------------------------

/**
 * One listed source with its metadata (#167). `stored_at` is the server's
 * stamp (epoch seconds), `date` the user-supplied document time; a source
 * stored before metadata existed lists as bare.
 */
export interface SourceEntry {
  name: string;
  stored_at?: number;
  date?: number;
  tags?: string[];
}

/**
 * `sources` is the bare id list; `entries` carries the same page with each
 * source's metadata beside it.
 */
export interface SourcePage {
  total: number;
  sources: string[];
  entries: SourceEntry[];
}

/** A dropped question/section named a paragraph its passage's split lacks. */
export interface StoredPassages {
  stored: number;
  questions_stored: number;
  questions_dropped: number;
  sections_stored: number;
  sections_dropped: number;
  locators_stored: number;
  locators_dropped: number;
}

export interface PassageLookup {
  passages: Record<string, string>;
  missing: string[];
}

export interface LaneEvidence {
  rank: number;
  score: number;
}

/** Per-lane evidence; a lane the server didn't run is omitted entirely. */
export interface PassageLanes {
  bm25?: LaneEvidence;
  vector?: LaneEvidence;
}

/** One paragraph hit from `searchPassages`. `text` is that paragraph alone. */
export interface PassageHit {
  source: string;
  paragraph: number;
  score: number;
  text: string;
  lanes: PassageLanes;
}

/**
 * A `PassageHit` tagged with its context. `score` compares within one context
 * only — the cross-context order is rank interleaving.
 */
export interface CrossPassageHit extends PassageHit {
  context: string;
}

/**
 * One lane's verdict for a whole search call (#151): it ran (the vector lane
 * also names the effective cosine `floor` it swept under — the resolved
 * override → context setting → server default chain), or it did not and
 * `reason` says why, in the same prose `explainSearchPassages` uses.
 */
export interface LanePlan {
  ran: boolean;
  reason?: string;
  floor?: number;
}

/** Both lanes' verdicts, mirroring the per-hit `lanes` shape. */
export interface SearchLanesPlan {
  bm25: LanePlan;
  vector: LanePlan;
}

/**
 * The source filter's account for one searched context (#167): how many
 * sources were eligible to answer, out of how many the context stores —
 * present exactly when the request carried a filter.
 */
export interface FilterPlan {
  eligible_sources: number;
  total_sources: number;
}

/** One searched context's account within a `SearchPlan`. */
export interface SearchContextPlan {
  context: string;
  lanes: SearchLanesPlan;
  filter?: FilterPlan;
}

/**
 * The execution plan of one passage search (#151): one entry per context
 * actually searched, in effective order. What the per-hit lane evidence
 * cannot say — "the semantic lane never ran here, and this is why" — lives
 * here, so a lexical-only answer is distinguishable from a fused one without
 * a separate explain call.
 */
export interface SearchPlan {
  contexts: SearchContextPlan[];
}

/** `searchPassages`' response: the `SearchPlan` beside the hits. */
export interface PassagePage {
  plan: SearchPlan;
  hits: PassageHit[];
}

/** `PassagePage` across contexts: the same wrap, every hit tagged. */
export interface CrossPassagePage {
  plan: SearchPlan;
  hits: CrossPassageHit[];
}

/**
 * One member concept of a ranked community, with its intra-community
 * strength (the membership edge's weight on the artifact).
 */
export interface CommunityHitMember {
  name: string;
  strength: number;
}

/**
 * One ranked community from `searchCommunities`: the matched summary
 * paragraph plus the community's manifest facts. The manifest fields are
 * absent when the artifact is mid-rewrite and this summary landed before its
 * manifest line — served honestly rather than dropped.
 */
export interface CommunityHit {
  community: string;
  score: number;
  text: string;
  paragraph: number;
  level?: number;
  parent?: string;
  concept_count?: number;
  members?: CommunityHitMember[];
  members_truncated?: boolean;
}

/** The two source-graph revisions the staleness verdict compares. */
export interface CommunityRevisions {
  recorded_graph: number;
  current_graph: number;
}

/**
 * `searchCommunities`' response: ranked community summaries and the verdict
 * that qualifies them — `stale` means the source graph moved since the
 * artifact was derived (the summaries describe an older graph; re-run
 * `taguru communities` to refresh).
 */
export interface CommunityPage {
  derived: string;
  algorithm: string;
  stale: boolean;
  revision: CommunityRevisions;
  plan: SearchPlan;
  hits: CommunityHit[];
}

// -- search explain ------------------------------------------------------------

/**
 * One query term against the target paragraph: `df` paragraphs carry it
 * corpus-wide (its `idf` follows), the target carries it `tf` times,
 * contributing `contribution` to the BM25 score. `tf` 0 with a high `df` is
 * the "matched only ubiquitous bigrams" signature.
 */
export interface TermContribution {
  term: string;
  tf: number;
  df: number;
  idf: number;
  contribution: number;
}

/**
 * The lexical lane's evidence for the target: its `rank` in that lane, its
 * BM25 `score`, and the score's per-term addends.
 */
export interface Bm25Explain {
  rank?: number;
  score: number;
  terms: TermContribution[];
}

/** The vector lane's evidence — or the `reason` there is none. */
export interface VectorExplain {
  ran: boolean;
  reason?: string;
  floor?: number;
  cosine?: number;
  rank?: number;
}

/**
 * Where the target stands in the fused ranking `searchPassages` truncates: its
 * `rank` against `ranked` scored candidates, the `cutoff_score` the request's
 * `limit` served down to, and a `limit_to_reach` verified by rerunning the
 * real serve computation (pool caps included).
 */
export interface RankingExplain {
  fused: boolean;
  ranked: number;
  rank?: number;
  score?: number;
  limit: number;
  served: boolean;
  cutoff_score?: number;
  limit_to_reach?: number;
}

/**
 * Why a source did (or didn't) appear for a query. `verdict` is machine-
 * readable, `summary` human-readable, the rest is evidence: the terms each
 * lane matched against and where the target ranked. A diagnosed miss is a
 * success — every explain call is a 200.
 */
export interface SearchExplanation {
  verdict: Open<
    | "not_stored"
    | "paragraph_out_of_range"
    | "no_query_terms"
    | "no_term_overlap"
    | "below_cutoff"
    | "served"
  >;
  summary: string;
  source: string;
  paragraph?: number;
  paragraphs?: number;
  paragraph_named?: boolean;
  query_terms?: string[];
  paragraph_terms?: string[];
  bm25?: Bm25Explain;
  vector?: VectorExplain;
  ranking?: RankingExplain;
}

/** One verbatim paragraph. `section`/`locator` are null when absent. */
export interface Citation {
  text: string;
  source: string;
  section: string | null;
  locator: Locator | null;
}

export interface RetractOutcome {
  associations_touched: number;
  passage_removed: boolean;
}

/**
 * `retracted: false` means the triple named no live edge — nothing changed.
 * `attributions_removed` counts the per-source records unlinked with the
 * edge (0 for one carrying only unsourced weight).
 */
export interface RetractAssociationOutcome {
  retracted: boolean;
  attributions_removed: number;
}

// -- maintenance ---------------------------------------------------------------------

export interface RefreshBreakdown {
  embedded: number;
  total: number;
  skipped_over_limit?: number;
}

export interface RefreshOutcome {
  embedded: number;
  total: number;
  glosses?: RefreshBreakdown;
  passages?: RefreshBreakdown;
}

export interface TwinPair {
  a: string;
  b: string;
  score: number;
}

/** Fork candidates, not verdicts — adjudicate each pair. */
export interface VocabularyAudit {
  lexical_concepts: TwinPair[];
  lexical_labels: TwinPair[];
  semantic_concepts: TwinPair[];
  semantic_labels: TwinPair[];
  semantic_note: string | null;
}

/** One edge carrying weight no named source explains. `unsourced_weight` can be negative. */
export interface UnsourcedEdge {
  unsourced_weight: number;
  unsourced_count: number;
  association: Association;
}

/**
 * Graph-vs-archive drift: unsourced weight, dead-canonical aliases, and
 * (opt-in) the same fork candidates `VocabularyAudit` finds.
 */
export interface DriftAudit {
  total: number;
  unsourced: UnsourcedEdge[];
  dead_concept_aliases: Record<string, string>;
  dead_label_aliases: Record<string, string>;
  twins: VocabularyAudit | null;
}

export interface CompactOutcome {
  bytes_before: number;
  bytes_after: number;
  dead_edges: number;
  aliases_dropped: number;
}

/** Outcome of one applied batch (one source's retract-then-apply). */
export interface ImportOutcome {
  context: string;
  source: string;
  created: boolean;
  retracted: number;
  associations: number;
  aliases: number;
  passage_stored: boolean;
  passage_dropped: boolean;
  questions_stored: number;
  questions_dropped: number;
  sections_stored: number;
  sections_dropped: number;
  locators_stored: number;
  locators_dropped: number;
  association_paragraphs_dropped: number;
  /**
   * `warn`-mode schema violations this batch's associations raised (ADR
   * 0009 §8.3). Absent only from a server that predates the field.
   */
  schema_violations?: number;
}

/**
 * Outcome of restoring one `taguru_group` record via import. A restore is a
 * replace of the whole record; `outcome` says what it replaced.
 */
export interface GroupImportOutcome {
  name: string;
  outcome: Open<"created" | "replaced" | "unchanged">;
  /** Member counts of the record as restored. */
  contexts: number;
  groups: number;
}

/** What `POST /import` accomplished: per-batch outcomes plus any group restores. */
export interface ImportResult {
  batches: ImportOutcome[];
  groups: GroupImportOutcome[];
}

/** Outcome of `addAssociationsBatched`: chunks are independent writes. */
export interface BatchApplyResult {
  applied: number;
  chunks: number;
}

/**
 * Everything one `retrieve()` pass gathered.
 *
 * `resolved` keeps every candidate (with glosses) so a lookalike anchor is
 * never hidden from the calling LLM. `citations` is keyed by
 * `citationKey(source, paragraph)`.
 */
export interface RetrievalResult {
  resolved: Record<string, TieredResolution[]>;
  outline: Record<string, ConceptDescription | null>;
  associations: Association[];
  activations: Activation[];
  citations: Map<string, Citation>;
  passage_hits: PassageHit[];
  /**
   * The fallback search's `SearchPlan` — absent when no text fallback ran,
   * so "the search never happened" and "the semantic lane was skipped" stay
   * distinguishable.
   */
  search_plan?: SearchPlan;
}

/**
 * The key `RetrievalResult.citations` is indexed by. NUL-delimited so no
 * source id can collide with another (source, paragraph) pair — always build
 * keys with this helper rather than by hand.
 */
export function citationKey(source: string, paragraph: number): string {
  return `${source}\u0000${paragraph}`;
}

// -- assembleEvidence (#216, #305, #306; ADR 0006 §10) -------------------------

/**
 * One lane's contribution to an `EvidenceItem`'s fused rank. `lane` is open:
 * `"graph_query"`, `"graph_activate"`, `"passage_bm25"`, `"passage_vector"`,
 * `"passage_fused"`, or `"community"` today, with more possible later.
 */
export interface LaneRank {
  lane: Open<"graph_query" | "graph_activate" | "passage_bm25" | "passage_vector" | "passage_fused" | "community">;
  rank: number;
}

/**
 * A `(source, paragraph)` locator with no citation text attached — the text
 * itself lives exactly once, keyed the same way, in the package's top-level
 * `citations`.
 */
export interface CitationRef {
  source: string;
  paragraph: number;
}

/**
 * Every independent source an admitted association/activation item's
 * underlying evidence traces back to. Present only for that item kind, and
 * only when it has at least one citation.
 */
export interface Corroboration {
  sources: string[];
  attributions: CitationRef[];
}

/**
 * One admitted piece of evidence from `assembleEvidence` (ADR 0006 §10).
 * Exactly one of `association`/`passage`/`community` is set, selected by
 * `kind` — embedding the *existing* wire type verbatim rather than a
 * parallel evidence-only type. `bytes`/`estimated_tokens` are this item's own
 * content-only contribution, excluding those two fields themselves.
 */
export interface EvidenceItem {
  candidate_id: string;
  kind: Open<"association" | "passage" | "community">;
  fused_rank: number;
  lane_ranks: LaneRank[];
  citation_refs: CitationRef[];
  corroboration?: Corroboration;
  /**
   * The `candidate_id` of every other item in this item's contradiction
   * group (ADR 0006 §9) — omitted when it contradicts nothing.
   */
  contradicts?: string[];
  bytes: number;
  estimated_tokens: number;
  association?: Association;
  passage?: PassageHit;
  community?: CommunityHit;
}

/**
 * One resolved citation, keyed by `(source, paragraph)`, shared by every
 * `EvidenceItem` whose `citation_refs` names it.
 */
export interface CitationEntry {
  source: string;
  paragraph: number;
  citation: Citation;
}

/** The three resolved, independent hard ceilings a call ran under. */
export interface BudgetLimits {
  max_items: number;
  max_bytes: number;
  max_tokens: number;
}

/** The three-budget account after selection finished. */
export interface BudgetUsage {
  items_used: number;
  bytes_used: number;
  tokens_used: number;
  limits: BudgetLimits;
}

/**
 * A candidate that did not make it into `items`. `duplicate_of` is present
 * only for `reason: "duplicate_passage"` and names the *surviving*
 * `candidate_id`, never a locator.
 */
export interface OmittedCandidate {
  candidate_id: string;
  kind: Open<"association" | "passage" | "community">;
  reason: Open<"duplicate_passage" | "budget_exceeded" | "contradiction_group_exceeds_budget">;
  duplicate_of?: string;
}

/**
 * One `LanePlan` per retrieval lane `assembleEvidence` fans out to — the
 * same shape `searchPassages`'s own `plan.contexts[].lanes` already uses.
 */
export interface EvidenceLanesPlan {
  resolve: LanePlan;
  query: LanePlan;
  activate: LanePlan;
  passages: LanePlan;
  communities: LanePlan;
  citations: LanePlan;
}

/**
 * The selection pipeline's own account of what it did, beyond the
 * per-item/per-omission detail.
 */
export interface SelectionPlan {
  dedup_dropped: number;
  contradiction_groups: number;
  diversity_tier_width: number;
}

/**
 * Whether a reranker is configured, whether it actually ran for this call,
 * its model identity on success, and a machine-readable `reason` token on
 * any degrade (never a non-2xx error).
 */
export interface RerankerPlan {
  configured: boolean;
  ran: boolean;
  model?: string;
  reason?: Open<
    | "not_configured"
    | "model_mismatch"
    | "empty_pool"
    | "invalid_permutation"
    | "circuit_open"
    | "timeout"
    | "provider_error"
  >;
}

/** Everything `assembleEvidence`'s response carries beyond the selected items/citations/budget themselves. */
export interface EvidencePlan {
  lanes: EvidenceLanesPlan;
  selection: SelectionPlan;
  reranker: RerankerPlan;
}

/**
 * The full result of one `assembleEvidence` call (ADR 0006 §10). `omitted`
 * is capped like other issue lists; `omitted_total`/`omitted_by_reason` are
 * always exact, so truncation of the itemized list is never silent.
 */
export interface EvidencePackage {
  items: EvidenceItem[];
  citations: CitationEntry[];
  budget: BudgetUsage;
  omitted: OmittedCandidate[];
  omitted_total: number;
  omitted_by_reason: Record<string, number>;
  plan: EvidencePlan;
}
