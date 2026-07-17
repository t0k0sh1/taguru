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

/** One name or an OR-set of names, for query() positions. */
export type OneOrMany = string | string[];

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

export interface DirectoryEntry {
  name: string;
  description: string;
  pinned: boolean;
  loaded: boolean;
  dice_floor: number | null;
  semantic_floor: number | null;
  stats: ContextStats;
  usage: ContextUsage;
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
}

/** One page of the group directory. `total` is the whole population. */
export interface GroupPage {
  total: number;
  groups: GroupEntry[];
}

// -- graph shapes ---------------------------------------------------------------

/** One source's contribution to an association. `weight` is raw cumulative. */
export interface Attribution {
  source: string;
  weight: number;
  count: number;
  paragraph: number | null;
  section: string | null;
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

/** Ranked matches. `total` above `matches.length` means truncation. */
export interface MatchPage {
  total: number;
  matches: Association[];
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
 * on score alone; read `gloss` first.
 */
export interface TieredResolution {
  name: string;
  score: number;
  tier: "lexical" | "semantic";
  kind?: "exact" | "alias" | "containment" | "fuzzy";
  gloss?: string;
}

// -- resolve explain -----------------------------------------------------------

/**
 * One stored spelling near a cue. `kind` names the lexical relation — never
 * adopt a containment/fuzzy neighbor as the fix without reading the gloss.
 */
export interface NearestResolution {
  name: string;
  score: number;
  kind: "exact" | "alias" | "containment" | "fuzzy";
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
  kind?: "exact" | "alias" | "containment" | "fuzzy";
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
  tier?: "lexical" | "semantic";
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
  verdict:
    | "not_in_vocabulary"
    | "served"
    | "cue_resolved_exactly"
    | "below_floor"
    | "below_cutoff"
    | "semantic_not_run"
    | "semantic_below_floor";
  summary: string;
  cue: string;
  expected: string;
  in_vocabulary: boolean;
  canonical?: string;
  expected_kind?: "exact" | "alias";
  lexical?: LexicalExplain;
  semantic?: SemanticExplain;
  ranking?: ResolveRanking;
  nearest?: NearestSpellings;
}

export interface ConceptDescription {
  concept: string;
  as_subject: LabelUsage[];
  as_object: LabelUsage[];
}

export interface LabelPage {
  total: number;
  labels: string[];
}

// -- aliases ---------------------------------------------------------------------

/** One page of aliases; the cursor spans both namespaces (concepts first). */
export interface AliasPage {
  total: number;
  concepts: Record<string, string>;
  labels: Record<string, string>;
}

/** A flattened alias row as yielded by `iterAliases`. */
export interface AliasEntry {
  namespace: "concept" | "label";
  alias: string;
  canonical: string;
}

// -- passages / sources ------------------------------------------------------------

export interface SourcePage {
  total: number;
  sources: string[];
}

/** A dropped question/section named a paragraph its passage's split lacks. */
export interface StoredPassages {
  stored: number;
  questions_stored: number;
  questions_dropped: number;
  sections_stored: number;
  sections_dropped: number;
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
  verdict:
    | "not_stored"
    | "paragraph_out_of_range"
    | "no_query_terms"
    | "no_term_overlap"
    | "below_cutoff"
    | "served";
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

/** One verbatim paragraph. `section` is null outside every stored section. */
export interface Citation {
  text: string;
  source: string;
  section: string | null;
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
  association_paragraphs_dropped: number;
}

/**
 * Outcome of restoring one `taguru_group` record via import. A restore is a
 * replace of the whole record; `outcome` says what it replaced.
 */
export interface GroupImportOutcome {
  name: string;
  outcome: "created" | "replaced" | "unchanged";
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
}

/**
 * The key `RetrievalResult.citations` is indexed by. NUL-delimited so no
 * source id can collide with another (source, paragraph) pair — always build
 * keys with this helper rather than by hand.
 */
export function citationKey(source: string, paragraph: number): string {
  return `${source}\u0000${paragraph}`;
}
