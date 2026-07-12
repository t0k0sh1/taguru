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
